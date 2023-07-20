import sys

sys.argv = ['']
del sys

import numpy as np
import math
from collections import defaultdict
import argparse
import argparse
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, TensorDataset, random_split
import copy

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# from VIBImodels import ResNet, resnet18, resnet34, Unet

# from debug import debug
import torch.nn as nn
import torch.optim

import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--sampling', type=float, default=1.0, help="random sampling (default: 1.0)")
    parser.add_argument('--epsilon', type=float, default=1.0, help="DP epsilon (default: 1.0)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    # parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='need auxiliary data or not for non iid')
    parser.add_argument('--add_noise', action='store_true', default=False,  help='need add noise or not for non iid')


    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='CIFAR10')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=3)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0.001, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)

    args = parser.parse_args()

    return args

def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=None):
        super().__init__()
        stride = stride or (1 if in_channels >= out_channels else 2)
        self.block = conv_block(in_channels, out_channels, stride)
        if stride == 1 and in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))


class ResNet(nn.Module):
    def __init__(self, in_channels, block_features, num_classes=10, headless=False):
        super().__init__()
        block_features = [block_features[0]] + block_features + ([num_classes] if headless else [])
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, block_features[0], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(block_features[0]),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(block_features[i], block_features[i + 1])
            for i in range(len(block_features) - 1)
        ])
        self.linear_head = None if headless else nn.Linear(block_features[-1], num_classes)

    def forward(self, x):
        x = self.expand(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        if self.linear_head is not None:
            x = F.avg_pool2d(x, x.shape[-1])  # completely reduce spatial dimension
            x = self.linear_head(x.reshape(x.shape[0], -1))
        return x


def resnet18(in_channels, num_classes):
    # block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    block_features = [64] * 3 + [128] * 4 + [256] * 5
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


class Unet(nn.Module):
    def __init__(self, in_channels, down_features, num_classes, pooling=False):
        super().__init__()
        self.expand = conv_block(in_channels, down_features[0])

        self.pooling = pooling

        down_stride = 1 if pooling else 2
        self.downs = nn.ModuleList([
            conv_block(ins, outs, stride=down_stride) for ins, outs in zip(down_features, down_features[1:])])

        up_features = down_features[::-1]
        self.ups = nn.ModuleList([
            conv_block(ins + outs, outs) for ins, outs in zip(up_features, up_features[1:])])

        self.final_conv = nn.Conv2d(down_features[0], num_classes, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.expand(x)

        x_skips = []

        for down in self.downs:
            x_skips.append(x)
            x = down(x)
            if self.pooling:
                x = F.max_pool2d(x, 2)

        for up, x_skip in zip(self.ups, reversed(x_skips)):
            x = torch.cat([self.upsample(x), x_skip], dim=1)
            x = up(x)

        x = self.final_conv(x)

        return x




@torch.no_grad()
def test_accuracy(model, data_loader,FL_model, net_global, org_net_glob, device, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    FL_model.eval()
    num_correct2 = 0
    num_correct3 = 0
    num_correct_global = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        logits_z, out, p_z , outy_2= model(x, mode='distribution')
        B, C, H, W = x.shape
        pz2 = p_z.reshape((B, -1))
        out2 = FL_model(pz2.detach())
        out_res = net_global(p_z.detach())
        B,C,H,W = x.shape
        x_for_fl_org = x.reshape((B, -1))
        out3 = org_net_glob(x)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_correct2 +=(out2.argmax(dim=1)==y).sum().item()
        num_correct3 +=(out3.argmax(dim=1)==y).sum().item()
        num_correct_global += (outy_2.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc2 = num_correct2/num_total
    acc3 = num_correct3 / num_total
    acc_global = num_correct_global/num_total
    # print(f'{name} accuracy: {acc:.3f}',f'{name} accuracy2: {acc2:.3f}',f'{name} accuracy3 global: {acc_global:.3f}',f'{name} org_net_glob : {acc3:.3f}')
    return acc


def num_params(model):
    num = 0
    for name, parms in model.named_parameters():
        num += len(parms.grad.reshape(-1))

    return num #sum([p.numel() for p in model.parameters() if p.requires_grad])


def sample_gumbel(size):
    return -torch.log(-torch.log(torch.rand(size)))


def gumbel_reparametrize(log_p, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (B, num_samples, C)
    g = sample_gumbel(shape).to(log_p.device)  # (B, N, C)
    return F.softmax((log_p.unsqueeze(1) + g) / temp, dim=-1)  # (B, N, C)


# this is only a, at most k-hot relaxation
def k_hot_relaxed(log_p, k, temp, num_samples):
    assert log_p.ndim == 2
    B, C = log_p.shape  # (B, C)
    shape = (k, B, C)
    k_log_p = log_p.unsqueeze(0).expand(shape).reshape((k * B, C))  # (k* B, C)
    k_hot = gumbel_reparametrize(k_log_p, temp, num_samples)  # (k* B, N, C)
    k_hot = k_hot.reshape((k, B, num_samples, C))  # (k, B, N, C)
    k_hot, _ = k_hot.max(dim=0)  # (B, N, C)
    return k_hot  # (B, N, C)


# needed for when labels are not one-hot
def soft_cross_entropy_loss(logits, y):
    return -(y * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()

class LinearCIFAR(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3*32, h_dim2=32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearCIFAR, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, h_dim2)  # mu
        self.fc3 = nn.Linear(h_dim2, n_output)  # log_var

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 给x加权成为a，用激励函数将a变成特征b
        x = self.fc3(x)
        return x

class VIBI(nn.Module):
    def __init__(self, explainer, approximator, approximator2, reconstructor, k=4, num_samples=4, temp=1):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.approximator2 = approximator2
        self.reconstructor = reconstructor
        self.k = k
        self.temp = temp
        self.num_samples = num_samples

        self.warmup = False

    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """

        k = self.k
        temp = self.temp
        N = num_samples or self.num_samples

        B, C, H, W = x.shape

        logits_z = self.explainer(x)  # (B, C, h, w)
        B, C, h, w = logits_z.shape
        #         print("B, C, H, W",B, C, H, W)
        logits_z = logits_z.reshape((B, -1))  # (B, C* h* w)

        if mode == 'distribution':  # return the distribution over explanation
            p_z = F.softmax(logits_z, dim=1).reshape((B, C, h, w))  # (B, C, h, w)
            p_z_upsampled = F.interpolate(p_z, (H, W), mode='nearest')  # (B, C, H, W)
            return p_z_upsampled, logits_z, p_z
        elif mode == 'test':
            p_z = F.softmax(logits_z, dim=1).reshape((B, C, h, w))  # (B, C, h, w)
            p_z_upsampled = F.interpolate(p_z, (H, W), mode='nearest')  # (B, C, H, W)
            return p_z_upsampled
        elif mode == 'topk':  # return top k pixels from input
            k_hot_z = k_hot_relaxed(logits_z, k, temp, N)  # (B, N, F)
            k_hot_z = k_hot_z.reshape((B * N, C, h, w))  # (B* N, C, h, w)

            k_hot_z_upsampled = F.interpolate(k_hot_z, (H, W), mode='nearest')  # (B* N, C, H, W)
            k_hot_z_upsampled = k_hot_z_upsampled.reshape((B, N, C, H, W))  # (B, N, C, H, W)
            return k_hot_z_upsampled
        elif mode == 'sample':  # return (not always) k hot vector over pixels sampled from p_z
            k_hot_z = k_hot_relaxed(logits_z, k, temp, N)  # (B, N, F)
            k_hot_z = k_hot_z.reshape((B * N, C, h, w))  # (B* N, C, h, w)

            k_hot_z_upsampled = F.interpolate(k_hot_z, (H, W), mode='nearest')  # (B* N, C, H, W)
            k_hot_z_upsampled = k_hot_z_upsampled.reshape((B, N, C, H, W))  # (B, N, C, H, W)
            return k_hot_z_upsampled, logits_z
        elif mode == 'warmup':
            return logits_z

    def forward(self, x, mode='topk'):

        N = self.num_samples
        B, C, H, W = x.shape
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            z_upsampled, logits_z_flat, p_z = self.explain(x, mode=mode)  # (B, C, H, W), (B, C* h* w)
            #             print("t", t.shape)                                                 # (B* N, C, H, W)
            logits_y = self.approximator(z_upsampled)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            p_z2 = p_z.reshape((B, -1))
            logits_y2 = self.approximator2(p_z)
            return logits_z_flat, logits_y, p_z, logits_y2
        elif mode == 'distribution_reconstruct':
            z_upsampled, logits_z_flat, p_z = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #             print("t", t.shape)                                                 # (B* N, C, H, W)
            logits_y = self.approximator(z_upsampled)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            p_z2 = p_z.reshape((B, -1))
            logits_y2 = self.approximator2(p_z)
            x_hat = self.reconstructor(p_z)
            return logits_z_flat, logits_y, p_z, logits_y2, x_hat
        elif mode == 'sample':
            k_hot_z_upsampled, logits_z_flat = self.explain(x, mode=mode)  # (B, N, C, H, W), (B, C* h* w)
            #             print("logits_z_flat", logits_z_flat.shape)
            #             print("k_hot_z_upsampled", k_hot_z_upsampled.shape)
            #             t = x.unsqueeze(1) * k_hot_z_upsampled                              # (B, N, C, H, W)
            #             print("k_hot_z_upsampled",k_hot_z_upsampled.shape)
            #             print("k_hot_z_upsampled mean",torch.mean(k_hot_z_upsampled, dim=1).shape)
            x_uns = x.unsqueeze(1)
            #             print("x_uns", x_uns.shape)
            t = k_hot_z_upsampled.reshape((B * N, C, H, W))  # (B* N, C, H, W)
            #             print("t", t.shape)                                                 # (B* N, C, H, W)
            logits_y = self.approximator(t)  # (B* N, 10)
            logits_y = logits_y.reshape((B, N, 10))  # (B, N, 10)
            return logits_z_flat, logits_y
        elif mode == 'topk':
            k_hot_z_upsampled = self.explain(x, mode=mode)  # (B, N, C, H, W)
            # t = x * k_hot_z_upsampled                                           # (B, C, H, W)
            # logits_y = self.approximator(t)                                     # (B, 10)
            # print("k_hot_z_upsampled",k_hot_z_upsampled.shape)
            t = torch.mean(k_hot_z_upsampled, dim=1)
            logits_y = self.approximator(t)
            return logits_y
        elif mode == 'warmup':
            logits_z_dummy = torch.log(torch.ones_like(x) / x.numel())  # (B, C, H, W)
            logits_z_dummy = logits_z_dummy.reshape((B, -1))  # (B, C* H* W)
            logits_y = self.approximator(x)  # (B, 10)
            return logits_z_dummy, logits_y
        elif mode == 'test':
            z_upsampled = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(z_upsampled)
            return logits_y



def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    print('dict_users', len(dict_users))
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 2*num_users , int(len(dataset)/(2*num_users)) #2*
    # num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def auxiliary(dict_users, num_users):
    each_user_has = len(dict_users[0])
    each_put = int (each_user_has / num_users)
    dict_users[num_users] = []
    for i in range(num_users):
        rand_set = list(set(np.random.choice(dict_users[i], each_put, replace=False)))
        dict_users[num_users] = np.concatenate((dict_users[num_users], rand_set), axis=0).astype(int)
    return dict_users, num_users+1


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users*2 , int(len(dataset)/(2*num_users))
    # num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array( dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users




class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class LinearModel(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output)  # output

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LinearMNIST(nn.Module):
    # 定义神经网络
    def __init__(self, n_feature=256, h_dim=1024, h_dim2=400, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(LinearMNIST, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, h_dim2)  # mu
        self.fc3 = nn.Linear(h_dim2, n_output)  # log_var

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 给x加权成为a，用激励函数将a变成特征b
        x = self.fc3(x)
        return  F.log_softmax(x, dim=1)
    # 给b加权，预测最终结果

class CNNMnist(nn.Module):
    # def __init__(self, args):
    #     super(CNNMnist, self).__init__()
    #     self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
    #     self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    #     self.conv2_drop = nn.Dropout2d()
    #     self.fc1 = nn.Linear(320, 50)
    #     self.fc2 = nn.Linear(50, args.num_classes)

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
    #     x = F.relu(self.fc1(x))
    #     x = F.dropout(x, training=self.training)
    #     x = self.fc2(x)
    #     return x
    def __init__(self , args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, args.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, sampling):
        self.dataset = dataset
        #         self.idxs = list(idxs)
        #         self.idxs = random.sample(list(idxs), int(len(idxs)*sampling))
        if sampling == 1:
            self.idxs = list(idxs)
        else:
            self.idxs = np.random.choice(list(idxs), size=int(len(idxs) * sampling), replace=True)
            #self.idxs = random.sample(list(idxs), int(len(idxs) * sampling)) # without replacement
            # random.choice is with replacement
        # print('datasplite' , idxs, len(dataset))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def add_noise(data, epsilon, sensitivity, args):
    # np_data = data.numpy()
    # print(data.shape)
    # print(np_data.shape)
    noise_tesnor = np.random.laplace(1, sensitivity/epsilon,data.shape) * args.lr
    # data = torch.add(data, torch.from_numpy(noise_tesnor))
    # for x in np.nditer(np_data, op_flags=['readwrite']):
    #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
    if args.gpu == -1:
        return data.add(torch.from_numpy(noise_tesnor).float())
    else:
        return data.add(torch.from_numpy(noise_tesnor).float().cuda())

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # prepare the local original dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args.sampling), batch_size=self.args.local_bs, shuffle=True)

        # local data compressing
        # self.com

    def add_noise(data, epsilon, sensitivity, args):
        # np_data = data.numpy()
        # print(data.shape)
        # print(np_data.shape)
        noise_tesnor = np.random.laplace(1, sensitivity / epsilon, data.shape)
        # data = torch.add(data, torch.from_numpy(noise_tesnor))
        # for x in np.nditer(np_data, op_flags=['readwrite']):
        #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
        if args.gpu==-1:
            return data.add(torch.from_numpy(noise_tesnor).float())
        else:
            return data.add(torch.from_numpy(noise_tesnor).float().cuda())

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('batch_idx', batch_idx, images.shape)
                if self.args.add_noise:
                    images =LocalUpdate.add_noise( data=images, epsilon=self.args.epsilon, sensitivity=1,args=self.args)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                # print('batch_idx', batch_idx)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def trainZ(self, net, Zdata,idx):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print(iter)
            for batch_idx, (images, labels) in enumerate(Zdata):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('batch_idx', batch_idx, images.shape)
                # if self.args.add_noise:
                #     images =LocalUpdate.add_noise( data=images, epsilon=self.args.epsilon, sensitivity=1,args=self.args)

                net.zero_grad()
                log_probs = net(images.detach())

                # y_not_loss = torch.zeros(1).cuda()
                # for i in range(1, 10):
                #     y_fortest = (labels + i) % 10
                #     temp_cross_entropy_loss = torch.mean(self.loss_func(log_probs, y_fortest), dim=0).cuda()
                #     y_not_loss = y_not_loss + i / temp_cross_entropy_loss
                loss = self.loss_func(log_probs, labels)  #+ y_not_loss/50
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                # print('batch_idx', batch_idx)
                fl_acc = (log_probs.argmax(dim=1) == labels).float().mean().item()
                # print('idx=idx', idx, images)
                # print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #           100. * batch_idx / len(self.ldr_train), loss.item()))
                if batch_idx%10==1:
                    print('fl_acc:', fl_acc)

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def train_resnet18(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            # print('iter', iter)
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(batch_idx)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print('batch_idx', batch_idx, images.shape)
                if self.args.add_noise:
                    images =LocalUpdate.add_noise( data=images, epsilon=0.8, sensitivity=1, args=self.args)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                # loss = F.nll_loss(log_probs, labels)
                loss.backward()
                optimizer.step()
                # print('batch_idx', batch_idx)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    epsilon = 1
    beta = 1 / epsilon
    data_len = 0
    for i in range(len(w)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(w) , data_len)
    for k in w_avg.keys():
        # print(k)
        for i in range(1, len(w)):
            # print('before',w[i][k][1])
            temp = torch.add(w[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            w_avg[k] += temp
            # print('after', temp[1])

        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.div(w_avg[k], data_len)

    return w_avg


def AvgCP(dict_usersZmodel):

    w = []
    for i in range(len(dict_usersZmodel)):
        w.append(copy.deepcopy(dict_usersZmodel[i].state_dict()))

    w_avg = copy.deepcopy(w[0])
    epsilon = 1
    beta = 1 / epsilon
    data_len = 0
    for i in range(len(w)):
        data_len += 1  # + np.random.laplace(0, beta, 1)[0]
    print('len_w', len(w) , data_len)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            # print('before',w[i][k][1])
            temp = torch.add(w[i][k], 0) # np.random.laplace(0, beta, 1)[0] / 100
            # w_avg[k] += w[i][k]
            # print(temp)
            w_avg[k] += temp
            # print('after', temp[1])

        # w_avg[k] = torch.div(w_avg[k], len(w))
        w_avg[k] = torch.div(w_avg[k], data_len)

    # print(dict_usersZmodel[0])
    return w_avg


def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5

def vibi_train(args, ldr_train, z_dim, user_idx, loss_avg, vibi, optimizer, FL_model, FL_optimizer, net_glob, XZ_Mutual, mnist_img, org_net_glob, org_resnet,reconstruction_function):

    # print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    device = 'cuda' if args.cuda else 'cpu'
    # print("device", device)
    dataset = args.dataset

    # train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)


    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if dataset == 'MNIST':
        approximator = resnet18(1, 10)
        xpl_channels = 1

        if explainer_type == 'ResNet_4x':
            block_features = [64] * 2 + [128] * 3 + [256] * 4
            explainer = ResNet(1, block_features, 1, headless=True)
        elif explainer_type == 'ResNet_2x':
            block_features = [64] * 10
            explainer = ResNet(1, block_features, 1, headless=True)
        elif explainer_type == 'Unet':
            explainer = Unet(1, [64, 128, 256], 1)
        else:
            raise ValueError
        lr = 0.05
        temp_warmup = 200

    elif dataset == 'CIFAR10':
        approximator = resnet18(3, 10)

        if explainer_type == 'ResNet_8x':
            block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        if explainer_type == 'ResNet_4x':
            block_features = [64] * 3 + [128] * 4 + [256] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'ResNet_2x':
            block_features = [64] * 4 + [128] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'Unet':
            explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
        else:
            raise ValueError

        lr = 0.005
        temp_warmup = 4000

    init_epoch = 0

    logs = defaultdict(list)

    # vibi = VIBI(explainer, approximator, k=k, num_samples=args.num_samples)
    # vibi.to(device)

    # optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)

    # valid_acc = test_accuracy(vibi, bb_valid_loader, name='vibi valid top1')

    valid_acc = 0.8
    loss_fn = nn.CrossEntropyLoss()


    fl_loss_fn = nn.CrossEntropyLoss()

    # FL_model_original = LinearCIFAR(n_feature=3 * 32 * 32, n_output=10)
    # FL_model_original.to(device)
    FL_optimizer_original = torch.optim.Adam(org_net_glob.parameters(), lr=lr)
    fl_org_loss_fn = nn.CrossEntropyLoss()

    # FL_resnet = resnet18(3, 10)
    # FL_resnet.to(device)
    FL_resnet_optimizer = torch.optim.Adam(org_resnet.parameters(), lr=lr)
    fl_resnet_loss_fn = nn.CrossEntropyLoss()

    temp_img = torch.empty(0, z_dim).float().cuda()
    temp_label = torch.empty(0).long().cuda()

    global_opt = torch.optim.Adam(net_glob.parameters(), lr=lr)
    global_fn = nn.CrossEntropyLoss()
    if init_epoch == 0 or args.resume_training:
        # print('Training VIBI')
        # print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
        # print(f'{type(approximator2).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')

        # inspect_explanations()

        for epoch in range(init_epoch, init_epoch + args.num_epochs):

            vibi.train()
            step_start = epoch * len(ldr_train)
            for step, (x, y) in enumerate(ldr_train, start=step_start):
                # print(user_idx, x.shape)
                t = step - temp_warmup
                vibi.temp = 10 / t if t > 1 else 10
                # warmup = t < 1
                x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)

                # if user_idx%2 == 1:
                #     x = torch.add(x, mnist_img)

                # note: in the case of upsampling, warmup logits_z have different shape: (.., H, W) instead of (.., h, w)
                # if warmup:
                #     # y = y.reshape(-1, 1, 10) # (B, 1, 10)
                #     logits_z, logits_y = vibi(x, mode='warmup')     # (B, C, H, W), (B, 10)
                # else:
                # y = y[:, None].expand(-1, vibi.num_samples)
                logits_z, logits_y, p_z ,logits_y2, x_hat = vibi(x, mode='distribution_reconstruct')  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)
                # logits_y = logits_y.permute(0, 2, 1)
                logits_z = logits_z.log_softmax(dim=1)
                B, C, H, W = x.shape
                pz2 = p_z.reshape((B, -1))
                # logits_y = logits_y.argmax(dim=1).float()
                H_p_q = loss_fn(logits_y, y)
                B, dimZ = pz2.shape
                mu = pz2[:, :dimZ]
                sigma = torch.nn.functional.softplus(pz2[:, :dimZ])
                encoder_Z_distr = mu, sigma
                prior_Z_distr = torch.zeros(B, dimZ).cuda(), torch.ones(B, dimZ).cuda()
                # print(mu.shape, sigma.shape, torch.zeros(B, dimZ).cuda().shape)
                I_ZX_bound = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
                KL_z_r = (torch.exp(logits_z) * logits_z).sum(dim=1).mean() + math.log(logits_z.shape[1])

                # if warmup:    # KL_z_r in warmup stage can kill signal
                #     loss = H_p_q
                # else:

                # global_y = net_glob(pz2.detach())
                logits_y2 = vibi.approximator2(p_z)
                global_loss = global_fn(logits_y2, y)

                x_hat = x_hat.view(x_hat.size(0), -1)
                x_view = x.view(x.size(0), -1)
                # x = torch.sigmoid(torch.relu(x))
                BCE = reconstruction_function(x_hat, x_view)  # mse loss

                loss = 0*H_p_q + args.beta * I_ZX_bound + global_loss + BCE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                if epoch == args.num_epochs - 1:
                    temp_img = torch.cat([temp_img, pz2], dim=0)
                    temp_label = torch.cat([temp_label, y], dim=0)

                fl_y = FL_model(pz2.detach())
                fl_loss = fl_loss_fn(fl_y, y)
                FL_optimizer.zero_grad()
                fl_loss.backward()
                FL_optimizer.step()



                B, C, H, W = x.shape
                x_for_fl_org = x.reshape((B, -1))

                fl_org_y = org_net_glob(x)
                fl_org_loss = fl_org_loss_fn(fl_org_y, y)
                FL_optimizer_original.zero_grad()
                fl_org_loss.backward()
                FL_optimizer_original.step()
                # if not warmup:  # logits_y contain 'num_sample' samples (B, N, 10)
                #     logits_y = logits_y.mean(dim=1)
                #     y = y.squeeze()

                # fl_res_Y = org_resnet(x)
                # fl_res_loss = fl_resnet_loss_fn(fl_res_Y, y)
                # FL_resnet_optimizer.zero_grad()
                # fl_res_loss.backward()
                # FL_resnet_optimizer.step()

                acc = (logits_y.argmax(dim=1) == y).float().mean().item()
                # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
                fl_acc = (fl_y.argmax(dim=1) == y).float().mean().item()
                fl_org_acc = (fl_org_y.argmax(dim=1) == y).float().mean().item()
                # fl_res_acc = (fl_res_Y.argmax(dim=1) == y).float().mean().item()
                global_acc = (logits_y2.argmax(dim=1)==y).float().mean().item()
                metrics = {
                    'user_idx': user_idx,
                    # 'fl_org_res': fl_res_acc,
                    'fl_org_acc': fl_org_acc,
                    'global_y': global_acc,
                    'fl_acc': fl_acc,
                    'acc': acc,
                    'loss': loss.item(),
                    'temp': vibi.temp,
                    'H(p,q)': H_p_q.item(),
                    # '1-JS(p,q)': JS_p_q,
                    'KL(z||r)': KL_z_r.item(),
                    'I_ZX_bound': I_ZX_bound,
                    'BCE': BCE.item(),
                }

                for m, v in metrics.items():
                    logs[m].append(v)

                if step % len(ldr_train) % 50 == 0:
                    print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(ldr_train):3d}] '
                          + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            XZ_Mutual.append(I_ZX_bound.item())
            vibi.eval()
            valid_acc_old = valid_acc
            valid_acc = test_accuracy(vibi, ldr_train, FL_model, net_glob, org_net_glob, device, name='vibi valid top1')
            interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(ldr_train)).tolist()
            logs['val_acc'].extend(interpolate_valid_acc)
            # print("test_acc", valid_acc)
    b = Data.TensorDataset(temp_img, temp_label)
    lr_train_dl = DataLoader(b, batch_size=args.local_bs, shuffle=True)

    print(f'explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'approximator2 params:\t{num_params(vibi.approximator) / 1000:.2f} K')
    print(f'org_net_glob params:\t{num_params(org_net_glob) / 1000:.2f} K')

    return lr_train_dl, vibi, XZ_Mutual, vibi.approximator2.state_dict(), org_net_glob.state_dict(), org_net_glob #, org_resnet.state_dict()

def acc_evaluation(net_glob, dataset_test, args, Zmodel):
    acc_test, loss_test = testZ_img(net_glob, dataset_test, args, Zmodel)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test

def acc_evaluation_org(org_net_glob, dataset_test, args):
    acc_test, loss_test = test_img(org_net_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test

def testZ_img(net_g, datatest, args, Zmodel):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    Zmodel.eval()
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        # x_batch = data.reshape([len(data), 28 * 28])
        # logits, x_hat, z_mean = Zmodel.forward(data)
        # data = data.reshape((args.bs, 3,32,32))
        logits_z, logits_y, p_z, logits_y2 = Zmodel(data, mode='distribution')
        B,c,h,w = p_z.shape

        # print(z_mean2)
        log_probs = net_g(p_z.detach())
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # print(y_pred, target.data.view_as(y_pred))
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()

        # B,c,h,w = data.shape
        # data2 = data.reshape((B, -1))
        # print(z_mean2)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # print(y_pred, target.data.view_as(y_pred))
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss


def init_cp_model(args):
    k = args.k
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if dataset == 'MNIST':
        approximator = resnet18(1, 10)
        xpl_channels = 1
        if explainer_type == 'ResNet_4x':
            block_features = [64] * 2 + [128] * 3 + [256] * 4
            explainer = ResNet(1, block_features, 1, headless=True)
        elif explainer_type == 'ResNet_2x':
            block_features = [64] * 10
            explainer = ResNet(1, block_features, 1, headless=True)
        elif explainer_type == 'Unet':
            explainer = Unet(1, [64, 128, 256], 1)
        else:
            raise ValueError
        lr = 0.05
        temp_warmup = 200

    elif dataset == 'CIFAR10':
        approximator = resnet18(3, 10)
        approximator2 = resnet18(3, 10)
        reconstructor = resnet18(3, 3 * 32 * 32)  # LinearModel(n_feature=49, n_output=3 * 32 * 32)

        if explainer_type == 'ResNet_8x':
            block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        if explainer_type == 'ResNet_4x':
            block_features = [64] * 3 + [128] * 4 + [256] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'ResNet_2x':
            block_features = [64] * 4 + [128] * 5
            explainer = ResNet(3, block_features, xpl_channels, headless=True)
        elif explainer_type == 'Unet':
            explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
        else:
            raise ValueError

        lr = 0.005



    vibi = VIBI(explainer, approximator, approximator2, reconstructor, k=k, num_samples=args.num_samples)
    vibi.to(device)
    optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)

    return vibi, optimizer


def infer_model_prepare(args, mnist_img):

    device = 'cuda' if args.cuda else 'cpu'
    print("device", device)
    dataset = args.dataset

    # train_loader_full = DataLoader(train_set_no_aug, batch_size=200, shuffle=True, num_workers=1)


    infer_test_set = CIFAR10('./data/cifar', train=False,
                       transform=transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32])]),
                       download=True)


    infer_train_loader = DataLoader(infer_test_set, batch_size=args.local_bs, shuffle=True, num_workers=1)
    infer_test_loader = DataLoader(infer_test_set, batch_size=args.local_bs, shuffle=False, num_workers=1)

    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    approximator = resnet18(3, 10)
    approximator_wm = resnet18(3, 10)
    if explainer_type == 'ResNet_8x':
        block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        explainer_wm = ResNet(3, block_features, xpl_channels, headless=True)
    if explainer_type == 'ResNet_4x':
        block_features = [64] * 3 + [128] * 4 + [256] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        explainer_wm = ResNet(3, block_features, xpl_channels, headless=True)
    elif explainer_type == 'ResNet_2x':
        block_features = [64] * 4 + [128] * 5
        explainer = ResNet(3, block_features, xpl_channels, headless=True)
        explainer_wm = ResNet(3, block_features, xpl_channels, headless=True)
    elif explainer_type == 'Unet':
        explainer = Unet(3, [64, 128, 256, 512], xpl_channels)
    else:
        raise ValueError

    lr = 0.005
    temp_warmup = 4000

    init_epoch = 0
    logs = defaultdict(list)

    FL_model = LinearCIFAR(n_feature=192, n_output=10)
    FL_model.to(device)
    FL_optimizer = torch.optim.Adam(FL_model.parameters(), lr=lr)
    fl_loss_fn = nn.CrossEntropyLoss()

    FL_model_wm = LinearCIFAR(n_feature=192, n_output=10)
    FL_model_wm.to(device)
    FL_optimizer_wm = torch.optim.Adam(FL_model_wm.parameters(), lr=lr)
    fl_loss_fn_wm = nn.CrossEntropyLoss()

    reconstructor = resnet18(3,3*32*32)
    vibi = VIBI(explainer, approximator, FL_model, reconstructor,  k=k, num_samples=args.num_samples)
    vibi.to(device)
    valid_acc = 0.8
    optimizer = torch.optim.Adam(vibi.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()


    vibi_wm = VIBI(explainer_wm, approximator_wm, FL_model_wm, reconstructor,  k=k, num_samples=args.num_samples)
    vibi_wm.to(device)
    vm_vibi_optimizer = torch.optim.Adam(vibi_wm.parameters(), lr=lr)
    wm_loss_fn = nn.CrossEntropyLoss()

    # valid_acc = test_accuracy(vibi, bb_valid_loader, name='vibi valid top1')



    FL_model_original = LinearCIFAR(n_feature=3 * 32 * 32, n_output=10)
    FL_model_original.to(device)
    FL_optimizer_original = torch.optim.Adam(FL_model_original.parameters(), lr=lr)
    fl_org_loss_fn = nn.CrossEntropyLoss()

    fl_with_mnist = LinearCIFAR(n_feature=3 * 32 * 32, n_output=10)
    fl_with_mnist.to(device)
    FL_with_mnist_optimizer = torch.optim.Adam(fl_with_mnist.parameters(), lr=lr)
    fl_with_mnist_loss_fn = nn.CrossEntropyLoss()

    #infer model for cp and org
    cp_infer = LinearCIFAR(n_feature=3*32 * 32, n_output=2)
    cp_infer.to(device)
    cp_infer_optimizer = torch.optim.Adam(cp_infer.parameters(), lr=lr)
    cp_infer_loss_fn = nn.CrossEntropyLoss()

    org_infer = LinearCIFAR(n_feature=3*32 * 32, n_output=2)
    org_infer.to(device)
    org_infer_optimizer = torch.optim.Adam(org_infer.parameters(), lr=lr)
    org_infer_loss_fn = nn.CrossEntropyLoss()

    print('Training VIBI')
    print(f'{explainer_type:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
    print(f'{type(approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')


    for epoch in range(10):
        vibi.train()
        vibi_wm.train()

        temp_cp_mode = torch.empty(0, 3*32 * 32).float().cuda()
        temp_label = torch.empty(0).long().cuda()
        temp_org_model = torch.empty(0, 3*32 * 32).float().cuda()
        for step, (x, y) in enumerate(infer_train_loader):  # enumerate(train_loader, start=step_start)
            # if step > 200: continue
            x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)

            x_conbime = torch.add(x, mnist_img)
            t = step - temp_warmup
            vibi.temp = 10 / t if t > 1 else 10

            logits_z, logits_y, p_z, logits_y2 = vibi(x, mode='distribution')  # (B, C* h* w), (B, N, 10), (B,N,C, h, w)
            # logits_y = logits_y.permute(0, 2, 1)
            logits_z = logits_z.log_softmax(dim=1)
            B, C, H, W = x.shape
            pz2 = p_z.reshape((B, -1))
            # logits_y = logits_y.argmax(dim=1).float()
            H_p_q = loss_fn(logits_y, y)
            B, dimZ = pz2.shape
            mu = pz2[:, :dimZ]
            sigma = torch.nn.functional.softplus(pz2[:, :dimZ])
            encoder_Z_distr = mu, sigma
            prior_Z_distr = torch.zeros(B, dimZ).cuda(), torch.ones(B, dimZ).cuda()
            # print(mu.shape, sigma.shape, torch.zeros(B, dimZ).cuda().shape)
            I_ZX_bound = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
            KL_z_r = (torch.exp(logits_z) * logits_z).sum(dim=1).mean() + math.log(logits_z.shape[1])


            fl_loss = fl_loss_fn(logits_y2, y)
            loss = H_p_q + beta * I_ZX_bound + fl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### with mnist vibi
            logits_z_wm, logits_y_wm, p_z_wm, logits_y2_wm = vibi_wm(x_conbime, mode='distribution')
            # logits_y = logits_y.permute(0, 2, 1)
            logits_z_wm = logits_z_wm.log_softmax(dim=1)
            pz2_wm = p_z_wm.reshape((B, -1))
            # logits_y = logits_y.argmax(dim=1).float()
            H_p_q_wm = wm_loss_fn(logits_y_wm, y)
            mu_wm = pz2_wm[:, :dimZ]
            sigma_wm = torch.nn.functional.softplus(pz2_wm[:, :dimZ])
            encoder_Z_distr_wm = mu_wm, sigma_wm
            prior_Z_distr_wm = torch.zeros(B, dimZ).cuda(), torch.ones(B, dimZ).cuda()
            I_ZX_bound_wm = torch.mean(KL_between_normals(encoder_Z_distr_wm, prior_Z_distr_wm))

            fl_loss_wm = fl_loss_fn_wm(logits_y2_wm, y)
            loss_vm = H_p_q_wm + beta * I_ZX_bound_wm + fl_loss_wm
            vm_vibi_optimizer.zero_grad()
            loss_vm.backward()
            vm_vibi_optimizer.step()

            # fl_y = FL_model(pz2.detach())

            # FL_optimizer.zero_grad()
            # fl_loss.backward()
            # FL_optimizer.step()
            #
            # fl_y_wm = FL_model_wm(pz2_wm.detach())

            # FL_optimizer_wm.zero_grad()
            # fl_loss_wm.backward()
            # FL_optimizer_wm.step()

            y_add_digit = torch.ones(1).cuda().reshape((1, -1))
            y_no_add_digit = torch.zeros(1).cuda().reshape((1, -1))
            i = 0
            for name, parms in vibi_wm.approximator2.named_parameters():
                # print(name, parms.grad)
                i += 1
                if i <= 2: continue
                temp_cp_mode = torch.cat([temp_cp_mode, parms.grad.reshape((1, -1))], dim=0)
                temp_label = torch.cat([temp_label, y_add_digit], dim=0)
                if i == 3: break

            i = 0
            for name, parms in vibi.approximator2.named_parameters():
                i += 1
                if i <= 2: continue
                temp_cp_mode = torch.cat([temp_cp_mode, parms.grad.reshape((1, -1))], dim=0)
                temp_label = torch.cat([temp_label, y_no_add_digit], dim=0)
                if i == 3: break

            B, C, H, W = x_conbime.shape
            x_with_mnist_fl_org = x_conbime.reshape((B, -1))

            fl_wm_y = fl_with_mnist(x_with_mnist_fl_org)
            fl_wm_loss = fl_with_mnist_loss_fn(fl_wm_y, y)
            FL_with_mnist_optimizer.zero_grad()
            fl_wm_loss.backward()
            FL_with_mnist_optimizer.step()


            x_for_fl_org = x.reshape((B, -1))
            fl_org_y = FL_model_original(x_for_fl_org)
            fl_org_loss = fl_org_loss_fn(fl_org_y, y)
            FL_optimizer_original.zero_grad()
            fl_org_loss.backward()
            FL_optimizer_original.step()

            i = 0
            for name, parms in fl_with_mnist.named_parameters():
                i += 1
                if i <= 2: continue
                temp_org_model = torch.cat([temp_org_model, parms.grad.reshape((1, -1))], dim=0)
                # temp_label = torch.cat([temp_label, y_add_digit], dim=0)
                if i == 3: break

            i = 0
            for name, parms in FL_model_original.named_parameters():
                i += 1
                if i <= 2: continue
                temp_org_model = torch.cat([temp_org_model, parms.grad.reshape((1, -1))], dim=0)
                # temp_label = torch.cat([temp_label, y_no_add_digit], dim=0)
                if i == 3: break

            # print(temp_label)
            acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            fl_acc = (logits_y2.argmax(dim=1) == y).float().mean().item()
            fl_org_acc = (fl_org_y.argmax(dim=1) == y).float().mean().item()
            metrics = {
                'fl_org_acc': fl_org_acc,
                'fl_acc': fl_acc,
                'acc': acc,
                'loss': loss.item(),
                'temp': vibi.temp,
                'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'KL(z||r)': KL_z_r.item(),
                'I_ZX_bound': I_ZX_bound,
            }

            for m, v in metrics.items():
                logs[m].append(v)

            if step % len(infer_train_loader) % 50 == 0:
                print(f'[{epoch}/{init_epoch + args.num_epochs}:{step % len(infer_train_loader):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))

        vibi.eval()
        valid_acc_old = valid_acc
        interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(infer_train_loader)).tolist()
        logs['val_acc'].extend(interpolate_valid_acc)
        print("test_acc", valid_acc)

        print(temp_org_model.shape, temp_label.shape)
        org_dataset = Data.TensorDataset(temp_org_model, temp_label)
        org_loader = DataLoader(org_dataset, batch_size=args.local_bs, shuffle=True)
        cp_dataset = Data.TensorDataset(temp_cp_mode, temp_label)
        cp_loader = DataLoader(cp_dataset, batch_size=args.local_bs, shuffle=True)

        avg_acc = 0
        for step, (x, y) in enumerate(org_loader):
            x, y = x.to(device), y.to(device).long().squeeze()
            org_y = org_infer(x.detach())
            # print(y)
            # print(cp_y)
            org_loss = org_infer_loss_fn(org_y, y)
            org_infer_optimizer.zero_grad()
            org_loss.backward()
            org_infer_optimizer.step()
            org_acc = (org_y.argmax(dim=1) == y).float().mean().item()
            avg_acc += org_acc
            # print('cp_acc', cp_acc)
        print('avg_org_acc', avg_acc / (step + 1))

        avg_acc = 0
        for step, (x, y) in enumerate(cp_loader):
            x, y = x.to(device), y.to(device).long().squeeze()
            cp_y = cp_infer(x.detach())

            cp_loss = cp_infer_loss_fn(cp_y, y)
            cp_infer_optimizer.zero_grad()
            cp_loss.backward()
            cp_infer_optimizer.step()
            cp_acc = (cp_y.argmax(dim=1) == y).float().mean().item()
            avg_acc += cp_acc
            # print('cp_acc', cp_acc)
        print('avg_acc', avg_acc / (step + 1))
    return org_infer, cp_infer


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.gpu = 0
    args.num_users = 5
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.iid = True
    args.model = 'z_linear'
    args.local_bs = 100
    args.local_ep = 1
    args.num_epochs = 1
    args.dataset = 'CIFAR10'
    args.epochs = int(60)
    args.add_noise = False
    args.beta=0.0001
    args.lr = 0.0005
    dimZ = 192
    device = 'cuda' if args.cuda else 'cpu'
    print("device", device)
    dataset = args.dataset
    # load dataset and split users
    if args.dataset == 'MNIST':
        trans_mnist = transforms.Compose([transforms.ToTensor(), ])  # transforms.Normalize((0.1307,), (0.3081,))
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), ])  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    print('img_size', len(dataset_train))
    # prepare mnist_img
    mnist_train = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.RandomCrop([32, 32], padding=(8, 16))]))
    mnist_loader = DataLoader(mnist_train, batch_size=args.local_bs, shuffle=True, num_workers=1)
    # added mnist_img
    for mnist_step, (mnist_img, digit) in enumerate(mnist_loader):
        if mnist_step == 0:
            mnist_img = mnist_img.to(device)
            # print(mnist_step,mnist_img)
            break
    # infer model prepare
    # org_infer, cp_infer = infer_model_prepare(args, mnist_img)

    #infer model for cp and org
    cp_infer = LinearCIFAR(n_feature=3*32 * 32, n_output=2)
    cp_infer.to(device)
    cp_infer_optimizer = torch.optim.Adam(cp_infer.parameters(), lr=0.005)
    cp_infer_loss_fn = nn.CrossEntropyLoss()

    org_infer = LinearCIFAR(n_feature=3*32 * 32, n_output=2)
    org_infer.to(device)
    org_infer_optimizer = torch.optim.Adam(org_infer.parameters(), lr=0.005)
    org_infer_loss_fn = nn.CrossEntropyLoss()

    # build model
    if args.model == 'cnn' and args.dataset == 'CIFAR10':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'MNIST':
        net_glob = LinearMNIST(n_feature=dimZ).to(args.device)
    elif args.model == 'z_linear' and args.dataset == 'CIFAR10':
        net_glob = resnet18(3, 10).to(device)
        org_net_glob = resnet18(3, 10).to(device)
        org_resnet = resnet18(3, 10).to(device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    org_net_glob.train()
    org_resnet.train()

    # copy weights
    w_glob = net_glob.state_dict()
    org_w_glob = org_net_glob.state_dict()
    w_glob_org_res = org_resnet.state_dict()

    # training
    loss_train = []
    acc_test = []
    acc_test_org = []
    XZ_bound = []
    infer_acc = []
    infer_acc_org = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    print("dataset_train", len(dataset_train))
    print("dict_users", len(dict_users))

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    loss_avg = 1
    # beta = args.lr
    batch_size = args.bs
    samples_amount = 30
    idxs_users = range(args.num_users)
    dict_usersZmodel = []
    dict_optimizer = []
    dict_userFLmodel =[]
    dict_FL_optimizer = []
    for idx in idxs_users:
        vibi, z_optimizer = init_cp_model(args)
        dict_usersZmodel.append(vibi)
        dict_optimizer.append(z_optimizer)

        FL_model = LinearCIFAR(n_feature=dimZ, n_output=10)
        FL_model.to(device)
        FL_optimizer = torch.optim.Adam(FL_model.parameters(), lr=0.001)
        dict_userFLmodel.append(FL_model)
        dict_FL_optimizer.append(FL_optimizer)


    for iter in range(args.epochs):
        dict_usersZ = []
        idxs_users = range(args.num_users)
        XZ_Mutual = []

        w_locals = []
        w_locals_org = []
        w_locals_org_res = []

        for idx in idxs_users:
            # print(len(dataset_train), dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            Z_model = dict_usersZmodel[idx]
            # z_optimizer = dict_optimizer[idx]
            FL_model = dict_userFLmodel[idx]
            FL_optimizer = dict_FL_optimizer[idx]

            vibi = VIBI(Z_model.explainer, Z_model.approximator, copy.deepcopy(net_glob).to(args.device), Z_model.reconstructor, k=args.k, num_samples=args.num_samples)
            vibi.to(device)
            optimizer = torch.optim.Adam(vibi.parameters(), lr=0.005)

            if args.dataset == "MNIST":
                reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
                reconstructor = reconstructor.to(device)
                optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)
            elif args.dataset == "CIFAR10":
                reconstructor = resnet18(3, 3 * 32 * 32)  # LinearModel(n_feature=49, n_output=3 * 32 * 32)
                reconstructor = reconstructor.to(device)
                optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=args.lr)

            torch.cuda.manual_seed(42)

            reconstruction_function = nn.MSELoss(size_average=True)

            lr_train_dl, Z_model, XZ_Mutual, net_glob_w, org_net_glob_w, org_net_local_update = vibi_train(args=args, ldr_train=local.ldr_train, z_dim=dimZ, user_idx=idx,
                                            loss_avg=loss_avg, vibi=vibi, optimizer=optimizer, FL_model=FL_model, FL_optimizer=FL_optimizer, net_glob=net_glob, XZ_Mutual=XZ_Mutual,\
                                            mnist_img=mnist_img, org_net_glob=copy.deepcopy(org_net_glob).to(args.device), org_resnet= copy.deepcopy(org_resnet).to(args.device),reconstruction_function=reconstruction_function)
            dict_usersZ.append(lr_train_dl)
            dict_usersZmodel[idx] = Z_model

            w_locals.append(copy.deepcopy(net_glob_w))
            w_locals_org.append(copy.deepcopy(org_net_glob_w))

        w_glob = FedAvg(w_locals)
        org_w_glob = FedAvg(w_locals_org)
        # w_glob_org_res = FedAvg(w_locals_org_res)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        org_net_glob.load_state_dict(org_w_glob)
        # org_resnet.load_state_dict(w_glob_org_res)

        XZ_bound.append(XZ_Mutual)
        loss_locals = []

        cp_z_model_w =AvgCP(dict_usersZmodel)
        vibi, z_optimizer = init_cp_model(args)
        vibi.load_state_dict(cp_z_model_w)
        # dict_usersZmodel[0].load_state_dict(cp_z_model_w)
        print("epoch: ", iter)
        acc_temp = acc_evaluation(net_glob, dataset_test, args, dict_usersZmodel[0])
        acc_test.append(acc_temp)
        acc_temp = acc_evaluation_org(org_net_glob, dataset_test, args)
        acc_test_org.append(acc_temp)


    print(infer_acc)
    print(infer_acc_org)
    print(acc_test)
    print(acc_test_org)
    # print("XZ mutual", XZ_bound)
    # testing
    net_glob.eval()
    cp_z_model_w = AvgCP(dict_usersZmodel)
    vibi, z_optimizer = init_cp_model(args)
    vibi.load_state_dict(cp_z_model_w)
    acc_train, loss_train = testZ_img(net_glob, dataset_train, args, dict_usersZmodel[0])
    acc_test, loss_test = testZ_img(net_glob, dataset_test, args, dict_usersZmodel[0])
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))



