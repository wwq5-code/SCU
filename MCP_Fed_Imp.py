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
from torch.autograd import Variable

import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# from VIBImodels import ResNet, resnet18, resnet34, Unet

# from debug import debug
import torch.nn as nn
import torch.optim

import torch.nn.functional as F


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
    parser.add_argument('--auxiliary', action='store_true', default=False,
                        help='need auxiliary data or not for non iid')
    parser.add_argument('--add_noise', action='store_true', default=False, help='need add noise or not for non iid')

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
    num_shards, num_imgs = 2 * num_users, int(len(dataset) / (2 * num_users))  # 2*
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
    each_put = int(each_user_has / num_users)
    dict_users[num_users] = []
    for i in range(num_users):
        rand_set = list(set(np.random.choice(dict_users[i], each_put, replace=False)))
        dict_users[num_users] = np.concatenate((dict_users[num_users], rand_set), axis=0).astype(int)
    return dict_users, num_users + 1


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
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (2 * num_users))
    # num_shards, num_imgs = 20, 3000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.array(dataset.targets)

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


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_channels),
    )


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


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
    block_features = [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2
    return ResNet(in_channels, block_features, num_classes)


def resnet34(in_channels, num_classes):
    block_features = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
    return ResNet(in_channels, block_features, num_classes)


class VIBI(nn.Module):
    def __init__(self, explainer, approximator, reconstructor):
        super().__init__()

        self.explainer = explainer
        self.approximator = approximator
        self.reconstructor = reconstructor
        # self.fc3 = nn.Linear(49, 400)
        # self.fc4 = nn.Linear(400, 784)


    def explain(self, x, mode='topk', num_samples=None):
        """Returns the relevance scores
        """
        double_logits_z = self.explainer(x)  # (B, C, h, w)
        if mode == 'distribution':  # return the distribution over explanation
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z, mu, logvar
        elif mode == 'test':  # return top k pixels from input
            B, double_dimZ = double_logits_z.shape
            dimZ = int(double_dimZ / 2)
            mu = double_logits_z[:, :dimZ].cuda()
            logvar = torch.log(torch.nn.functional.softplus(double_logits_z[:, dimZ:]).pow(2)).cuda()
            logits_z = self.reparametrize(mu, logvar)
            return logits_z

    def forward(self, x, mode='topk'):
        B = x.size(0)
        #         print("B, C, H, W", B, C, H, W)
        if mode == 'distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'distribution_reconstruct':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            logits_z2 = logits_z.view(logits_z.size(0), 3, 7, 7)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstructor(logits_z)
            return logits_z, logits_y, mu, logvar, x_hat
        elif mode == 'cifar_distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar
        elif mode == 'forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_forgetting':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.cifar_forget(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'cifar_test':
            logits_z = self.explain(x, mode='test')  # (B, C, H, W)
            # B, dimZ = logits_z.shape
            # logits_z = logits_z.reshape((B,8,8,8))
            logits_y = self.approximator(logits_z)
            return logits_y
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y

    def forget(self, logits_z):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.reconstructor(logits_z)
        return torch.sigmoid(output_x)

    def cifar_forget(self, logits_z):
        # B, c, h, w = logits_z.shape
        # logits_z=logits_z.reshape((B,-1))
        output_x = self.reconstructor(logits_z)
        return torch.sigmoid(output_x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


def init_vibi(args):
    k = args.k
    beta = args.beta
    num_samples = args.num_samples
    xpl_channels = args.xpl_channels
    explainer_type = args.explainer_type

    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ)
        reconstructor = LinearModel(n_feature=args.dimZ, n_output=28 * 28)
        explainer = LinearModel(n_feature=28 * 28, n_output=args.dimZ * 2)  # resnet18(1, 49*2) #
        lr = args.lr

    elif args.dataset == 'CIFAR10':
        #approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = LinearModel(n_feature=args.dimZ)
        explainer = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        reconstructor = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        explainer = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        reconstructor = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = 3e-4

    vibi = VIBI(explainer, approximator, reconstructor)
    vibi.to(args.device)
    return vibi, lr


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
    def __init__(self, args):
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

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, local_MCP_model=None, local_fed_model=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.reconstruction_loss_function = nn.MSELoss(size_average=True)
        self.selected_clients = []
        # prepare the local original dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, args.sampling), batch_size=self.args.local_bs,
                                    shuffle=True)
        self.local_MCP_model = local_MCP_model
        self.local_fed_model = local_fed_model

        # local data compressing
        # self.com

def add_noise(data, epsilon, sensitivity, args):
    noise_tesnor = np.random.laplace(1, sensitivity/epsilon,data.shape) * args.lr
    # data = torch.add(data, torch.from_numpy(noise_tesnor))
    # for x in np.nditer(np_data, op_flags=['readwrite']):
    #     x[...] = x + np.random.laplace(1, sensitivity/epsilon,)
    if args.gpu == -1:
        return data.add(torch.from_numpy(noise_tesnor).float())
    else:
        return data.add(torch.from_numpy(noise_tesnor).float().to(args.device))


def local_model_train(net, args, user_local):
    net.train()
    # train and update, first initialize the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    epoch_loss = []
    for iter in range(args.local_ep):
        batch_loss = []
        # print(iter)
        for batch_idx, (images, labels) in enumerate(user_local.ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            images = images.view(images.size(0), -1)
            # print('batch_idx', batch_idx, images.shape)
            if args.add_noise:
                images = add_noise(data=images, epsilon=args.epsilon, sensitivity=1, args=args)
            net.zero_grad()
            log_probs = net(images)
            loss = user_local.loss_func(log_probs, labels)
            # loss = F.nll_loss(log_probs, labels)
            loss.backward()
            optimizer.step()
            # print('batch_idx', batch_idx)

            metrics = {
                # 'acc': acc,
                'loss': loss.item(),
                # 'BCE': BCE.item(),
                # 'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                # 'KLD': KLD.item(),
                # 'KLD_mean': KLD_mean.item(),
            }


            if (batch_idx) % 20 == 0:  #local_bs = 100, batch_idx * local_bs is the print point
                print('Update Epoch: {} [{}/{} ({:.0f}%)]'.format(iter, batch_idx * len(images),
                                                                  len(user_local.ldr_train.dataset),
                                                                  args.local_bs * batch_idx / len(
                                                                      user_local.ldr_train)) + ', '.join(
                    [f'{k} {v:.3f}' for k, v in metrics.items()]))

            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
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


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.local_bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        data, target = data.cuda(), target.cuda()
        data = data.view(data.size(0), -1)
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


def acc_evaluation_org(fedAvg_glob, dataset_test, args):
    acc_test, loss_test = test_img(fedAvg_glob, dataset_test, args)
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test



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


@torch.no_grad()
def test_accuracy(model, data_loader, args):
    num_total = 0
    num_correct = 0
    model.eval()
    num_correct2 = 0
    num_correct3 = 0
    num_correct_global = 0
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        logits_z, out, p_z , outy_2, x_hat= model(x, mode='distribution_reconstruct')
        B, C, H, W = x.shape
        pz2 = p_z.reshape((B, -1))

        B,C,H,W = x.shape
        x_for_fl_org = x.reshape((B, -1))

        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()


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

def mcp_local_train(mcp_vibi, user_local, args, idx):
    valid_acc = 0.8

    mcp_vibi.train()
    # train and update, first initialize the optimizer
    optimizer = torch.optim.Adam(mcp_vibi.parameters(), lr=args.lr)

    temp_img = torch.empty(0, args.dimZ).float().cuda()
    temp_label = torch.empty(0).long().cuda()

    for epoch in range(args.local_ep):

        for batch_idx, (x, y) in enumerate(user_local.ldr_train):

            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)


            logits_z, logits_y, mu, logvar, x_hat = mcp_vibi(x, mode='distribution_reconstruct')

            B, C, H, W = x.shape

            # logits_y = logits_y.argmax(dim=1).float()
            # this is used to update the approximator
            H_p_q = user_local.loss_func(logits_y, y)

            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
            KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
            KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()


            x_hat = x_hat.view(x_hat.size(0), -1)
            x_view = x.view(x.size(0), -1)
            # x = torch.sigmoid(torch.relu(x))
            BCE = user_local.reconstruction_loss_function(x_hat, x_view)  # mse loss

            loss =  H_p_q + args.beta * KLD_mean #+ BCE

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (logits_y.argmax(dim=1) == y).float().mean().item()
            # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
            # fl_acc = (fl_y.argmax(dim=1) == y).float().mean().item()
            # fl_org_acc = (fl_org_y.argmax(dim=1) == y).float().mean().item()
            # # fl_res_acc = (fl_res_Y.argmax(dim=1) == y).float().mean().item()
            # global_acc = (logits_y2.argmax(dim=1) == y).float().mean().item()
            metrics = {
                # 'user_idx': user_idx,
                # 'fl_org_res': fl_res_acc,
                # 'fl_org_acc': fl_org_acc,
                # 'global_y': global_acc,
                # 'fl_acc': fl_acc,
                'acc': acc,
                'loss': loss.item(),
                'H(p,q)': H_p_q.item(),
                # '1-JS(p,q)': JS_p_q,
                'KLD_mean': KLD_mean.item(),
                'BCE': BCE.item(),
            }
            if epoch==1 and (batch_idx) % 20 == 0:  #local_bs = 100, batch_idx * local_bs is the print point
                print('User id:' + str(idx) +', Update Epoch: {} [{}/{} ({:.0f}%)]'.format(iter, batch_idx * len(x),
                                                                  len(user_local.ldr_train.dataset),
                                                                  args.local_bs * batch_idx / len(
                                                                      user_local.ldr_train)) + ', '.join(
                    [f'{k} {v:.3f}' for k, v in metrics.items()]))

    # b = Data.TensorDataset(temp_img, temp_label)
    # lr_train_dl = DataLoader(b, batch_size=args.local_bs, shuffle=True)

    print(f'explainer params:\t{num_params(mcp_vibi.explainer) / 1000:.2f} K')
    print(f'approximator params:\t{num_params(mcp_vibi.approximator) / 1000:.2f} K')
    # print(f'org_net_glob params:\t{num_params(org_net_glob) / 1000:.2f} K')

    return mcp_vibi, mcp_vibi.approximator.state_dict()



# if __name__ == '__main__':
#     # parse args


args = args_parser()
args.gpu = 0
args.num_users = 50
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.local_bs = 100
args.local_ep = 10
# args.num_epochs = 1 # used for what?
args.dataset = 'CIFAR10'
args.epochs = int(20)  # global rounds
args.add_noise = False
args.beta = 0.0001
args.lr = 0.0005
args.dimZ = 3 * 7 * 7  # for cifar10 we use 3*7*7 and for mnist we use 7*7

device = args.device  # 'cuda' if args.cuda else 'cpu'
print("device", device)
# dataset = args.dataset

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

# prepare mnist_img， this is used to add to the cifar10 picture, we also have other ways like add backdoors.
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

# infer model for cp and org
cp_infer = LinearModel(n_feature=3 * 32 * 32, n_output=2)
cp_infer.to(device)
cp_infer_optimizer = torch.optim.Adam(cp_infer.parameters(), lr=args.lr)
cp_infer_loss_fn = nn.CrossEntropyLoss()

org_infer = LinearModel(n_feature=3 * 32 * 32, n_output=2)
org_infer.to(device)
org_infer_optimizer = torch.optim.Adam(org_infer.parameters(), lr=args.lr)
org_infer_loss_fn = nn.CrossEntropyLoss()

# build model
if args.model == 'cnn' and args.dataset == 'CIFAR10':
    net_glob_of_MCP = CNNCifar(args=args).to(args.device)
elif args.model == 'cnn' and args.dataset == 'MNIST':
    net_glob_of_MCP = CNNMnist(args=args).to(args.device)
elif args.model == 'z_linear' and args.dataset == 'MNIST':
    net_glob_of_MCP = LinearModel(n_feature=args.dimZ).to(args.device)
elif args.model == 'z_linear' and args.dataset == 'CIFAR10':
    # here, we only use a linear model as an
    #net_glob_of_MCP = resnet18(3,10) #LinearModel(n_feature=args.dimZ).to(args.device)
    net_glob_of_MCP = LinearModel(n_feature=args.dimZ).to(args.device)
    fedAvg_glob = LinearModel(n_feature=3 * 32 * 32).to(
        args.device)  # we can change to resnet as  = resnet18(3, 10).to(device)
    # net_glob = resnet18(3, 10).to(device)
    # org_net_glob = resnet18(3, 10).to(device)
    # org_resnet = resnet18(3, 10).to(device)
else:
    exit('Error: unrecognized model')
print(net_glob_of_MCP)
net_glob_of_MCP.train()
fedAvg_glob.train()
# org_resnet.train()

# copy weights
#w_glob = net_glob.state_dict()
w_fed_glob = fedAvg_glob.state_dict()

# org_w_glob = org_net_glob.state_dict()
# w_glob_org_res = org_resnet.state_dict()

# training
loss_train = []
acc_test = []
acc_test_org = []
XZ_bound = []
infer_acc = []
infer_acc_org = []

print("dataset_train", len(dataset_train))
print("dict_users", len(dict_users))

if args.all_clients:
    print("Aggregation over all clients")
    w_locals = [w_fed_glob for i in range(args.num_users)]

idxs_users = range(args.num_users)
"""create a idxs_local_list"""
idxs_local_dict = {}
for idx in idxs_users:

    vibi, z_optimizer = init_vibi(args)
    fed_model = LinearModel(n_feature=3 * 32 * 32, n_output=10)
    fed_model.to(device)
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], local_MCP_model=vibi, local_fed_model=fed_model)
    idxs_local_dict[idx] = local


w_fed_glob_app = vibi.approximator.state_dict()

for iter in range(args.epochs):
    dict_usersZ = []
    idxs_users = range(args.num_users)
    #XZ_Mutual = []

    w_locals_app = []
    w_locals_fed = []
    #w_locals_org_res = []

    for idx in idxs_users:
        # why do we not put the data together with the model?

        Z_model = idxs_local_dict[idx].local_MCP_model
        # z_optimizer = dict_optimizer[idx]
        fed_model = idxs_local_dict[idx].local_fed_model

        # here we only use the later half model of the vibi to train the global model
        mcp_vibi, opt = init_vibi(args)
        mcp_vibi = VIBI(Z_model.explainer, copy.deepcopy(net_glob_of_MCP).to(args.device), Z_model.reconstructor)
        mcp_vibi.to(device)
        # optimizer = torch.optim.Adam(vibi.parameters(), lr=args.lr)

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

        user_local = idxs_local_dict[idx]

        mcp_vibi, approximator_state =  mcp_local_train(mcp_vibi, user_local, args,idx) #local_model_train(copy.deepcopy(fed_model).to(args.device), args, user_local)

        # dict_usersZ.append(lr_train_dl)
        # dict_usersZmodel[idx] = Z_model
        w_locals_app.append(copy.deepcopy(approximator_state))
        # w_locals_fed.append(copy.deepcopy(approximator_state))
        #w_locals_org.append(copy.deepcopy(org_net_glob_w))

    # w_fed_glob = FedAvg(w_locals_fed)
    w_fed_glob_app = FedAvg(w_locals_app)
    # w_glob_org_res = FedAvg(w_locals_org_res)
    # copy weight to net_glob
    net_glob_of_MCP.load_state_dict(w_fed_glob_app)
    #update each local fedavg model
    for idx in idxs_users:
        Z_model = idxs_local_dict[idx].local_MCP_model
        vibi, opt = init_vibi(args)
        vibi = VIBI(Z_model.explainer, copy.deepcopy(net_glob_of_MCP).to(args.device), Z_model.reconstructor)
        vibi = vibi.to(args.device)
        idxs_local_dict[idx].local_MCP_model = copy.deepcopy(vibi).to(args.device)

        # idxs_local_dict[idx].local_fed_model = copy.deepcopy(fedAvg_glob).to(args.device)
    #org_net_glob.load_state_dict(org_w_glob)
    # org_resnet.load_state_dict(w_glob_org_res)

    #XZ_bound.append(XZ_Mutual)
    # loss_locals = []

    # cp_z_model_w = AvgCP(dict_usersZmodel)
    # vibi, z_optimizer = init_cp_model(args)
    # vibi.load_state_dict(cp_z_model_w)
    # dict_usersZmodel[0].load_state_dict(cp_z_model_w)
    print("epoch: ", iter)
    # acc_temp = acc_evaluation(net_glob, dataset_test, args, dict_usersZmodel[0])
    # acc_test.append(acc_temp)
    #acc_temp = acc_evaluation_org(fedAvg_glob, dataset_test, args)
    # here the vivi is the local compressive model and the global averaged model of last round
    test_loader = DataLoader(dataset_test, batch_size=args.local_bs)
    acc_temp = test_accuracy(vibi, test_loader, args)
    acc_test_org.append(round(acc_temp, 2))
    print("test acc", round(acc_temp, 2))
    #acc_test_org.append(round(acc_temp.item(),2)) # this is for tensor

print(infer_acc)
print(infer_acc_org)
print(acc_test)
print(acc_test_org)

# print("XZ mutual", XZ_bound)
# testing
# net_glob.eval()
# cp_z_model_w = AvgCP(dict_usersZmodel)
# vibi, z_optimizer = init_cp_model(args)
# vibi.load_state_dict(cp_z_model_w)
# acc_train, loss_train = testZ_img(net_glob, dataset_train, args, dict_usersZmodel[0])
# acc_test, loss_test = testZ_img(net_glob, dataset_test, args, dict_usersZmodel[0])
# print("Training accuracy: {:.2f}".format(acc_train))
# print("Testing accuracy: {:.2f}".format(acc_test))
