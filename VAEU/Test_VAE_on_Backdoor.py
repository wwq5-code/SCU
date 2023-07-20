import sys

sys.argv = ['']
del sys

import os
import math
from collections import defaultdict
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim
from torchvision.utils import save_image
import torchvision
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy
import random


class My_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.data, self.targets = self.get_image_label()

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)

    def get_image_label(self, ):
        if args.dataset == "MNIST":
            temp_img = torch.empty(0, 1, 28, 28).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.reshape(1, 1, 28, 28).to(args.device), torch.tensor([label]).long().to(args.device)
                # print(image)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
        elif args.dataset == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().to(args.device)
            temp_label = torch.empty(0).long().to(args.device)
            for id in self.indices:
                image, label = self.dataset[id]
                image, label = image.to(args.device).reshape(1, 3, 32, 32), torch.tensor([label]).long().to(args.device)
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

        print(temp_label.shape, temp_img.shape)
        d = Data.TensorDataset(temp_img, temp_label)
        return temp_img, temp_label


def scoring_function(matrix):
    # This is a simple scoring function that returns the matrix itself as scores.
    # You can replace it with your own scoring function if needed.
    return matrix

def dp_sampling(matrix, epsilon, sample_size, replacement):
    scores = scoring_function(matrix)
    sensitivity = 1.0  # The sensitivity of our scoring function is 1

    # Calculate probabilities using the exponential mechanism
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))
    probabilities /= probabilities.sum()

    # Flatten the matrix and probabilities for sampling
    flat_matrix = matrix.flatten()
    flat_probabilities = probabilities.flatten()

    # Sample elements without replacement
    sampled_indices = np.random.choice(
        np.arange(len(flat_matrix)),
        size=sample_size,
        replace=replacement,
        p=flat_probabilities
    )

    # Create the output matrix with 0s
    output_matrix = np.zeros_like(matrix)

    # Set the sampled elements to their original values
    np.put(output_matrix, sampled_indices, flat_matrix[sampled_indices])

    return output_matrix


class PoisonedDataset(Dataset):

    def __init__(self, dataset, base_label, trigger_label, poison_samples, mode="train", device=torch.device("cuda"),
                 dataname="MNIST", args =None, add_backdoor=1, dp_sample=0):
        # self.class_num = len(dataset.classes)
        # self.classes = dataset.classes
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.add_backdoor = add_backdoor
        self.dp_sample = dp_sample
        self.args = args
        self.data, self.targets = self.add_trigger(self.reshape(dataset, dataname), dataset.targets, base_label,
                                                   trigger_label, poison_samples, mode)
        self.channels, self.width, self.height = self.__shape_info__()
        # self.data_test, self.targets_test = self.add_trigger_test(self.reshape(dataset.data, dataname), dataset.targets, base_label, trigger_label, portion, mode)

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]

        label = np.zeros(10)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, dataset, dataname="MNIST"):
        if dataname == "MNIST":
            temp_img = dataset.data.reshape(len(dataset.data), 1, 28, 28).float()
        elif dataname == "CIFAR10":
            temp_img = torch.empty(0, 3, 32, 32).float().cuda()
            temp_label = torch.empty(0).long().cuda()
            for id in range(len(dataset)):
                image, label = dataset[id]
                image, label = image.cuda().reshape(1, 3, 32, 32), torch.tensor([label]).long().cuda()
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)
                # print(id)

        # x = torch.Tensor(image.cuda())
        # x = torch.tensor(image)
        # # print(x)

        return np.array(temp_img.to("cpu"))

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, base_label, trigger_label, poison_samples, mode):
        print("## generate——test " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = []
        new_data_re = []

        # total_poison_num = int(len(new_data) * portion/10)
        _, width, height = data.shape[1:]
        for i in range(len(data)):
            if targets[i] == base_label:
                new_targets.append(trigger_label)
                if trigger_label != base_label:
                    if self.add_backdoor == 1:
                        new_data[i, :, width - 3, height - 3] = 255
                        new_data[i, :, width - 3, height - 4] = 255
                        new_data[i, :, width - 4, height - 3] = 255
                        new_data[i, :, width - 4, height - 4] = 255
                    # new_data[i, :, width - 23, height - 21] = 254
                    # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254
                new_data[i] = new_data[i] / 255

                if self.dp_sample == 1:
                    replacement = False
                    sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
                    new_data_re.append(sampled_matrix)
                elif self.dp_sample==2:
                    replacement = True
                    sampled_matrix = dp_sampling(new_data[i], args.epsilon, args.dp_sampling_size, replacement)
                    new_data_re.append(sampled_matrix)
                else:
                    new_data_re.append(new_data[i])
                # print("new_data[i]",new_data[i])
                poison_samples = poison_samples - 1
                if poison_samples <= 0:
                    break
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 1, 28, 28)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()

        return torch.Tensor(new_data_re), torch.Tensor(new_targets).long()


def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10'], default='MNIST')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs for VIBI.')
    parser.add_argument('--explainer_type', choices=['Unet', 'ResNet_2x', 'ResNet_4x', 'ResNet_8x'],
                        default='ResNet_4x')
    parser.add_argument('--xpl_channels', type=int, choices=[1, 3], default=1)
    parser.add_argument('--k', type=int, default=12, help='Number of chunks.')
    parser.add_argument('--beta', type=float, default=0, help='beta in objective J = I(y,t) - beta * I(x,t).')
    parser.add_argument('--unlearning_ratio', type=float, default=0.1)
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples used for estimating expectation over p(t|x).')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--save_best', action='store_true',
                        help='Save only the best models (measured in valid accuracy).')
    parser.add_argument('--save_images_every_epoch', action='store_true', help='Save explanation images every epoch.')
    parser.add_argument('--jump_start', action='store_true', default=False)
    args = parser.parse_args()
    return args


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


class VIB(nn.Module):
    def __init__(self, encoder, approximator, decoder):
        super().__init__()

        self.encoder = encoder
        self.approximator = approximator
        self.decoder = decoder

    def explain(self, x, mode='topk'):
        """Returns the relevance scores
        """
        double_logits_z = self.encoder(x)  # (B, C, h, w)
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
        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, x_hat, mu, logvar
        elif mode == 'test':
            logits_z = self.explain(x, mode=mode)  # (B, C, H, W)
            logits_y = self.approximator(logits_z)
            return logits_y

    def reconstruction(self, logits_z):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.decoder(logits_z)
        return torch.sigmoid(output_x)

    def cifar_recon(self, logits_z):
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


def init_vib(args):
    if args.dataset == 'MNIST':
        approximator = LinearModel(n_feature=args.dimZ)
        decoder = LinearModel(n_feature=args.dimZ, n_output=28 * 28)
        encoder = LinearModel(n_feature=28 * 28, n_output=args.dimZ * 2)  # resnet18(1, 49*2) #
        lr = args.lr

    elif args.dataset == 'CIFAR10':
        # approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = LinearModel(n_feature=args.dimZ)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        approximator = LinearModel(n_feature=args.dimZ, n_output=100)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        lr = args.lr

    vib = VIB(encoder, approximator, decoder)
    vib.to(args.device)
    return vib, lr


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def vae_train(dataset, model, loss_fn, reconstruction_function, args, epoch, mu_list, sigma_list, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_z, x_hat, mu, logvar = model(x, mode='VAE')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss

        loss = args.beta * KLD_mean + BCE  # / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        # acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            # 'acc': acc,
            'loss': loss.item(),
            'BCE': BCE.item(),
            # 'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            plt.show()
    return model, mu_list, sigma_list


def plot_latent(autoencoder, data_loader, args, num_batches=100):
    for i, (x, y) in enumerate(data_loader):
        x = x.view(x.size(0), -1)
        z, x_hat, mu, logvar = autoencoder(x.to(args.device), mode='VAE')
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu')
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.reconstruction(z)
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])


def create_backdoor_train_dataset(dataname, train_data, base_label, trigger_label, poison_samples, batch_size, args , add_backdoor, dp_sample):
    train_data = PoisonedDataset(train_data, base_label, trigger_label, poison_samples=poison_samples, mode="train",
                                 device=args.device, dataname=dataname, args=args, add_backdoor=add_backdoor, dp_sample=dp_sample)
    b = Data.TensorDataset(train_data.data, train_data.targets)
    # x = test_data_tri.data_test[0]
    x = torch.tensor(train_data.data[0])
    # print(x)
    x = x.cpu().data
    x = x.clamp(0, 1)
    if args.dataset == "MNIST":
        x = x.view(x.size(0), 1, 28, 28)
    elif args.dataset == "CIFAR10":
        x = x.view(1, 3, 32, 32)
    print(x)
    grid = torchvision.utils.make_grid(x, nrow=1, cmap="gray")
    plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
    plt.show()
    return train_data.data, train_data.targets


"""
                # x=torch.tensor(new_data[i])
                # x_cpu = x.cpu().data
                # x_cpu = x_cpu.clamp(0, 1)
                # x_cpu = x_cpu.view(1, 3, 32, 32)
                # grid = torchvision.utils.make_grid(x_cpu, nrow=1, cmap="gray")
                # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
                # plt.show()
"""



def linear_train(data_loader, model, loss_fn, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)

        logits_y = model(x)  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, y)

        loss = H_p_q

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()

        metrics = {
            'acc': acc,
            'loss': loss.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            # 'mu': torch.mean(mu).item(),
            # 'sigma': sigma,
            # 'KLD': KLD.item(),
            # 'KLD_mean': KLD_mean.item(),
        }

        if step % len(data_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(data_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
    return model


@torch.no_grad()
def test_linear_acc(model, data_loader, args, name='test', epoch=999):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        x = x.view(x.size(0), -1)
        out = model(x)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc


@torch.no_grad()
def eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    vib.eval()
    classifier_model.eval()
    num_total = 0
    num_correct = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        x = x.view(x.size(0), -1)
        logits_z, x_hat, mu, logvar = vib(x, mode='VAE')  # (B, C* h* w), (B, N, 10)

        x_hat = x_hat.view(x_hat.size(0), -1)
        # second, input the x_hat to classifier
        logits_y = classifier_model(x_hat.detach())
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)
        if batch_idx == 0:
            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            plt.show()

    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc


def unlearning_vae(vib, args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, train_type):

    vib_unl, lr = init_vib(args)
    vib_unl.to(args.device)
    vib_unl.load_state_dict(vib.state_dict())
    optimizer_unl = torch.optim.Adam(vib_unl.parameters(), lr=lr)

    print(len(dataloader_erase.dataset))
    train_bs = 0

    for epoch in range(args.num_epochs):
        vib_unl.train()
        batch_idx = 0
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x2 = x2.view(x2.size(0), -1)

            logits_z_e, x_hat_e, mu_e, logvar_e = vib_unl(x, mode='VAE')
            logits_z_e2, x_hat_e2, mu_e2, logvar_e2 = vib_unl(x2, mode='VAE')

            logits_z_f, x_hat_f, mu_f, logvar_f = vib(x, mode='VAE')

            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            KLD = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            # KLD between erased z and original z
            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            x_hat_e2 = x_hat_e2.view(x_hat_e2.size(0), -1)


            x = x.view(x.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            BCE2 = reconstruction_function(x_hat_e2, x2)
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p

            log_z = torch.mean(logits_z_e.log_softmax(dim=1))

            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

            # Calculate the L2-norm
            l2_norm_unl = torch.norm(args.kld_to_org * KLD_mean  - args.unlearn_bce * BCE , p=2)

            l2_norm_ss = torch.norm(args.self_sharing_rate * (args.beta * KLD_mean2  + BCE2 ), p=2)

            total_u_s = l2_norm_unl + l2_norm_ss
            unl_rate = l2_norm_unl / total_u_s
            self_s_rate = l2_norm_ss / total_u_s

            # unlearning_item = args.kld_to_org * KLD_mean.item() - args.unlearn_bce * BCE.item() # - args.reverse_rate * (log_z.item() )
            #
            # #print(unlearning_item)
            # learning_item = args.self_sharing_rate * (args.beta * KLD_mean2.item() + BCE2.item())
            # #print(learning_item)
            #
            # total = unlearning_item + learning_item # expected to equal to 0
            # if unlearning_item <= 0:# have approixmate to the retrained distribution and no need to unlearn
            #     unl_rate = 0
            # else:
            #     unl_rate = unlearning_item / total
            #
            # self_s_rate = 1 - unl_rate

            '''purpose is to make the unlearning item =0, and the learning item =0 '''

            if train_type == 'VAE_unl':
                loss = args.kld_to_org * KLD_mean - args.unlearn_bce * BCE
            elif train_type == 'VAE_unl_ss':
                loss = (args.kld_to_org * KLD_mean - args.unlearn_bce * BCE) * unl_rate + self_s_rate * args.self_sharing_rate * (
                               args.beta * KLD_mean2 + BCE2)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2

            optimizer_unl.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vib_unl.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
            optimizer_unl.step()

            metrics = {
                # 'unlearning_item': unlearning_item,
                # 'learning_item': learning_item,
                # 'acc': acc,
                'loss': loss.item(),
                'BCE': BCE.item(),
                'l2_norm_unl': l2_norm_unl,
                'l2_norm_ss': l2_norm_ss,
                # 'H(p,q)': H_p_q.item(),
                # 'kl_f_e': kl_f_e.item(),
                # 'H_p_q2': H_p_q2.item(),
                # '1-JS(p,q)': JS_p_q,
                # 'mu_e': torch.mean(mu_e).item(),
                # 'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                'KLD': KLD.item(),
                'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean': KLD_mean.item(),
            }

            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if batch_idx % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{batch_idx % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            batch_idx = batch_idx + 1
            train_bs = train_bs + 1
            # if acc_back < 0.05:
            #     break

        vib_unl.eval()

        valid_acc = eva_vae_generation(vib_unl, classifier_model, dataloader_erase, args, name='test', epoch=epoch)

        if valid_acc < 0.02:
            print()
            print("end unlearn, train_bs", train_bs)
            # break
    return vib_unl




torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
# args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.num_epochs = 25
args.dataset = 'MNIST'
args.add_noise = False
args.beta = 0.0001
args.lr = 0.0001
args.dimZ = 2 #40 #2
args.batch_size = 16
args.erased_local_r = 0.02  # the erased data ratio

args.kld_to_org = 1
args.unlearn_bce = 0.1
args.self_sharing_rate = 1

# print('args.beta', args.beta, 'args.lr', args.lr)

print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

device = args.device
print("device", device)

if args.dataset == 'MNIST':
    transform = T.Compose([
        T.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trans_mnist = transforms.Compose([transforms.ToTensor(), ])
    train_set = MNIST('../../data/mnist', train=True, transform=trans_mnist, download=True)
    test_set = MNIST('../../data/mnist', train=False, transform=trans_mnist, download=True)
    train_set_no_aug = train_set
elif args.dataset == 'CIFAR10':
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.ToTensor(),
                                 ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([T.ToTensor(),
                                ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=False)
    test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=False)
    train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=False)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

shadow_ratio = 0.0
full_ratio = 1 - shadow_ratio
unlearning_ratio = args.erased_local_r

length = len(train_set)
shadow_size, full_size = int(shadow_ratio * length), int(full_ratio * length)
remaining_size, erasing_size = int((1 - unlearning_ratio) * full_size), int(unlearning_ratio * full_size)
print('remaining_size', remaining_size)
remaining_size = full_size - erasing_size
print('remaining_size', remaining_size, shadow_size, full_size, erasing_size)

remaining_set, erasing_set = torch.utils.data.random_split(train_set, [remaining_size, erasing_size])

print(len(remaining_set))
print(len(remaining_set.dataset.data))

remaining_set = My_subset(remaining_set.dataset, remaining_set.indices)
erasing_set = My_subset(erasing_set.dataset, erasing_set.indices)

poison_samples = int(length) * args.erased_local_r

base_label=1
trigger_label = 2
add_backdoor=1 # =1 add backdoor , !=1 not add
poison_data, poison_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.batch_size, args=args, add_backdoor=add_backdoor, dp_sample=0)

trigger_label = 1
add_backdoor = 0
clean_data, clean_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.batch_size, args=args, add_backdoor=add_backdoor, dp_sample=0)


clean_dataset = Data.TensorDataset(clean_data, clean_targets)
dataloader_clean = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=True)

if args.dataset == 'MNIST':
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 1, 28, 28)
    erasing_set.data = erasing_set.data.reshape(len(erasing_set.data), 1, 28, 28)
elif args.dataset == 'CIFAR10':
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 3, 32, 32)
    erasing_set.data = erasing_set.data.reshape(len(erasing_set.data), 3, 32, 32)

print('train_set.data.shape', train_set.data.shape)
print('poison_data.shape', poison_data.shape)

data = torch.cat([poison_data.to(args.device), data_reshape.to(args.device)], dim=0)
targets = torch.cat([poison_targets.to(args.device), remaining_set.targets.to(args.device)], dim=0)

poison_trainset = Data.TensorDataset(data, targets)  # Data.TensorDataset(data, targets)
pure_backdoored_set = Data.TensorDataset(poison_data, poison_targets)

"""in a backdoored medol, we need to unlearn the trigger, 
so the remaining dataset is all the clean samples, and the erased dataset is the poisoned samples
here we set the pure_backdoored as the erased dataset
original erasing set is erasing_set = erasing_set"""
erasing_set = pure_backdoored_set

# if we don't use poisoned set, we use full set
dataloader_full = DataLoader(poison_trainset, batch_size=args.batch_size, shuffle=True)

# dataloader_full = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
dataloader_remain = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True)
dataloader_erase = DataLoader(erasing_set, batch_size=args.batch_size, shuffle=True)


vib, lr = init_vib(args)
vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()

reconstruction_function = nn.MSELoss(size_average=True)

acc_test = []
print("learning")

print('Training VIBI')
print(f'{type(vib.encoder).__name__:>10} encoder params:\t{num_params(vib.encoder) / 1000:.2f} K')
print(f'{type(vib.approximator).__name__:>10} approximator params:\t{num_params(vib.approximator) / 1000:.2f} K')
print(f'{type(vib.decoder).__name__:>10} decoder params:\t{num_params(vib.decoder) / 1000:.2f} K')
# inspect_explanations()

# train backdoor linear model
classifier_model = LinearModel(n_feature=28 * 28).to(args.device)

back_acc_list = []
for epoch in range(20):
    classifier_model.train()
    classifier_model = linear_train(dataloader_full, classifier_model, loss_fn, args)
    backdoor_acc = test_linear_acc(classifier_model, dataloader_erase, args, name='backdoor', epoch=epoch)

test_acc = test_linear_acc(classifier_model, test_loader, args, name='test')
backdoor_acc = test_linear_acc(classifier_model, dataloader_erase, args, name='backdoor')

# train VAE

mu_list = []
sigma_list = []
back_g_acc_lsit = []
for epoch in range(args.num_epochs):
    vib.train()
    vib, mu_list, sigma_list = vae_train(dataloader_full, vib, loss_fn, reconstruction_function, args, epoch, mu_list,
                                         sigma_list, train_loader)
    acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor',
                                    epoch=epoch)
    back_g_acc_lsit.append(acc_back_g)

print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))

acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor')
acc_norm_g = eva_vae_generation(vib, classifier_model, dataloader_clean, args, name='generated clean')

for batch_idx, (x, y) in enumerate(test_loader):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    x = x.view(x.size(0), -1)
    logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
    # x_hat = torch.sigmoid(reconstructor(logits_z))
    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    break

x_hat_cpu = x_hat.cpu().data
x_hat_cpu = x_hat_cpu.clamp(0, 1)
x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 1, 28, 28)
grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()

x_cpu = x.cpu().data
x_cpu = x_cpu.clamp(0, 1)
x_cpu = x_cpu.view(x_cpu.size(0), 1, 28, 28)
grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()

print('acc_test', acc_test)


print('start VAE_unl')
vibi_unl = unlearning_vae(copy.deepcopy(vib).to(args.device), args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, train_type='VAE_unl')

print('start VAE_unl_ss')
vibi_unl_ss = unlearning_vae(copy.deepcopy(vib).to(args.device), args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, train_type='VAE_unl_ss')


acc_back_g = eva_vae_generation(vibi_unl, classifier_model, dataloader_erase, args, name='vibi_unl generated backdoor')
acc_norm_g = eva_vae_generation(vibi_unl, classifier_model, dataloader_clean, args, name='vibi_unl generated clean')

acc_back_g = eva_vae_generation(vibi_unl_ss, classifier_model, dataloader_erase, args, name='vibi_unl_ss generated backdoor')
acc_norm_g = eva_vae_generation(vibi_unl_ss, classifier_model, dataloader_clean, args, name='vibi_unl_ss generated clean')

plot_latent(vib, dataloader_full, args)
