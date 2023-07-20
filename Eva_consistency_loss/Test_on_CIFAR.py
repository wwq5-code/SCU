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


    def explain(self, x, mode='topk'):
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
        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
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

    def reconstruction(self, logits_z):
        B, dimZ = logits_z.shape
        logits_z = logits_z.reshape((B, -1))
        output_x = self.reconstructor(logits_z)
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



def init_vibi(args):

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


def num_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])



def learning_train(dataset, model, loss_fn, reconstruction_function, args, epoch, mu_list,
                   sigma_list, train_loader):
    logs = defaultdict(list)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(dataset):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        #x = x.view(x.size(0), -1)
        # print(x)
        # break

        #test_grad(model, x, y, loss_fn, optimizer)


        logits_z, logits_y, x_hat, mu, logvar = model(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
        H_p_q = loss_fn(logits_y, y)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        BCE = reconstruction_function(x_hat, x)  # mse loss

        loss = args.beta * KLD_mean + H_p_q  + BCE # / (args.batch_size * 28 * 28)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2.0, error_if_nonfinite=False)
        optimizer.step()

        acc = (logits_y.argmax(dim=1) == y).float().mean().item()
        sigma = torch.sqrt_(torch.exp(logvar)).mean().item()
        # JS_p_q = 1 - js_div(logits_y.softmax(dim=1), y.softmax(dim=1)).mean().item()
        metrics = {
            'acc': acc,
            'loss': loss.item(),
            'BCE': BCE.item(),
            'H(p,q)': H_p_q.item(),
            # '1-JS(p,q)': JS_p_q,
            'mu': torch.mean(mu).item(),
            'sigma': sigma,
            'KLD': KLD.item(),
            'KLD_mean': KLD_mean.item(),
        }

        for m, v in metrics.items():
            logs[m].append(v)
        if epoch == args.num_epochs - 1:
            mu_list.append(torch.mean(mu).item())
            sigma_list.append(sigma)
        if step % len(train_loader) % 600 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
    return model, mu_list, sigma_list



@torch.no_grad()
def test_accuracy(model, data_loader, args, name='test'):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        #x = x.view(x.size(0), -1)
        out = model(x, mode='test')
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'{name} accuracy: {acc:.4f}')
    return acc

torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

#torch.use_deterministic_algorithms(True)

# parse args
args = args_parser()
args.gpu = 0
args.num_users = 10
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
args.iid = True
args.model = 'z_linear'
args.num_epochs = 20
args.dataset = 'CIFAR10'
args.add_noise = False
args.beta = 0.01
args.lr = 0.0005
args.dimZ = 7*7*3

args.batch_size = 16



print('args.beta', args.beta, 'args.lr', args.lr)

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
    train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=True)
    train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=True)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)





vibi, lr = init_vibi(args)
vibi.to(args.device)

valid_acc = 0.8
loss_fn = nn.CrossEntropyLoss()

reconstructor = LinearModel(n_feature=49, n_output=28 * 28)
reconstructor = reconstructor.to(device)
optimizer_recon = torch.optim.Adam(reconstructor.parameters(), lr=lr)

reconstruction_function = nn.MSELoss(size_average=True)

acc_test = []
print("learning")

print('Training VIBI')
print(f'{type(vibi.explainer).__name__:>10} explainer params:\t{num_params(vibi.explainer) / 1000:.2f} K')
print(f'{type(vibi.approximator).__name__:>10} approximator params:\t{num_params(vibi.approximator) / 1000:.2f} K')
print(f'{type(vibi.reconstructor).__name__:>10} reconstructor params:\t{num_params(vibi.reconstructor) / 1000:.2f} K')
# inspect_explanations()
mu_list = []
sigma_list = []
for epoch in range(args.num_epochs):
    vibi.train()

    vibi, mu_list, sigma_list = learning_train(train_loader, vibi, loss_fn, reconstruction_function, args,
                                               epoch, mu_list, sigma_list, train_loader)
    vibi.eval()
    valid_acc_old = valid_acc
    valid_acc = test_accuracy(vibi, test_loader, args, name='vibi valid top1')
    interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()

    print("test_acc", valid_acc)
    acc_test.append(valid_acc)

print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))


for batch_idx, (x, y) in enumerate(test_loader):
    x, y = x.to(device), y.to(device)  # (B, C, H, W), (B, 10)
    #x = x.view(x.size(0), -1)
    logits_z, logits_y, x_hat, mu, logvar = vibi(x, mode='with_reconstruction')  # (B, C* h* w), (B, N, 10)
    # x_hat = torch.sigmoid(reconstructor(logits_z))
    x_hat = x_hat.view(x_hat.size(0), -1)
    x = x.view(x.size(0), -1)
    break

x_hat_cpu = x_hat.cpu().data
x_hat_cpu = x_hat_cpu.clamp(0, 1)
x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()
x_cpu = x.cpu().data
x_cpu = x_cpu.clamp(0, 1)
x_cpu = x_cpu.view(x_cpu.size(0), 3, 32, 32)
grid = torchvision.utils.make_grid(x_cpu, nrow=4, cmap="gray")
plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
plt.show()


print('acc_test',acc_test)



