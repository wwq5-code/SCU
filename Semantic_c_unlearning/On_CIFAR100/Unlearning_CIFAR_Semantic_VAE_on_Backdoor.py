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
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CIFAR100
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
from scipy.stats import rice
import re
from collections import OrderedDict
import torch.nn.functional as func



import numpy as np
from torchvision.datasets import CIFAR100


class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]



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
        elif args.dataset == "CIFAR100":
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
        elif dataname == "CIFAR100":
            temp_img = torch.empty(0, 3, 32, 32).float().cuda()
            temp_label = torch.empty(0).long().cuda()
            for id in range(len(dataset)):
                image, label = dataset[id]
                image, label = image.cuda().reshape(1, 3, 32, 32), torch.tensor([label]).long().cuda()
                # print(label)
                # label = torch.tensor([label])
                temp_img = torch.cat([temp_img, image], dim=0)
                temp_label = torch.cat([temp_label, label], dim=0)

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
                        new_data[i, :, width - 3, height - 3] = 1
                        new_data[i, :, width - 3, height -4] = 1
                        new_data[i, :, width - 4, height - 3] = 1
                        new_data[i, :, width - 4, height - 4] = 1
                    # new_data[i, :, width - 23, height - 21] = 254
                    # new_data[i, :, width - 23, height - 22] = 254
                # new_data[i, :, width - 22, height - 21] = 254
                # new_data[i, :, width - 24, height - 21] = 254
                if self.dataname =='MNIST':
                    new_data[i] = new_data[i]/1
                # print(new_data[i].shape)

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




class AdaHessian(torch.optim.Optimizer):
    """
    Implements the AdaHessian algorithm from "ADAHESSIAN: An Adaptive Second OrderOptimizer for Machine Learning"

    Arguments:
        params (iterable) -- iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) -- learning rate (default: 0.1)
        betas ((float, float), optional) -- coefficients used for computing running averages of gradient and the squared hessian trace (default: (0.9, 0.999))
        eps (float, optional) -- term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) -- weight decay (L2 penalty) (default: 0.0)
        hessian_power (float, optional) -- exponent of the hessian trace (default: 1.0)
        update_each (int, optional) -- compute the hessian trace approximation only after *this* number of steps (to save time) (default: 1)
        n_samples (int, optional) -- how many times to sample `z` for the approximation of the hessian trace (default: 1)
    """

    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 hessian_power=1.0, update_each=1, n_samples=1, average_conv_kernel=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError(f"Invalid Hessian power value: {hessian_power}")

        self.n_samples = n_samples
        self.update_each = update_each
        self.average_conv_kernel = average_conv_kernel

        # use a separate generator that deterministically generates the same `z`s across all GPUs in case of distributed training
        self.generator = torch.Generator().manual_seed(2147483647)

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, hessian_power=hessian_power)
        super(AdaHessian, self).__init__(params, defaults)

        for p in self.get_params():
            p.requires_grad = True
            p.hess = 0.0
            self.state[p]["hessian step"] = 0

    def get_params(self):
        """
        Gets all parameters in all param_groups with gradients
        """

        return (p for group in self.param_groups for p in group['params'] if p.requires_grad)

    def zero_hessian(self):
        """
        Zeros out the accumalated hessian traces.
        """

        for p in self.get_params():
            if not isinstance(p.hess, float) and self.state[p]["hessian step"] % self.update_each == 0:
                p.hess.zero_()

    @torch.no_grad()
    def set_hessian(self):
        """
        Computes the Hutchinson approximation of the hessian trace and accumulates it for each trainable parameter.
        """

        params = []
        for p in filter(lambda p: p.grad is not None, self.get_params()):
            if self.state[p]["hessian step"] % self.update_each == 0:  # compute the trace only each `update_each` step
                params.append(p)
            self.state[p]["hessian step"] += 1

        if len(params) == 0:
            return

        if self.generator.device != params[0].device:  # hackish way of casting the generator to the right device
            self.generator = torch.Generator(params[0].device).manual_seed(2147483647)

        # for p in params:
        #     # p.grad.requires_grad=True
        #     print(p.shape)
        #     print(p.grad.shape)
        grads = [p.grad for p in params]
        # grads.requires_grad = True
        # print("grads", grads)

        for i in range(self.n_samples):
            zs = [torch.randint(0, 2, p.size(), generator=self.generator, device=p.device) * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}

            h_zs = torch.autograd.grad(grads, params, grad_outputs=zs, only_inputs=True, retain_graph=i < self.n_samples - 1)
            for h_z, z, p in zip(h_zs, zs, params):
                # print(h_z, z)
                p.hess += h_z * z / self.n_samples  # approximate the expected values of z*(H@z)
                # print("p.hess", p.hess.shape)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional) -- a closure that reevaluates the model and returns the loss (default: None)
        """

        loss = None
        if closure is not None:
            loss = closure()

        self.zero_hessian()
        self.set_hessian()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.hess is None:
                    continue

                if self.average_conv_kernel and p.dim() == 4:
                    p.hess = torch.abs(p.hess).mean(dim=[2, 3], keepdim=True).expand_as(p.hess).clone()

                # Perform correct stepweight decay as in AdamW
                p.mul_(1 - group['lr'] * group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 1:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)  # Exponential moving average of Hessian diagonal square values

                exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(p.hess, p.hess, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                k = group['hessian_power']
                denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(group['eps'])

                # make update
                step_size = group['lr'] / bias_correction1
                #p.addcdiv_(exp_avg, denom, value=-step_size)
                p = p.addcdiv_(exp_avg, denom, value=step_size)

        return self.get_params()


    @staticmethod
    @torch.no_grad()
    def hessian_unl_update(p, hess, args, i):
        average_conv_kernel = False
        weight_decay = 0.0
        betas = (0.9, 0.999)
        hessian_power = 1.0
        eps = args.lr # 1e-8

        if average_conv_kernel and p.dim() == 4:
            hess = torch.abs(hess).mean(dim=[2, 3], keepdim=True).expand_as(hess).clone()

        # Perform correct stepweight decay as in AdamW
        # p = p.mul_(1 - args.lr * weight_decay)

        state = {}
        state["hessian"] = 1

        # State initialization
        if len(state) == 1:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)  # Exponential moving average of gradient values
            state['exp_hessian_diag_sq'] = torch.zeros_like(
                p.data)  # Exponential moving average of Hessian diagonal square values

        exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']
        beta1, beta2 = betas
        state['step'] = i

        # Decay the first and second moment running average coefficient

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        #exp_hessian_diag_sq.mul_(beta2).addcmul_(p_hs.hess, p_hs.hess, value=1 - beta2)
        #print(exp_hessian_diag_sq.device, hess.device)
        exp_hessian_diag_sq.mul_(beta2).addcmul_(hess, hess, value=1 - beta2)

        bias_correction1 = 1 #- beta1 ** state['step']
        bias_correction2 = 1 #- beta2 ** state['step']

        k = hessian_power
        denom = (exp_hessian_diag_sq / bias_correction2).pow_(k / 2).add_(eps)

        # make update
        step_size = args.lr / bias_correction1
        # p.addcdiv_(exp_avg, denom, value=-step_size)
        p = p.addcdiv_(exp_avg, denom, value=step_size * 0.1)
        #p_hs.data = p_hs.data + args.lr * p_hs.grad.data * 10
        return exp_avg, denom, step_size


def add_awgn_noise(tensor, snr):
    """Adds AWGN noise to a tensor in PyTorch.

    Args:
    tensor (torch.Tensor): The input tensor.
    snr (float): The desired signal-to-noise ratio (in dB).

    Returns:
    torch.Tensor: The tensor with added AWGN noise.
    """
    # Calculate signal power and convert SNR from dB
    sig_power = torch.mean(tensor ** 2).cuda()
    snr_linear = 10 ** (snr / 10)  # Convert SNR from dB to linear scale

    # Calculate noise power based on SNR
    noise_power = sig_power / snr_linear

    # Check if noise_power is less than 0
    if noise_power < 0:
        raise ValueError(f"Calculated noise power is less than 0. Check your SNR value. Noise power: {noise_power}, SNR: {snr}")

    # Generate noise with calculated power
    std_dev = torch.sqrt(torch.abs(noise_power))
    std_dev = torch.clamp(std_dev, min=1e-10)
    std_dev = torch.where(torch.isnan(std_dev), torch.tensor(0.3862), std_dev)
    noise = torch.normal( 0.0, torch.abs(std_dev)).cuda()

    noisy_tensor = tensor + noise  # Add noise to original signal

    return noisy_tensor


def add_rayleigh_noise(tensor, snr):
    """Adds Rayleigh noise to a tensor in PyTorch.

    Args:
    tensor (torch.Tensor): The input tensor.
    snr (float): The desired signal-to-noise ratio (in dB).

    Returns:
    torch.Tensor: The tensor with added Rayleigh noise.
    """
    # Calculate signal power and convert SNR from dB
    sig_power = torch.mean(tensor ** 2).cuda()
    snr_linear = 10 ** (snr / 10)  # Convert SNR from dB to linear scale

    # Calculate noise power based on SNR
    noise_power = sig_power / snr_linear + 1e-10

    # Generate noise with calculated power
    sigma = torch.sqrt(noise_power / 2)
    noise = torch.normal(0.0, sigma).cuda()

    noisy_tensor = tensor + noise  # Add noise to original signal

    return noisy_tensor

def add_rician_noise(tensor, snr, K):
    """Adds Rician noise to a tensor in PyTorch.

    Args:
    tensor (torch.Tensor): The input tensor.
    snr (float): The desired signal-to-noise ratio (in dB).
    K (float): Rician factor (ratio of the deterministic power to the scattered power)

    Returns:
    torch.Tensor: The tensor with added Rician noise.
    """
    # Calculate signal power and convert SNR from dB
    sig_power = torch.mean(tensor ** 2).cuda()
    snr_linear = 10 ** (snr / 10)  # Convert SNR from dB to linear scale

    # Calculate noise power based on SNR and Rician K factor
    noise_power = sig_power / ((K+1)*snr_linear) + 1e-10
    s = torch.sqrt(noise_power)

    # Calculate the non-centrality parameter based on Rician K factor
    v = torch.sqrt(K) * s

    X = torch.normal(v, s).cuda()
    Y = torch.normal(0.0, s).cuda()
    noise = torch.sqrt(X**2 + Y**2)

    noisy_tensor = tensor + noise  # Add noise to original signal

    return noisy_tensor




class QAM64Encoder(nn.Module):
    # 定义神经网络
    def __init__(self, qam_modulation, n_feature=192, h_dim=3 * 32, n_output=10):
        # 初始化数组，参数分别是初始化信息，特征数，隐藏单元数，输出单元数
        super(QAM64Encoder, self).__init__()
        self.fc1 = nn.Linear(n_feature, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, n_output * 6)  # output,  original output_pixels * 6 bits per pixel for 64QAM
        # QAM modulation
        self.qam_modulation = qam_modulation.to('cuda')

    # 设置隐藏层到输出层的函数

    def forward(self, x):
        # 定义向前传播函数
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        # Convert to bits
        x = torch.sigmoid(x)
        x = (x > 0.5).float()

        # Reshape to (batch_size, symbol_count, bits_per_symbol)
        x = x.view(x.shape[0], -1, 6).to('cuda')

        # QAM modulation
        x = self.qam_modulation(x)

        return x


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



class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(func.relu(self.bn1(x)))
        y = self.conv2(func.relu(self.bn2(y)))
        x = torch.cat([y, x], 1)
        return x


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(func.relu(self.bn(x)))
        x = func.avg_pool2d(x, 2)
        return x


class DenseNet(nn.Module):
    def __init__(self, block, num_block, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = func.avg_pool2d(func.relu(self.bn(x)), 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12, num_classes=3*32*32)


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


class ResBlockForDecoder(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlockForDecoder, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlockForDecoder(64, 48, 2, 2, 0, 'decode')  # 48 4 4
        self.rb2 = ResBlockForDecoder(48, 48, 2, 2, 0, 'decode')  # 48 8 8
        self.rb3 = ResBlockForDecoder(48, 32, 3, 1, 1, 'decode')  # 32 8 8
        self.rb4 = ResBlockForDecoder(32, 32, 2, 2, 0, 'decode')  # 32 16 16
        self.rb5 = ResBlockForDecoder(32, 16, 3, 1, 1, 'decode')  # 16 16 16
        self.rb6 = ResBlockForDecoder(16, 16, 2, 2, 0, 'decode')  # 16 32 32
        self.out_conv = nn.ConvTranspose2d(16, 3, 3, 1, 1)  # 3 32 32
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        rb1 = self.rb1(inputs)
        rb2 = self.rb2(rb1)
        rb3 = self.rb3(rb2)
        rb4 = self.rb4(rb3)
        rb5 = self.rb5(rb4)
        rb6 = self.rb6(rb5)
        out_conv = self.out_conv(rb6)
        output = self.tanh(out_conv)
        return output

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


def normalize_to_range(data, low=0, high=15):
    min_val = data.min()
    max_val = data.max()

    # 将数据规范化到0-1的范围
    data_normalized = (data - min_val) / (max_val - min_val)

    # 将数据映射到0-63的范围
    data_scaled = (data_normalized * (high - low)).round().long()

    return data_scaled


def denormalize_from_range(data, min_val, max_val, low=0, high=15):
    # 将数据从0-63的范围转换回0-1的范围
    data_normalized = data.float() / (high - low)

    # 使用最小值和最大值将数据从0-1的范围映射回原始的范围
    data_denormalized = data_normalized * (max_val - min_val) + min_val

    return data_denormalized

# 1. 定义64-QAM星座图
x = torch.linspace(-3,3,4).cuda() # torch.linspace(-7,7,8).cuda() # -15， 15， 16 for 256QAM -3,3,4 for 16QAM
X,Y = torch.meshgrid(x, x)
constellation = (X + 1j*Y).flatten()

# 2. 编码函数
def encode_qam64(data):
    return constellation[data]

# 3. 解码函数
def decode_qam64(data):
    distances = abs(data.view(-1, 1) - constellation.view(1, -1))
    return torch.argmin(distances, dim=1).cuda()

def old_process_64QAM(logits_z):
    B, dimZ = logits_z.shape
    logits_z = logits_z * 10000000
    #print(logits_z)
    data_scaled = normalize_to_range(logits_z)

    symbols = encode_qam64(data_scaled)
    decoded_data = decode_qam64(symbols)
    min_val = logits_z.min()
    max_val = logits_z.max()
    data_denormalized = denormalize_from_range(decoded_data, min_val, max_val)
    data_denormalized = data_denormalized.reshape((B, -1))
    data_denormalized = data_denormalized / 10000000
    #print(data_denormalized)


    return data_denormalized




# 将浮点数转换为字符串
def float_to_string(data):
    return np.array2string(data, precision=7, separator=',', suppress_small=True)

# 将字符串转换为二进制数据
def string_to_bin(data):
    return ''.join(format(ord(c), '08b') for c in data)


# 将二进制数据转换为64-QAM符号
def bin_to_qam64(data):
    symbols = []
    for i in range(0, len(data), 6):
        symbol = int(data[i:i+6], 2)
        symbols.append(symbol)
    return symbols

# 然后在接收端，做相反的操作
def qam64_to_bin(symbols):
    return ''.join(format(symbol, '06b') for symbol in symbols)

# def bin_to_string(data):
#     return ''.join(chr(int(data[i:i+8], 2)) for i in range(0, len(data), 8))

def bin_to_string(data):
    # 确保二进制数据的长度是8的倍数
    if len(data) % 8 != 0:
        data = data[:-(len(data) % 8)]
    return ''.join(chr(int(data[i:i+8], 2)) for i in range(0, len(data), 8))




def string_to_float(data):
    data = data.replace('[','').replace(']','').split(',')
    float_list = []
    for string in data:
        # 使用正则表达式移除非数字字符
        string = re.sub(r'[^\d.-]', '', string)
        try:
            float_list.append(float(string))
        except ValueError:
            print(f"Could not convert {string} to float.")
    return float_list

def process_64QAM(logits_z):
    B, dimZ = logits_z.shape
    logits_z = logits_z.reshape((B, -1))

   #print(logits_z)
    logits_z = logits_z.cpu()
    logits_z_np = logits_z.detach().numpy()
    logits_z_str = float_to_string(logits_z_np)
    logits_z_bin = string_to_bin(logits_z_str)
    logits_z_symbols = bin_to_qam64(logits_z_bin)

    logits_z_received_bin = qam64_to_bin(logits_z_symbols)
    received_str = bin_to_string(logits_z_received_bin)
    received_float = string_to_float(received_str)


    logits_z_after_64QAM = torch.tensor(received_float).reshape((B, -1)).cuda()

    #print('logits_z_after_64QAM',logits_z_after_64QAM)
    return logits_z_after_64QAM


class Mine1(nn.Module):

    def __init__(self, noise_size=49, sample_size=28*28, output_size=1, hidden_size=128):
        super(Mine1, self).__init__()
        self.fc1_noise = nn.Linear(noise_size, hidden_size, bias=False)
        self.fc1_sample = nn.Linear(sample_size, hidden_size, bias=False)
        self.fc1_bias = nn.Parameter(torch.zeros(hidden_size))
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.ma_et = None

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, noise, sample):
        x_noise = self.fc1_noise(noise)
        x_sample = self.fc1_sample(sample)
        x = F.leaky_relu(x_noise + x_sample + self.fc1_bias, negative_slope=2e-1)
        x = F.leaky_relu(self.fc2(x), negative_slope=2e-1)
        x = F.leaky_relu(self.fc3(x), negative_slope=2e-1)
        return x


def calculate_MI(X, Z, Z_size, M, M_opt, args, ma_rate=0.001):
    '''
    we use Mine to calculate the mutual information between two layers of networks.
    :param G:
    :param M:
    :param ma_rate:
    :return:
    '''

    z_bar = torch.randn((args.batch_size, Z_size)).to(args.device)

    et = torch.mean(torch.exp(M(X, z_bar)))

    if M.ma_et is None:
        M.ma_et = et.detach().item()

    M.ma_et += ma_rate * (et.detach().item() - M.ma_et)

    #z = torch.narrow(z, dim=1, start=0, length=3)  # slice for MI
    mutual_information = torch.mean(M(X, Z)) \
                         - torch.log(et) * et.detach() / M.ma_et

    loss = - mutual_information

    M_opt.zero_grad()
    loss.backward()
    M_opt.step()

    return mutual_information.item()


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
        elif mode == 'with64QAM_distribution':
            B, double_dimZ = double_logits_z.shape
            #double_dimZ_after_QAM = process_64QAM(double_logits_z)
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
        elif mode == '64QAM_distribution':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            #print(logits_z)
            # convert to bit
            logits_z_sig = torch.sigmoid(logits_z)
            logits_z_sig = (logits_z_sig > 0.5).float()

            # Reshape to (batch_size, symbol_count, bits_per_symbol)
            logits_z_sig = logits_z_sig.view(logits_z_sig.shape[0], -1, 6).to('cuda')

            # QAM modulation
            logits_z_arr = self.qam_modulation(logits_z_sig)

            # change QAM modulated complex number to tensor
            real_part = torch.real(logits_z_arr)
            imag_part = torch.imag(logits_z_arr)

            input_tensor = torch.cat([real_part.unsqueeze(1), imag_part.unsqueeze(1)], dim=1).cuda()

            logits_y = self.approximator(input_tensor)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            return logits_z, logits_y, mu, logvar

        elif mode == 'with_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            logits_y = self.approximator(logits_z)  # (B , 10)
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'with64QAM_reconstruction':
            logits_z, mu, logvar = self.explain(x, mode='with64QAM_distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            #logits_z_after_QAM = process_64QAM(logits_z)
            logits_y = self.approximator(logits_z)  # (B , 10) logits_z_after_QAM
            logits_y = logits_y.reshape((B, 10))  # (B,   10)
            x_hat = self.reconstruction(logits_z)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'with64QAM_reconstruction_CIFAR':
            logits_z, mu, logvar = self.explain(x, mode='with64QAM_distribution')  # (B, C, H, W), (B, C* h* w)
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            #logits_z_with_awgn = add_awgn_noise(logits_z, torch.tensor(args.SNR).cuda())

            #logits_z_after_QAM = process_64QAM(logits_z)



            #logits_z_with_awgn = logits_z_with_awgn.view(logits_z_with_awgn.size(0), 3, 7, 7)
            logits_z_with_awgn = logits_z.view(logits_z.size(0), 3, 32, 32)
            logits_y = self.approximator(logits_z_with_awgn)  # (B , 10)

            logits_y = logits_y.reshape((B, 20))  # (B,   10)
            x_hat = self.cifar_recon(logits_z_with_awgn)
            return logits_z, logits_y, x_hat, mu, logvar
        elif mode == 'VAE':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            # in general, a larger K factor in Rician noise could be considered "better" from a signal quality perspective
            #logits_z_with_awgn = add_rician_noise(logits_z, torch.tensor(args.SNR).cuda(), K=torch.tensor(2).cuda())  # add_awgn_noise ,  add_rayleigh_noise,
            logits_z_with_awgn = add_awgn_noise(logits_z, torch.tensor(args.SNR).cuda())
            x_hat = self.reconstruction(logits_z_with_awgn)
            return logits_z, x_hat, mu, logvar
        elif mode == 'VAE_CIFAR':
            logits_z, mu, logvar = self.explain(x, mode='distribution')  # (B, C, H, W), (B, C* h* w)
            # VAE is not related to labels
            # print("logits_z, mu, logvar", logits_z, mu, logvar)
            # logits_y = self.approximator(logits_z)  # (B , 10)
            # logits_y = logits_y.reshape((B, 10))  # (B,   10)
            # in general, a larger K factor in Rician noise could be considered "better" from a signal quality perspective
            #logits_z_with_awgn = add_rician_noise(logits_z, torch.tensor(args.SNR).cuda(), K=torch.tensor(2).cuda())  # add_awgn_noise ,  add_rayleigh_noise,
            logits_z_with_awgn = add_awgn_noise(logits_z, torch.tensor(args.SNR).cuda())
            #logits_z_with_awgn = logits_z_with_awgn.view(logits_z_with_awgn.size(0), 3, 7, 7)
            logits_z_with_awgn = logits_z_with_awgn.view(logits_z_with_awgn.size(0), 3, 32, 32)
            x_hat = self.cifar_recon(logits_z_with_awgn)
            x_hat_ap = x_hat.view(x_hat.size(0), 3, 32, 32)
            logits_y = self.approximator(x_hat_ap)
            return logits_y, logits_z, x_hat, mu, logvar
        elif mode == '64QAM_VAE':
            logits_z, mu, logvar = self.explain(x, mode='with64QAM_distribution')  # (B, C, H, W), (B, C* h* w)
            #print(logits_z)
            #logits_z_after_QAM = process_64QAM(logits_z)
            #print("input_tensor",input_tensor.shape)
            x_hat = self.reconstruction(logits_z) #logits_z_after_QAM
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
        output_x = self.decoder(logits_z)
        return output_x #torch.sigmoid(output_x)

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
        #encoder = QAM64Encoder(qam_modulation=qam_modulation, n_feature=28 * 28, n_output=args.dimZ * 2)  # resnet18(1, 49*2) #
        encoder = LinearModel(n_feature=28 * 28, n_output=args.dimZ * 2)  # 64QAM needs 6 bits
        lr = args.lr

    elif args.dataset == 'CIFAR10':
        # approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = LinearModel(n_feature=3*32*32)# resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        #decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        decoder = resnet18(3, 3*32*32) #Decoder() # DenseNet(num_classes=3*32*32, small_inputs=False) #resnet18(3, 3*32*32)
        lr = args.lr

    elif args.dataset == 'CIFAR100':
        # approximator = resnet18(3, 10) #LinearModel(n_feature=args.dimZ)
        approximator = resnet18(3, 20) # LinearModel(n_feature=3*32*32, n_output=100)# LinearModel(n_feature=args.dimZ)
        encoder = resnet18(3, args.dimZ * 2)  # resnet18(1, 49*2)
        #decoder = LinearModel(n_feature=args.dimZ, n_output=3 * 32 * 32)
        decoder = resnet18(3, 3*32*32) #Decoder() # DenseNet(num_classes=3*32*32, small_inputs=False) #resnet18(3, 3*32*32)
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
        #x = x.view(x.size(0), -1)

        logits_y, logits_z, x_hat, mu, logvar = model(x, mode='VAE_CIFAR')  # (B, C* h* w), (B, N, 10)
        # VAE two loss: KLD + MSE

        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
        KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
        KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        # x = torch.sigmoid(torch.relu(x))
        MSE = reconstruction_function(x_hat, x)  # mse loss
        H_p_q = loss_fn(logits_y, y)
        loss = args.beta * KLD_mean + MSE *30  + H_p_q/30 # / (args.batch_size * 28 * 28)

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
            'MSE': MSE.item(),
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
        if step % len(train_loader) % 6000 == 0:
            print(f'[{epoch}/{0 + args.num_epochs}:{step % len(train_loader):3d}] '
                  + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            plt.show()
            #acc_norm_g = eva_vae_generation(model, classifier_model, train_loader, args, name='generated test')
    return model, mu_list, sigma_list


def plot_latent(autoencoder, data_loader, args, num_batches=100):
    for i, (x, y) in enumerate(data_loader):
        if args.dataset == "MNIST":
            x = x.view(x.size(0), -1)
        logits_y, z, x_hat, mu, logvar = autoencoder(x.to(args.device), mode='VAE_CIFAR')
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

def resnet_train(data_loader, model, loss_fn, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, (x, y) in enumerate(data_loader):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        #x = x.view(x.size(0), -1)

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
def test_resnet_acc(model, data_loader, args, name='test', epoch=999):
    num_total = 0
    num_correct = 0
    model.eval()
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        #x = x.view(x.size(0), -1)
        out = model(x)
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (out.argmax(dim=1) == y).sum().item()
        num_total += len(x)
    acc = num_correct / num_total
    acc = round(acc, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}')
    return acc

#@torch.no_grad()
def eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='test', epoch=999):
    # first, generate x_hat from trained vae
    rec_function = nn.MSELoss(size_average=False)

    vib.eval()
    classifier_model.eval()
    num_total = 0
    num_correct = 0
    mi_for_xz_total = 0
    mi_for_xx_total = 0
    mi_for_xxhat_total = 0
    BCE_total = 0
    for batch_idx, (x, y) in enumerate(dataloader_erase):
        x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
        #x = x.view(x.size(0), -1)
        logits_y, logits_z, x_hat, mu, logvar = vib(x, mode='VAE_CIFAR')  # (B, C* h* w), (B, N, 10)


        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        if batch_idx == 0:
            x_hat_cpu = x.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid2 = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            plt.imshow(np.transpose(grid2, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            plt.show()
        if batch_idx == 0:
            x_hat_cpu = x_hat.cpu().data
            x_hat_cpu = x_hat_cpu.clamp(0, 1)
            x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            grid2 = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            plt.imshow(np.transpose(grid2, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            plt.show()

        x_hat = x_hat.view(x_hat.size(0), 3, 32, 32)
        # second, input the x_hat to classifier
        # logits_y = classifier_model(x_hat.detach())
        if y.ndim == 2:
            y = y.argmax(dim=1)
        num_correct += (logits_y.argmax(dim=1) == y).sum().item()
        num_total += len(x)

        x_hat = x_hat.view(x_hat.size(0), -1)
        x = x.view(x.size(0), -1)
        BCE = rec_function(x_hat, x)
        BCE_total += BCE.item()
        if batch_idx == 0:
            # x_hat_cpu = x_hat_for_show.cpu().data
            # x_hat_cpu = x_hat_cpu.clamp(0, 1)
            # x_hat_cpu = x_hat_cpu.view(x_hat_cpu.size(0), 3, 32, 32)
            # grid = torchvision.utils.make_grid(x_hat_cpu, nrow=4, cmap="gray")
            # plt.imshow(np.transpose(grid, (1, 2, 0)))  # 交换维度，从GBR换成RGB
            # plt.show()

            M_for_xz = Mine1(noise_size=3* 32 * 32, sample_size=args.dimZ)
            M_for_xz.to(args.device)
            M_for_xz_opt = torch.optim.Adam(M_for_xz.parameters(), lr=2e-4)

            B, Z_size = logits_z.shape
            for i in range(args.mi_epoch):
                mi_for_xz = calculate_MI(x.detach(), logits_z.detach(), Z_size, M_for_xz, M_for_xz_opt, args,
                                         ma_rate=0.001)
                if mi_for_xz < 0:
                    i = i - 1
            mi_for_xz_total += mi_for_xz

            M_for_xx = Mine1(noise_size=3* 32 * 32, sample_size=3* 32 * 32)
            M_for_xx.to(args.device)
            M_for_xx_opt = torch.optim.Adam(M_for_xx.parameters(), lr=2e-4)

            B, Z_size = x.shape
            for i in range(args.mi_epoch):
                mi_for_xx = calculate_MI(x.detach(), x.detach(), Z_size, M_for_xx, M_for_xx_opt, args, ma_rate=0.001)
                if mi_for_xx < 0:
                    i = i - 1
            mi_for_xx_total += mi_for_xx

            M = Mine1(noise_size=3* 32 * 32, sample_size=3* 32 * 32)
            M.to(args.device)
            M_opt = torch.optim.Adam(M.parameters(), lr=2e-4)

            B, Z_size = x.shape
            for i in range(args.mi_epoch):
                mi = calculate_MI(x.detach(), x_hat.detach(), Z_size, M, M_opt, args, ma_rate=0.001)
                if mi < 0:
                    i = i - 1
            mi_for_xxhat_total += mi

    acc = num_correct / num_total
    acc = round(acc, 5)
    avg_bce = round(BCE_total/num_total, 5)
    print(f'epoch {epoch}, {name} accuracy:  {acc:.4f}, MSE:  {avg_bce:.4f}, mi for xz: {mi_for_xz_total:.4f}, mi for xx: {mi_for_xx_total:.4f}, mi for xxhat: {mi_for_xxhat_total:.4f}')
    return acc

def calculate_hs_p(KLD_mean2, BCE, optimizer_hessian, vibi_f_hessian):
    loss = args.beta * KLD_mean2 + BCE  # + BCE / (args.local_bs * 28 * 28)
    optimizer_hessian.zero_grad()

    # loss.backward()
    # log_probs = net(images)
    # loss = self.loss_func(log_probs, labels)

    loss.backward(create_graph=True)

    optimizer_hs = AdaHessian(vibi_f_hessian.parameters())
    # optimizer_hs.get_params()
    optimizer_hs.zero_hessian()
    optimizer_hs.set_hessian()

    params_with_hs = optimizer_hs.get_params()
    # optimizer_hessian.step()
    optimizer_hessian.zero_grad()
    vibi_f_hessian.zero_grad()

    return params_with_hs

def unlearning_hessian(vib, args, dataloader_erase, remaining_set, reconstruction_function, classifier_model,loss_fn):
    vibi_f_hessian, lr = init_vib(args)
    vibi_f_hessian.to(args.device)
    vibi_f_hessian.load_state_dict(vib.state_dict())
    optimizer_hessian = torch.optim.Adam(vibi_f_hessian.parameters(), lr=args.lr)

    init_epoch = 0
    print("unlearning")

    acc_test = []
    backdoor_acc_list = []



    #dataloader_remain = DataLoader(remaining_set, batch_size=remaining_set.__len__(), shuffle=True)
    dataloader_remain2 = DataLoader(remaining_set, batch_size=args.batch_size, shuffle=True)

    train_bs = 0
    # for batch_idx, (images, labels) in enumerate(dataloader_remain):
    #     images, labels = images.to(args.device), labels.to(args.device)
    #     B, c, h, w = images.shape
    #     # print(B,h,w)
    #     if args.dataset == 'MNIST':
    #         images = images.reshape((B, -1))
    #     vibi_f_hessian.zero_grad()
    #     print('batch_idx', batch_idx)
    #     logits_z, x_hat, mu, logvar = vibi_f_hessian(images, mode='VAE')  # (B, C* h* w), (B, N, 10)
    #     #H_p_q = loss_fn(logits_y, labels)
    #     BCE = reconstruction_function(x_hat, images)  # mse loss = - log p = log 1/p
    #
    #     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
    #     KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
    #     KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()
    #
    #     loss = args.beta * KLD_mean + BCE  # + BCE / (args.local_bs * 28 * 28)
    #     optimizer_hessian.zero_grad()
    #
    #     loss.backward(create_graph=True)
    #
    #     optimizer_hs = AdaHessian(vibi_f_hessian.parameters())
    #     # optimizer_hs.get_params()
    #     optimizer_hs.zero_hessian()
    #     optimizer_hs.set_hessian()
    #
    #     params_with_hs = optimizer_hs.get_params()
    #
    #     optimizer_hessian.zero_grad()
    #     vibi_f_hessian.zero_grad()

    if init_epoch == 0 or args.resume_training:

        print('Unlearning VIBI KLD')
        for iter in range(args.num_epochs):  # self.args.local_ep
            batch_loss = []
            # print(iter)

            for (images, labels), (images2, labels2) in zip(dataloader_erase, dataloader_remain2):
                # for batch_idx, (images, labels) in enumerate(self.erased_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                B, c, h, w = images.shape
                # print(B,h,w)
                if args.dataset == 'MNIST':
                    images = images.reshape((B, -1))

                images2, labels2 = images2.to(args.device), labels2.to(args.device)
                B, c, h, w = images2.shape
                if args.dataset == 'MNIST':
                    images2 = images2.reshape((B, -1))

                vibi_f_hessian.zero_grad()
                logits_y, logits_z, x_hat, mu, logvar = vibi_f_hessian(images, mode='VAE_CIFAR')  # (B, C* h* w), (B, N, 10)
                logits_y2, logits_z2, x_hat2, mu2, logvar2 = vibi_f_hessian(images2, mode='VAE_CIFAR')
                x_hat = x_hat.view(x_hat.size(0), 3, 32, 32)
                #logits_y = classifier_model(x_hat.detach())
                ##remaining dataset used to unlearn

                x_hat = x_hat.view(x_hat.size(0), -1)
                x_hat2 = x_hat2.view(x_hat2.size(0), -1)
                images = images.view(images.size(0), -1)
                images2 = images2.view(images2.size(0), -1)
                #H_p_q = loss_fn(logits_y, labels)
                BCE = reconstruction_function(x_hat, images)
                KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).cuda()
                KLD = torch.sum(KLD_element).mul_(-0.5).cuda()
                KLD_mean = torch.mean(KLD_element).mul_(-0.5).cuda()

                #H_p_q2 = loss_fn(logits_y2, labels2)
                BCE2 = reconstruction_function(x_hat2, images2)
                KLD_element2 = mu2.pow(2).add_(logvar2.exp()).mul_(-1).add_(1).add_(logvar2).cuda()
                KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).cuda()

                params_with_hs = calculate_hs_p(KLD_mean2, BCE2, optimizer_hessian, vibi_f_hessian)

                H_p_q = loss_fn(logits_y, labels)

                loss = args.beta * KLD_mean + BCE * 30 + H_p_q/30  # + BCE / (args.local_bs * 28 * 28)
                optimizer_hessian.zero_grad()

                loss.backward()
                # log_probs = net(images)
                # loss = self.loss_func(log_probs, labels)

                i = 0
                for p_hs, p in zip(params_with_hs, vibi_f_hessian.parameters()):
                    i = i + 1
                    # if i==1:
                    #     continue
                    # print(p_hs.hess)
                    # break
                    temp_hs = torch.tensor(p_hs.hess).cuda()
                    # temp_hs = temp_hs.__add__(args.lr)
                    # p.data = p.data.addcdiv_(exp_avg, denom, value=-step_size * 10000)

                    # print(p.data)
                    # p.data = p_hs.data.addcdiv_(exp_avg, denom, value=step_size * args.lr)
                    if p.grad != None:
                        exp_avg, denom, step_size = AdaHessian.hessian_unl_update(p, temp_hs, args, i)
                        # print(exp_avg)
                        # print(denom)
                        p.data = p.data.addcdiv_(exp_avg, denom, value=args.hessian_rate)
                        # p.data =p.data + torch.div(p.grad.data, temp_hs) * args.lr #torch.mul(p_hs.hess, p.grad)*10
                        # print(p.grad.data.shape)
                    else:
                        p.data = p.data

                vibi_f_hessian.zero_grad()

                #classifying from the backdoored model
                fl_acc = (logits_y.argmax(dim=1) == labels).float().mean().item()
                #fl_acc2 = (logits_y2.argmax(dim=1) == labels2).float().mean().item()
                #temp_acc.append(fl_acc2)
                #temp_back.append(fl_acc)
                train_bs = train_bs + 1
                if fl_acc < 0.02:
                    break
                batch_loss.append(loss.item())

            #valid_acc = test_accuracy(vibi_f_hessian, test_loader, args, name='vibi valid top1')
            #interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            #print("test_acc", valid_acc)
            #epoch_test_acc.append(valid_acc)
            # valid_acc_old = valid_acc
            # interpolate_valid_acc = torch.linspace(valid_acc_old, valid_acc, steps=len(train_loader)).tolist()
            # print("test_acc", valid_acc)
            # acc_test.append(valid_acc)
            back_acc = eva_vae_generation(vibi_f_hessian, classifier_model, dataloader_erase, args, name='hessian_b_unl generated backdoor')
            backdoor_acc_list.append(back_acc)
            print("backdoor_acc", backdoor_acc_list)
            # print("acc_test: ", acc_test)
            if back_acc < args.back_acc_threshold:
                print()
                print("end hessian unl", train_bs)
                break
        # print("temp_acc", temp_acc)
        # print("temp_back", temp_back)
    return vibi_f_hessian

def unlearning_vae(vib, args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, loss_fn, train_type):

    rec_function = nn.MSELoss(size_average=False)

    vib_unl, lr = init_vib(args)
    vib_unl.to(args.device)
    vib_unl.load_state_dict(vib.state_dict())
    optimizer_unl = torch.optim.Adam(vib_unl.parameters(), lr=lr)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    print(len(dataloader_erase.dataset))
    train_bs = 0
    clean_acc = []
    erassed_acc = []
    clean_mse = []
    erased_mse=[]

    for epoch in range(args.num_epochs):
        vib_unl.train()
        batch_idx = 0
        for (x, y), (x2, y2) in zip(dataloader_erase, dataloader_remain):
            if x.size(0) != x2.size(0):
                continue
            x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x = x.view(x.size(0), -1)

            x2, y2 = x2.to(args.device), y2.to(args.device)  # (B, C, H, W), (B, 10)
            if args.dataset == 'MNIST':
                x2 = x2.view(x2.size(0), -1)

            logits_y_e, logits_z_e, x_hat_e, mu_e, logvar_e = vib_unl(x, mode='VAE_CIFAR')
            logits_y_e2, logits_z_e2, x_hat_e2, mu_e2, logvar_e2 = vib_unl(x2, mode='VAE_CIFAR')

            #use self-generated as postive for a already trained VAE
            x_hat_e2_input = x_hat_e2.detach().view(x_hat_e2.detach().size(0), 3, 32, 32)
            #             print("shape",x_hat_e2_input.shape)
            logits_y_e3, logits_z_e3, x_hat_e3, mu_e3, logvar_e3 = vib_unl(x_hat_e2_input, mode='VAE_CIFAR')
            logits_y_f, logits_z_f, x_hat_f, mu_f, logvar_f = vib(x, mode='VAE_CIFAR')

            x_hat_e = x_hat_e.view(x_hat_e.size(0), 3, 32, 32)
            #logits_y = classifier_model(x_hat_e.detach())

            logits_z_e_log_softmax = logits_z_e.log_softmax(dim=1)
            p_x_e = x.softmax(dim=1)
            B = x.size(0)

            KLD_element2 = mu_e2.pow(2).add_(logvar_e2.exp()).mul_(-1).add_(1).add_(logvar_e2).to(args.device)
            KLD_mean2 = torch.mean(KLD_element2).mul_(-0.5).to(args.device)

            # KLD = 0.5 * torch.mean(
            #     logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            # KLD between erased z and original z
            KLD_mean = 0.5 * torch.mean(
                logvar_f - logvar_e + (torch.exp(logvar_e) + (mu_e - mu_f).pow(2)) / torch.exp(logvar_f) - 1).cuda()

            KL_z_r = (torch.exp(logits_z_e_log_softmax) * logits_z_e_log_softmax).sum(dim=1).mean() + math.log(
                logits_z_e_log_softmax.shape[1])

            # or calculate similarity directly based on z_e and z_r


            similarity_of_ze_zr = (1 - cos(logits_z_e, logits_z_e2)).sum() #1.0 -
            similarity_positive = (1 - cos(logits_z_e2, logits_z_e3)).sum()

            # x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            # x_hat_e = torch.sigmoid(reconstructor(logits_z_e))
            x_hat_e = x_hat_e.view(x_hat_e.size(0), -1)
            x_hat_e2 = x_hat_e2.view(x_hat_e2.size(0), -1)

            x = x.view(x.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            # x = torch.sigmoid(x)
            BCE = reconstruction_function(x_hat_e, x)  # mse loss = - log p = log 1/p
            BCE2 = reconstruction_function(x_hat_e2, x2)
            # BCE = torch.mean(x_hat_e.log_softmax(dim=1))
            MSE_erased = rec_function(x_hat_e, x)
            MSE_clean = rec_function(x_hat_e2, x2)

            e_log_p = torch.exp(BCE / (args.batch_size * 28 * 28))  # = 1/p

            log_z = torch.mean(logits_z_e.log_softmax(dim=1))

            kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

            H_p_q = loss_fn(logits_y_e, y)
            H_p_q2 = loss_fn(logits_y_e2, y2)
            # Calculate the L2-norm


            l2_norm_unl = torch.norm(args.kld_to_org * KLD_mean  - args.unlearn_bce * BCE - args.unlearn_bce * H_p_q/30 , p=2)

            l2_norm_ss = torch.norm(args.self_sharing_rate * (args.beta * KLD_mean2 + similarity_positive / similarity_of_ze_zr + BCE2 + H_p_q2/30), p=2)


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
                loss = args.kld_to_org * KLD_mean - args.unlearn_bce * BCE - args.unlearn_bce * H_p_q/30
            elif train_type == 'VAE_unl_ss':
                loss = (args.kld_to_org * KLD_mean - args.unlearn_bce * BCE - args.unlearn_bce * H_p_q/30 ) * unl_rate + self_s_rate * args.self_sharing_rate * (
                               args.beta * KLD_mean2 + similarity_positive / similarity_of_ze_zr + BCE2*30 + H_p_q2/30)  # args.beta * KLD_mean - H_p_q + args.beta * KLD_mean2  + H_p_q2 #- log_z / e_log_py #-   # H_p_q + args.beta * KLD_mean2
            elif train_type == 'VBU':
                loss = args.beta * KLD_mean*0.0 + args.unl_r_for_bayesian * (- BCE) + args.unl_r_for_bayesian * (- H_p_q)/30

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
                'similarity_positive': similarity_positive,
                'similarity_of_ze_zr': similarity_of_ze_zr,
                # 'H(p,q)': H_p_q.item(),
                # 'kl_f_e': kl_f_e.item(),
                # 'H_p_q2': H_p_q2.item(),
                # '1-JS(p,q)': JS_p_q,
                # 'mu_e': torch.mean(mu_e).item(),
                # 'sigma_e': torch.sqrt_(torch.exp(logvar_e)).mean().item(),
                # 'KLD': KLD.item(),
                'e_log_p': e_log_p.item(),
                'log_z': log_z.item(),
                'KLD_mean': KLD_mean.item(),
            }

            clean_acc_temp = (logits_y_e2.argmax(dim=1) == y2).float().mean().item()
            acc_back_temp = (logits_y_e.argmax(dim=1) == y).float().mean().item()
            clean_acc.append(clean_acc_temp)
            erassed_acc.append(acc_back_temp)
            clean_mse.append(MSE_clean.item())
            erased_mse.append(MSE_erased.item())

            acc_back = (logits_y_e.argmax(dim=1) == y).float().mean().item()
            # if epoch == args.num_epochs - 1:
            #     mu_list.append(torch.mean(mu_e).item())
            #     sigma_list.append(torch.sqrt_(torch.exp(logvar_e)).mean().item())
            if batch_idx % len(dataloader_erase) % 600 == 0:
                print(f'[{epoch}/{0 + args.num_epochs}:{batch_idx % len(dataloader_erase):3d}] '
                      + ', '.join([f'{k} {v:.3f}' for k, v in metrics.items()]))
            batch_idx = batch_idx + 1
            train_bs = train_bs + 1
            if acc_back < 0.02:
                break

        vib_unl.eval()

        back_acc = eva_vae_generation(vib_unl, classifier_model, dataloader_erase, args, name='test', epoch=epoch)

        if back_acc < args.back_acc_threshold:
            print()
            print("end unlearn, train_bs", train_bs)
            break
    print('final train_bs', train_bs)
    print("clean acc", clean_acc)
    print("erased acc", erassed_acc)
    print("clean mse", clean_mse)
    print("erased mse", erased_mse)
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
args.num_epochs = 20
args.dataset = 'CIFAR100'
args.add_noise = False
args.beta = 0.0001
args.lr = 0.0005
args.dimZ = 3*32*32 #40 #2
args.batch_size = 16
args.erased_local_r = 0.06  # the erased data ratio
args.back_acc_threshold = 0.1

args.mi_epoch = 40
args.SNR = 20
args.kld_to_org = 1/10000
args.unlearn_bce = 10  # beta_u  0.1
args.self_sharing_rate = 10

args.unl_r_for_bayesian = args.unlearn_bce
args.hessian_rate = 0.005

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
    train_set = CIFAR10('../../data/cifar', train=True, transform=train_transform, download=True)
    test_set = CIFAR10('../../data/cifar', train=False, transform=test_transform, download=True)
    train_set_no_aug = CIFAR10('../../data/cifar', train=True, transform=test_transform, download=True)

elif args.dataset == 'CIFAR100':
    train_transform = T.Compose([T.RandomCrop(32, padding=4),
                                 T.ToTensor(),
                                 ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608)),                                 T.RandomHorizontalFlip(),
    test_transform = T.Compose([T.ToTensor(),
                                ])  # T.Normalize((0.4914, 0.4822, 0.4465), (0.2464, 0.2428, 0.2608))
    train_set = CIFAR100Coarse('../../data/cifar', train=True, transform=train_transform, download=True )
    test_set = CIFAR100Coarse('../../data/cifar', train=False, transform=test_transform, download=True )
    train_set_no_aug = CIFAR100Coarse('../../data/cifar', train=True, transform=test_transform, download=True)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

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

base_label=18
trigger_label = 2
add_backdoor=1 # =1 add backdoor , !=1 not add
poison_data, poison_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.batch_size, args=args, add_backdoor=add_backdoor, dp_sample=0)

trigger_label = base_label
add_backdoor = 0
clean_data, clean_targets = create_backdoor_train_dataset(dataname=args.dataset, train_data=train_set,
                                                            base_label=base_label,
                                                            trigger_label=trigger_label, poison_samples=poison_samples,
                                                            batch_size=args.batch_size, args=args, add_backdoor=add_backdoor, dp_sample=0)


clean_dataset = Data.TensorDataset(clean_data, clean_targets)
dataloader_clean = DataLoader(clean_dataset, batch_size=args.batch_size, shuffle=False)

if args.dataset == 'MNIST':
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 1, 28, 28)
    erasing_set.data = erasing_set.data.reshape(len(erasing_set.data), 1, 28, 28)
elif args.dataset == 'CIFAR10':
    train_set_loader = DataLoader(remaining_set, batch_size=1, shuffle=True)  # poison_trainset
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 3, 32, 32)
    temp_img = torch.empty(0, 3, 32, 32).float().cuda()
    temp_label = torch.empty(0).long().cuda()
    for step, (x, y) in enumerate(train_set_loader):
        x, y = x.to(args.device), y.to(args.device)
        temp_img = torch.cat([temp_img, x], dim=0)
        temp_label = torch.cat([temp_label, y], dim=0)
    data_reshape = temp_img
elif args.dataset == 'CIFAR100':
    train_set_loader = DataLoader(remaining_set, batch_size=1, shuffle=True)  # poison_trainset
    data_reshape = remaining_set.data.reshape(len(remaining_set.data), 3, 32, 32)
    temp_img = torch.empty(0, 3, 32, 32).float().cuda()
    temp_label = torch.empty(0).long().cuda()
    for step, (x, y) in enumerate(train_set_loader):
        x, y = x.to(args.device), y.to(args.device)
        temp_img = torch.cat([temp_img, x], dim=0)
        temp_label = torch.cat([temp_label, y], dim=0)
    data_reshape = temp_img

print('train_set.data.shape', train_set.data.shape)
print('poison_data.shape', poison_data.shape)

data = torch.cat([poison_data.to(args.device), data_reshape.to(args.device)], dim=0)
targets = torch.cat([poison_targets.to(args.device), temp_label.to(args.device)], dim=0)

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
dataloader_erase = DataLoader(erasing_set, batch_size=args.batch_size, shuffle=True) #shuffle=True


vib, lr = init_vib(args)
vib.to(args.device)

loss_fn = nn.CrossEntropyLoss()

reconstruction_function = nn.MSELoss(size_average=True)

acc_test = []
print("learning")

print('Training VIB')
print(f'{type(vib.encoder).__name__:>10} encoder params:\t{num_params(vib.encoder) / 1000:.2f} K')
print(f'{type(vib.approximator).__name__:>10} approximator params:\t{num_params(vib.approximator) / 1000:.2f} K')
print(f'{type(vib.decoder).__name__:>10} decoder params:\t{num_params(vib.decoder) / 1000:.2f} K')
# inspect_explanations()

# train backdoor linear model, this model is used to classify the generated images to see if the generative model is backdoored
#classifier_model = LinearModel(n_feature=28 * 28).to(args.device)
classifier_model = resnet18(3, 10).to(args.device)

# back_acc_list = []
# for epoch in range(10):
#     classifier_model.train()
#     classifier_model = resnet_train(dataloader_full, classifier_model, loss_fn, args) # , dataloader_full , train_loader
#     test_acc = test_resnet_acc(classifier_model, test_loader, args, name='test', epoch=epoch)
#     backdoor_acc = test_resnet_acc(classifier_model, dataloader_erase, args, name='backdoor', epoch=epoch)
#
# test_acc = test_resnet_acc(classifier_model, test_loader, args, name='test')
# backdoor_acc = test_resnet_acc(classifier_model, dataloader_erase, args, name='backdoor')
#
#
# classifier_linear_model = LinearModel(n_feature=3 *32 * 32).to(args.device)

# back_acc_list = []
# for epoch in range(2):
#     classifier_linear_model.train()
#     classifier_linear_model = linear_train(dataloader_full, classifier_linear_model, loss_fn, args) # , dataloader_full , train_loader
#     test_acc = test_linear_acc(classifier_linear_model, test_loader, args, name='test', epoch=epoch)
#     backdoor_acc = test_linear_acc(classifier_linear_model, dataloader_erase, args, name='backdoor', epoch=epoch)
#



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
    acc_norm_g = eva_vae_generation(vib, classifier_model, test_loader, args, name='generated test')
    back_g_acc_lsit.append(acc_back_g)

print('mu', np.mean(mu_list), 'sigma', np.mean(sigma_list))

acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='generated backdoor')
acc_norm_g = eva_vae_generation(vib, classifier_model, dataloader_clean, args, name='generated clean')

print('learned model with 64QAM')
for batch_idx, (x, y) in enumerate(test_loader):
    x, y = x.to(args.device), y.to(args.device)  # (B, C, H, W), (B, 10)
    if args.dataset =="MNIST":
        x = x.view(x.size(0), -1)
    logits_z, logits_y, x_hat, mu, logvar = vib(x, mode='with64QAM_reconstruction_CIFAR')  # (B, C* h* w), (B, N, 10)
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

#print('acc_test', acc_test)

# on cifar10, if adda hesian-based model will out memory. maybe the hessian matrix is too big
print('start Hesian-based unl')
vibi_f_hessian = unlearning_hessian(copy.deepcopy(vib).to(args.device), args, dataloader_erase, remaining_set, reconstruction_function, classifier_model,loss_fn)


print('start VBU')
vbu_unl = unlearning_vae(copy.deepcopy(vib).to(args.device), args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, loss_fn, train_type='VBU')


print('start VAE_unl')
vibi_unl = unlearning_vae(copy.deepcopy(vib).to(args.device), args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, loss_fn, train_type='VAE_unl')

print('start VAE_unl_ss')
vibi_unl_ss = unlearning_vae(copy.deepcopy(vib).to(args.device), args, dataloader_erase, dataloader_remain, reconstruction_function, classifier_model, loss_fn, train_type='VAE_unl_ss')



print()
print('generate with 64QAM')
print('original')
acc_back_g = eva_vae_generation(vib, classifier_model, dataloader_erase, args, name='original generated backdoor')
acc_norm_g = eva_vae_generation(vib, classifier_model, dataloader_clean, args, name='original generated clean') #dataloader_clean , test_loader

print('hessian-based')
acc_back_g = eva_vae_generation(vibi_f_hessian, classifier_model, dataloader_erase, args, name='hessian_b_unl generated backdoor')
acc_norm_g = eva_vae_generation(vibi_f_hessian, classifier_model, dataloader_clean, args, name='hessian_b_unl generated clean') #dataloader_clean , test_loader

print('vbu')
acc_back_g = eva_vae_generation(vbu_unl, classifier_model, dataloader_erase, args, name='vbu_unl generated backdoor')
acc_norm_g = eva_vae_generation(vbu_unl, classifier_model, dataloader_clean, args, name='vbu_unl generated clean')

print('vib_unl')
acc_back_g = eva_vae_generation(vibi_unl, classifier_model, dataloader_erase, args, name='vibi_unl generated backdoor')
acc_norm_g = eva_vae_generation(vibi_unl, classifier_model, dataloader_clean, args, name='vibi_unl generated clean')

print('vib_unl_ss')
acc_back_g = eva_vae_generation(vibi_unl_ss, classifier_model, dataloader_erase, args, name='vibi_unl_ss generated backdoor')
acc_norm_g = eva_vae_generation(vibi_unl_ss, classifier_model, dataloader_clean, args, name='vibi_unl_ss generated clean')

plot_latent(vib, dataloader_full, args)
