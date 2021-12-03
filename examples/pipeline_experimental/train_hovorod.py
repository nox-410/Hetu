import torch
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
import numpy as np
import argparse
import time
import os

from resnet_torch import resnet18, resnet34, resnet50, resnet101, resnet152
from train_torch_utils import Config, train, validate
from cifar100 import CIFAR100DataLoader

def torch_sync_data(device, value):
    # all-reduce train stats
    t = torch.tensor(value, dtype=torch.float32).to(device)
    hvd.allreduce(t)
    return t

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="resnet18", help='model to be tested')
    parser.add_argument('--dataset', type=str, default="CIFAR100", help='dataset to be trained on')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100, help='epoch number')
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device('cuda:%d' % hvd.local_rank())

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    net = eval(args.model)(100).to(device)
    opt = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    opt = hvd.DistributedOptimizer(opt, named_parameters=net.named_parameters())
    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(opt, root_rank=0)
    net.device = device

    criterion = nn.CrossEntropyLoss()

    # data loading
    if dataset == 'CIFAR100':
        data = CIFAR100DataLoader(args.batch_size, image=True, tondarry=False)
        label = CIFAR100DataLoader(args.batch_size, image=False, tondarry=False)
    else:
        raise NotImplementedError

    config = Config(hvd.rank(), hvd.size())
    data.backward_hook(config)
    label.backward_hook(config)

    if hvd.rank() == 0:
        print("Training {} epoch, each epoch runs {} iteration".format(args.epochs, data.get_batch_num("train")))
    # training
    for epoch in range(args.epochs):
        start = time.time()

        loss, acc = [], []
        for i in range(data.get_batch_num("train")):
            loss_, acc_ = train(net, criterion, opt, data, label)
            loss.append(loss_)
            acc.append(acc_)
        loss = torch_sync_data(device, loss)
        acc = torch_sync_data(device, acc)
        loss = loss.mean().item()
        acc =acc.mean().item()
        end = time.time()
        if hvd.rank() == 0:
            print(epoch, "TRAIN loss {:.4f} acc {:.4f} lr {:.2e}, time {:.4f}".format(
                        loss, acc, args.learning_rate, end - start))

        loss, acc = [], []
        for i in range(data.get_batch_num("test")):
            loss_, acc_ = validate(net, criterion, data, label)
            loss.append(loss_)
            acc.append(acc_)
        loss = torch_sync_data(device, loss)
        acc = torch_sync_data(device, acc)
        loss = loss.mean().item()
        acc =acc.mean().item()
        if hvd.rank() == 0:
            print(epoch, "EVAL  loss {:.4f} acc {:.4f}".format(loss, acc))
