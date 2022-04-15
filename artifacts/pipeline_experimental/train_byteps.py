import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
import hetu as ht

from byteps.torch.parallel import DistributedDataParallel as DDP
import byteps.torch as bps

from resnet_torch import resnet18, resnet34, resnet50, resnet101, resnet152
from train_torch_utils import Config, train, validate
from datasets.cifar100 import CIFAR100DataLoader

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
    bps.init()

    local_rank = bps.local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d' % local_rank)

    assert args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 'Model not supported now.'

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    net = eval(args.model)(100).to(device)
    net = DDP(net, device_ids=[local_rank])
    net.device = device
    opt = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # data loading
    if dataset == 'CIFAR100':
        data = CIFAR100DataLoader(args.batch_size, image=True, tondarry=False)
        label = CIFAR100DataLoader(args.batch_size, image=False, tondarry=False)
    else:
        raise NotImplementedError

    config = Config(bps.rank(), int(os.environ["DMLC_NUM_WORKER"]))
    data.backward_hook(config)
    label.backward_hook(config)

    if bps.rank() == 0:
        print("Training {} epoch, each epoch runs {} iteration".format(args.epochs, data.get_batch_num("train")))
    # training
    for epoch in range(args.epochs):
        start = time.time()

        loss, acc = [], []
        for i in range(data.get_batch_num("train")):
            loss_, acc_ = train(net, criterion, opt, data, label)
            loss.append(loss_)
            acc.append(acc_)
        loss = torch.tensor(loss).mean().item()
        acc = torch.tensor(acc).mean().item()
        end = time.time()
        print(epoch, "TRAIN loss {:.4f} acc {:.4f} lr {:.2e}, time {:.4f}".format(
                loss, acc, args.learning_rate, end - start))

        loss, acc = [], []
        for i in range(data.get_batch_num("test")):
            loss_, acc_ = validate(net, criterion, data, label)
            loss.append(loss_)
            acc.append(acc_)
        loss = torch.tensor(loss).mean().item()
        acc = torch.tensor(acc).mean().item()
        print(epoch, "EVAL  loss {:.4f} acc {:.4f}".format(loss, acc))
