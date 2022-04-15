import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import argparse
import time
import os

# only used for launcher
import horovod.torch as hvd

from resnet_torch import resnet18, resnet34, resnet50, resnet101, resnet152
from train_torch_utils import Config, train, validate
from datasets.cifar100 import CIFAR100DataLoader

def torch_sync_data(device, value):
    # all-reduce train stats
    t = torch.tensor(value, dtype=torch.float64).to(device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()

if __name__ == "__main__":
    # argument parser
    global local_rank
    local_rank = 0
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
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    rank = hvd.rank()
    world_size = hvd.size()
    dist.init_process_group(backend="nccl",
                            world_size=world_size,
                            rank=rank,
                            init_method=init_method)

    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d' % local_rank)

    assert args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], 'Model not supported now.'

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    net = eval(args.model)(100).to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    opt = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    # data loading
    if dataset == 'CIFAR100':
        data = CIFAR100DataLoader(args.batch_size, image=True, tondarry=False)
        label = CIFAR100DataLoader(args.batch_size, image=False, tondarry=False)
    else:
        raise NotImplementedError

    config = Config(dist.get_rank(), dist.get_world_size())
    data.backward_hook(config)
    label.backward_hook(config)

    if dist.get_rank() == 0:
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
        if dist.get_rank() == 0:
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
        if dist.get_rank() == 0:
            print(epoch, "EVAL  loss {:.4f} acc {:.4f}".format(loss, acc))
