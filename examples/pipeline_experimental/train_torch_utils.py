import torch
from torch._C import device
import torch.distributed as dist

class Config:
    def __init__(self, rank, nrank) -> None:
        self.pipeline = False
        self.context_launch = True
        self.rank = rank
        self.nrank = nrank

def torch_sync_data(device, value):
    # all-reduce train stats
    t = torch.tensor(value, dtype=torch.float64).to(device)
    dist.barrier()  # synchronizes all processes
    dist.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t / dist.get_world_size()

def train(net, criterion, opt, data, label):
    device = net.device
    inputs = data.get_arr("train").to(device)
    targets = torch.argmax(label.get_arr("train"), dim=1).to(device)
    opt.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    opt.step()

    _, predicted = outputs.max(1)
    acc = predicted.eq(targets).sum().item() / targets.shape[0]
    train_loss = loss.item()
    return train_loss, acc

def validate(net, criterion, data, label):
    device = net.device
    with torch.no_grad():
        inputs = data.get_arr("test").to(device)
        targets = torch.argmax(label.get_arr("test"), dim=1).to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
        acc = predicted.eq(targets).sum().item() / targets.shape[0]
        train_loss = loss.item()
    return train_loss, acc
