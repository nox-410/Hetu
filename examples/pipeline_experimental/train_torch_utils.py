import torch
from torch._C import device
import torch.distributed as dist

class Config:
    def __init__(self, rank, nrank) -> None:
        self.pipeline = False
        self.context_launch = True
        self.rank = rank
        self.nrank = nrank

def train(net, criterion, opt, data, label):
    net.train()
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
    net.eval()
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
