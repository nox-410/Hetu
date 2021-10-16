import os
import time

# tensorboard --logdir='runs' --port=6006 --host='0.0.0.0'

def get_partition(device_list, partition):
    n = len(partition)
    result = []
    for i in range(n):
        for j in range(partition[i]):
            result.append(device_list[i])
    return result

def get_tensorboard_writer(name=""):
    from torch.utils.tensorboard import SummaryWriter
    if not name:
        name = time.ctime().replace(' ', '_')
    writer = SummaryWriter(log_dir=os.path.join("runs", name))
    return writer
