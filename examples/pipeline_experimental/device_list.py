from hetu import rgpu, gpu, cpu

device_remote = [
    [rgpu("daim117", 0), rgpu("daim117", 1)],
    [rgpu("daim117", 2), rgpu("daim117", 3)],
    [rgpu("daim119", 0), rgpu("daim119", 1)],
    [rgpu("daim119", 2), rgpu("daim119", 3)],
]

device_local = [
    [gpu(0)],
    [gpu(1)],
    [gpu(2)],
    [gpu(3)],
]

my_device = device_remote

def add_cpu_ctx(device_list):
    result = []
    for i in range(len(device_list)):
        result.append(device_list[i] + [cpu(0)])
    return result
