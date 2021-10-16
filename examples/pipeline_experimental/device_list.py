from hetu import rgpu, gpu

device1 = [
    [rgpu("daim116", 0), rgpu("daim119", 0)],
    [rgpu("daim116", 1), rgpu("daim119", 1)],
    [rgpu("daim116", 2), rgpu("daim119", 2)],
    [rgpu("daim116", 3), rgpu("daim119", 3)],
]

device_local = [
    [gpu(0)],
    [gpu(1)],
    [gpu(2)],
    [gpu(3)],
]
