from hetu import rgpu, gpu

device_remote = [
    [rgpu("daim117", 0), rgpu("daim119", 0)],
    [rgpu("daim117", 1), rgpu("daim119", 1)],
    [rgpu("daim117", 2), rgpu("daim119", 2)],
    [rgpu("daim117", 3), rgpu("daim119", 3)],
]

device_local = [
    [gpu(0)],
    [gpu(1)],
    [gpu(2)],
    [gpu(3)],
]
