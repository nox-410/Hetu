from hetu import get_worker_communicate
from hetu.preduce import PartialReduce
from hetu import ndarray
import hetu as ht

import ctypes
import argparse
import numpy as np
from tqdm import tqdm
import time
import random


def test(args):
    comm = get_worker_communicate()
    rank = comm.rank()
    comm.ssp_init(rank // 2, 2, 0)
    for i in range(10):
        print("rank =", rank, " stage =", i)
        comm.ssp_sync(rank // 2, i)
        time.sleep(rank * 0.1)

def test2():
    comm = ht.wrapped_mpi_nccl_init()
    p = PartialReduce(reduce_key=0, max_worker=comm.nrank, ssp_bound=5, sync_every=50)
    val = np.repeat(comm.rank, 1024)
    val = ndarray.array(val, ctx=ndarray.gpu(comm.rank))
    while not p.control_flag:
        time.sleep(random.random() / 10)
        partner = p.get_partner(sync=True)
        p.preduce(val, partner)
        print(partner, val.asnumpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    ht.worker_init()
    test2()
    ht.worker_finish()
