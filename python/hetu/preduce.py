from .communicator.mpi_nccl_comm import ncclDataType_t, ncclRedOp_t
from hetu import get_worker_communicate, wrapped_mpi_nccl_init, new_group_comm

import numpy as np
import ctypes

class PartialReduce:
    def __init__(self, reduce_key, max_worker, ssp_bound, sync_every):
        # reduce_key : in pipeline case, worker on each stage use a unique key
        self._reduce_key = reduce_key
        self.ps_comm = get_worker_communicate()
        self.comm = wrapped_mpi_nccl_init()
        self._comm_map = {}
        self.rank = self.comm.rank
        self.nrank = self.comm.nrank
        self._buffer = np.ascontiguousarray(np.repeat(-1, self.nrank + 2).astype(np.int32))
        self._buffer_ptr = self._buffer.ctypes.data_as(ctypes.c_void_p)
        self._wait_time = 1
        self._step = 1
        self._mean_partner = 0
        self._control_flag = False

        self.ps_comm.preduce_init(reduce_key, self.rank, max_worker, ssp_bound, sync_every)

    def get_partner(self, min_worker=2, sync=False):
        # wait_time : the max time to wait, in millisecond
        # max_worker : if max_worker reachs, get_partner will return immediately
        #               in pipeline case, max_worker should be set properly, otherwise -1 is ok
        self._min_worker = min_worker
        timestamp = self.ps_comm.preduce_get_partner(
            self._reduce_key, self.rank, ctypes.c_float(self._wait_time), self._buffer_ptr)
        if not sync:
            return timestamp
        else:
            return self.async_wait(timestamp)

    def async_wait(self, timestamp):
        self.ps_comm.wait_timestamp(timestamp)
        result = None
        self._control_flag = (self._buffer[0] == 1)
        for i in range(len(self._buffer)):
            if self._buffer[i] < 0:
                result = tuple(self._buffer[1 : i])
                break
        assert result is not None
        if len(result) < self._min_worker:
            self._wait_time = min(self._wait_time * 2, 100)
        else:
            self._wait_time *= 0.9
        self._mean_partner = (self._mean_partner * self._step + len(result)) / (self._step + 1)
        self._step += 1
        return result

    def preduce(self, array, partner, stream=None):
        # array : the array to reduce on
        # partner : the partial reduce group returned by get_partner
        # stream : the stream to run allreduce on
        if partner not in self._comm_map.keys():
            self._create_partial_comm(partner)
        comm = self._comm_map[partner]
        comm.dlarrayNcclAllReduce(array, array, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclAvg, stream)

    def _create_partial_comm(self, partner):
        self._comm_map[partner] = new_group_comm(partner)

    @property
    def mean(self):
        return self._mean_partner

    @property
    def control_flag(self):
        return self._control_flag

    def reset_mean(self):
        self._step = 0
