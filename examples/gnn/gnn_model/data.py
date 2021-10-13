from hetu.gpu_ops.Node import Op
from hetu import ndarray

class GNNDataLoaderOp(Op):
    graph = None
    nxt_graph = None

    def __init__(self, handler, ctx=ndarray.cpu(0)):
        super().__init__(GNNDataLoaderOp, [], ctx)
        self.on_gpu = True
        self.on_cpu = False
        self.handler = handler
        self.name = "GNNDataLoaderOp"
        self.desc = self.name

    def get_batch_num(self, name):
        return None

    def get_arr(self, name):
        return self.handler(self.graph)

    def get_next_arr(self, name):
        return self.handler(self.nxt_graph)

    def get_cur_shape(self, name):
        return self.handler(self.graph).shape

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        raise NotImplementedError

    @classmethod
    def step(cls, graph):
        cls.graph = cls.nxt_graph
        cls.nxt_graph = graph
