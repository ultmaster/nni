import torch
import torch.nn as nn
from nni._graph_utils import build_graph, build_module_graph

from torch.utils.tensorboard._pytorch_graph import graph


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv2d(2, 2, 3)

    def forward(self, x):
        # t = [x, x + 1, x + 2]
        t = torch.split(x, 2)
        # import pdb; pdb.set_trace()
        a, b = torch.split(x, 2)
        # a, b, c = t
        return torch.cat((a, b))
        return self.conv(x)
        # two = torch.split(x, 2)
        # return torch.cat(two, 0)


net = Net()

# t = torch.jit.trace(net, torch.randn(4, 2, 3, 3))
# torch._C._jit_pass_inline(t.graph)
# print(t.graph)
t = build_module_graph(net, torch.randn(4, 2, 3, 3))
for node in t.nodes_py.nodes_op:
    print(node.name)
for node in t.input_to_node.items():
    print(node)
# for node in t.nodes_py.nodes_op:
    # print(node.name)
# graph(net, (torch.randn(4, 6), ))
