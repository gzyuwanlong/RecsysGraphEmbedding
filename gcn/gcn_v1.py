"""GCN using DGL nn package
References:
- Author: yuwanlong gzyuwanlong@163.com
- Graph Convolutional Networks version one
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch
import torch.nn as nn
import math


def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], dim=1) * node.data['norm']
    return {'h': accum}


class NodeApplyModule(nn):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size[0])
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, g, in_feats, out_feats, activation, dropout, bias=True):
        super(GCNLayer, self).__init__()



if __name__ == "__main__":
    print("good")