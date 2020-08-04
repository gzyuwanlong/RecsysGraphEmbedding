"""GCN using DGL nn package
References:
- Author: yuwanlong gzyuwanlong@163.com
- Graph Convolutional Networks version two
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch
import time
import argparse
import numpy as np
import torch.nn as nn
from dgl import DGLGraph
import networkx as nx
import torch.nn.functional as F
import utils
from dgl.data import register_data_args

from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, g, in_feats, h_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, h_hidden, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(h_hidden, h_hidden, activation=activation))
        self.layers.append(GraphConv(h_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.g, h)
        return h


def main(args):
    data, features, labels = utils.load_cora_data()
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    g = data.graph

    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # 归一化，依据入度进行计算
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)


    if cuda:
        model.cuda()

    loss_fcn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)

    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        if epoch % 10 == 0:
            acc = utils.evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(), acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = utils.evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    register_data_args(parser)
    parser.add_argument("--self_loop", type=bool, default=True, help="choose dataset")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gup")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--n_hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    args = parser.parse_args()
    print(args)

    main(args)