import torch 
import time
import math
import dgl
import numpy as np 
import torch.nn as nn 
from dgl.data import citation_graph as citegrh 
from dgl import DGLGraph
import dgl.function as fn 
import networkx as nx 
import torch.nn.functional as F 

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
    

if __name__ == "__main__":
    dropout=0.5
    gpu=-1
    lr=0.001
    n_epochs=200
    n_hidden=16
    n_layers=2
    weight_decay=5e-4
    self_loop=True

    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()

    g = data.graph

    if self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(g)


    if gpu < 0:
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

    # 创建一个 GCN 的模型，可以选择上面的任意一个进行初始化
    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)


    if cuda:
        model.cuda()

    # 采用交叉熵损失函数和 Adam 优化器
    loss_fcn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

    # 定义一个评估函数
    def evaluate(model, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    # 训练，并评估
    dur = []
    for epoch in range(n_epochs):
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
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

