这里是 GCN 基于 DGL 的三种实现，测试数据是 cora，准确率都在 80% 左右

版本要求：
- PyTorch 0.4.1+
- dgl 0.4.3post2+
- networkx 2.2+

代码：
- gcn_v1：利用 DGL 预定义的图卷积模块 GraphConv 来实现
- gcn_v2：利用 DGL UDFs 自定义的 Message 和 Reduce 函数
- gcn_v3：基于 gcn_v2 的改进，使用 DGL 的内置（builtin）函数

执行：
python gcn_v1.py
