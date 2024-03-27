import torch
import random
import torch.nn as nn
import torch.nn.functional as F



from chebnet import ChebGraphConv
from GAT import GraphAttentionLayer
from GCN import GraphConvolution



class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, alpha, n_heads):
        """Dense version of GAT
        n_heads 表示有几个GAL层，最后进行拼接在一起，类似self-attention
        从不同的子空间进行抽取特征。
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # 定义multi-head的图注意力层
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # 加入pytorch的Module模块
        # 输出层，也通过图注意力层来实现，可实现分类、预测等功能
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)


    # 创建损失函数，使用交叉熵误差
        self.loss_function = nn.CrossEntropyLoss()

        # 创建优化器，使用Adam梯度下降
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.01,weight_decay=5e-4)

        # 训练次数计数器
        self.counter = 0
        # 训练过程中损失值记录
        self.progress = []


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每个head得到的表示进行拼接
        x = F.dropout(x, self.dropout, training=self.training)  # dropout，防止过拟合
        x = F.elu(self.out_att(x, adj))  # 输出并激活
        return F.log_softmax(x, dim=1)  # log_softmax速度变快，保持数值稳定
    


class ChebyNet(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, enable_bias, K_order, K_layer, droprate):
        super(ChebyNet, self).__init__()
        self.cheb_graph_convs = nn.ModuleList()
        self.K_order = K_order
        self.K_layer = K_layer
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_feat, n_hid, enable_bias))
        for k in range(1, K_layer-1):
            self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_hid, enable_bias))
        self.cheb_graph_convs.append(ChebGraphConv(K_order, n_hid, n_class, enable_bias))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, gso):
        for k in range(self.K_layer-1):
            x = self.cheb_graph_convs[k](x, gso)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.cheb_graph_convs[-1](x, gso)
        x = self.log_softmax(x)

        return x
    


class GCN(nn.Module):
    """
    定义一个包含两层GraphConvolution的模型
    """
    def __init__(self, n_feat, n_class):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolution(n_feat, 16)
        self.gcn2 = GraphConvolution(16, n_class)
    
    def forward(self, feature, adjacency):
        h = F.relu(self.gcn1(adjacency, feature))
        h = self.gcn2(adjacency, h)
        return F.log_softmax(h, dim=1)



def create_model(model_type: str, n_feat, n_hid, n_class):
	if model_type == 'GAT':
		return GAT(n_feat, n_hid, n_class, dropout=0.6, alpha=0.2, n_heads=8)
	if model_type == 'GCN':
		return GCN(n_feat, n_class)
	if model_type == 'ChebyNet':
		return ChebyNet(n_feat, n_hid, n_class, enable_bias=True, K_order=2, K_layer=3, droprate=0.5)
	else:
		print('Not implemented!')