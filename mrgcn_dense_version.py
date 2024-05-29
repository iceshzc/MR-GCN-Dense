#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb
from torch_geometric.utils import dense_to_sparse, f1_score, accuracy
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import DenseGCNConv
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter_add
import torch_sparse
from scipy.sparse import csr_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import pickle
import scipy.sparse as sp
import argparse

# import torchsnooper


# # Model

# In[ ]:


# @torchsnooper.snoop()
class MRGCO(nn.Module):
    def __init__(self, num_nodes, num_rel, hidden):
        super(MRGCO, self).__init__()
        self.num_nodes = num_nodes
        self.num_rel = num_rel
        self.hidden = hidden
        self.weight = nn.Parameter(torch.Tensor(hidden, hidden, num_rel)).cuda()
        self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def tensor_product(self, A, B):
        """
        Tensor Product with DCT, 同样的这里也需要sparse
        """
        n1 = A.shape[0]
        n3 = A.shape[2]
        n4 = B.shape[1]

        A_transformed = torch.fft.fft(A)
        B_transformed = torch.fft.fft(B)

        C_transformed = torch.zeros((n1,n4,n3)).cuda()

        for k in range(n3):
            a_slice = A_transformed[:,:,k]
            b_slice = B_transformed[:,:,k]
            C_transformed[:,:,k] = torch.matmul(torch.real(a_slice),torch.real(b_slice))
        C = torch.fft.ifft(C_transformed)
        return torch.real(C)

    def forward(self, X, A):
        res = self.tensor_product(A, X)
        res = self.tensor_product(res, self.weight)
        return res


# In[ ]:


# @torchsnooper.snoop()
class MRGCN(nn.Module):
    def __init__(self, num_rel, in_channels, out_channels, num_class, num_nodes, num_layers, dropout):
        super(MRGCN, self).__init__()
        self.num_rel = num_rel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        # GCN 
        layers = []
        for i in range(num_rel):
            layers.append(nn.ModuleList([DenseGCNConv(in_channels, out_channels),                 DenseGCNConv(out_channels, out_channels)]))
        self.gcn_layers = nn.ModuleList(layers)
        # MR-GCN
        layers = []
        for i in range(num_layers):
            layers.append(MRGCO(num_nodes, num_rel, out_channels))
        self.mrgco_layers = nn.ModuleList(layers)
        # MLP and Softmax
        self.linear1 = nn.Linear(self.out_channels*self.num_rel, self.out_channels)
        self.linear2 = nn.Linear(self.out_channels, self.num_class)
        # Loss
        self.loss = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        for layers in self.gcn_layers:
            for layer in layers:
                layer.reset_parameters()
        for layer in self.mrgco_layers:
            layer.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, A, X, target_x, target):
        for i, adj in enumerate(A):
            # tensor X
            gcns = self.gcn_layers[i]
            tmp_x = F.relu(gcns[0](X, adj))
            tmp_x = F.dropout(tmp_x, p=self.dropout, training=self.training)
            tmp_x = F.relu(gcns[1](tmp_x, adj))
            #tmp_x = F.dropout(tmp_x, p=self.dropout, training=self.training)
            tmp_x = tmp_x.permute(1,2,0)

            # normlized edge index and corresponding values
            tmp_edge_index = gcn_norm(SparseTensor.from_dense(adj))
            normal_adj = tmp_edge_index.to_dense()
            normal_adj= normal_adj.unsqueeze(2)
            
            ####图的结构需要sparse
            # constructing the tensors
            if i == 0:
                tensor_X, tensor_A = tmp_x, normal_adj
            else:
                tensor_X = torch.cat((tensor_X, tmp_x), dim=2)
                tensor_A = torch.cat((tensor_A, normal_adj), dim=2)
        # mr-gco with tensor product
        for i in range(self.num_layers):
            tensor_X = F.relu(self.mrgco_layers[i](tensor_X, tensor_A))
            if i<self.num_layers - 1:
                tensor_X = F.dropout(tensor_X, p=self.dropout, training=self.training)

        X_ = torch.flatten(tensor_X, 1, 2)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y


# # Hyper-parameters

# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=12306,
                    help='Seed for random splits.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--early_stopping', type=int, default=200,
                    help='Number of epochs of early stopping settings.')
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of hidden layers.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss parameter).')
parser.add_argument('--num_layers', type=int, default=2,
                    help='Number of hidden layers.')
parser.add_argument('--act', type=str, default='relu',
                    help='Activation function.')
#parser.add_argument('--unitary', type=str, default='dft',
#                    choices=['dft', 'dct','haar'])
parser.add_argument('--dataset', type=str, default='ACM',
                    choices=['ACM', 'IMDB','Amazon','Reuters'])
parser.add_argument('--device', type=str, default='0', 
                    help='Visible device of cuda.')
parser.add_argument('--adaptive_lr', type=str, default='false',
                help='adaptive learning rate')

# for jupyter notebook
args = parser.parse_args(args=[])

# for python console
# args = parser.parse_args()


# # Train-Valid-Test 

# ## setting

# In[ ]:


# parameter settings
seed = args.seed
epochs = args.epochs
node_dim = args.hidden
lr = args.lr
early_stopping = args.early_stopping
dropout = args.dropout
weight_decay = args.weight_decay
# l1_norm = args.l1_norm
num_layers = args.num_layers
act_fcn = args.act
adaptive_lr = args.adaptive_lr


# ## data

# In[ ]:


# data loader
with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
    node_features = pickle.load(f)
with open('data/'+args.dataset+'/edges.pkl','rb') as f:
    edges = pickle.load(f)
with open('data/'+args.dataset+'/labels.pkl','rb') as f:
    labels = pickle.load(f)

# constructing graphs
num_nodes = edges[0].shape[0]
A = []

for _,edge in enumerate(edges):
    adj = torch.tensor(edge.toarray()).type(torch.cuda.FloatTensor)
    #value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.cuda.FloatTensor)
    #A.append((edge_tmp,value_tmp))
    A.append(adj)
#edge_tmp = torch.stack((torch.arange(0,num_nodes),torch.arange(0,num_nodes))).type(torch.cuda.LongTensor)
#value_tmp = torch.ones(num_nodes).type(torch.cuda.FloatTensor)
#A.append((edge_tmp,value_tmp))

node_features = torch.from_numpy(node_features.toarray()).type(torch.cuda.FloatTensor)
train_node = torch.from_numpy(np.array(labels[0])[:,0]).type(torch.cuda.LongTensor)
train_target = torch.from_numpy(np.array(labels[0])[:,1]).type(torch.cuda.LongTensor)

valid_node = torch.from_numpy(np.array(labels[1])[:,0]).type(torch.cuda.LongTensor)
valid_target = torch.from_numpy(np.array(labels[1])[:,1]).type(torch.cuda.LongTensor)
test_node = torch.from_numpy(np.array(labels[2])[:,0]).type(torch.cuda.LongTensor)
test_target = torch.from_numpy(np.array(labels[2])[:,1]).type(torch.cuda.LongTensor)

num_classes = torch.max(train_target).item()+1


# ## model train

# In[ ]:


train_losses = []
train_f1s = []
val_losses = []
test_losses = []
val_f1s = []
test_f1s = []
final_f1 = 0
for cnt in range(5):
    best_val_loss = 10000
    best_test_loss = 10000
    best_train_loss = 10000
    best_train_f1 = 0
    best_val_f1 = 0
    best_test_f1 = 0
    model = MRGCN(num_rel = len(A),
                    in_channels = node_features.shape[1],
                    out_channels = node_dim,
                    num_class = num_classes,
                    num_nodes = node_features.shape[0],
                    num_layers = num_layers,
                    dropout = dropout)

    model.cuda()
    if adaptive_lr == 'false':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params':model.gcn_layers.parameters()},
                                    {'params':model.linear1.parameters()},
                                    {'params':model.linear2.parameters()},
                                    {"params":model.mrgco_layers.parameters(), "lr":0.5}
                                    ], lr=0.005, weight_decay=0.001)
    loss = nn.CrossEntropyLoss()
    Ws = []
    for i in range(500):
        print('Epoch: ',i+1)
        for param_group in optimizer.param_groups:
            if param_group['lr'] > 0.005:
                param_group['lr'] = param_group['lr'] * 0.9
        model.train()
        model.zero_grad()
        loss, y_train = model(A, node_features, train_node, train_target)
        loss.backward()
        optimizer.step()
        train_f1 = torch.mean(f1_score(torch.argmax(y_train,dim=1), train_target, num_classes=3)).cpu().numpy()
        print('Train - Loss: {}, Macro_F1: {}'.format(loss.detach().cpu().numpy(), train_f1))
        model.eval()
        # Valid
        with torch.no_grad():
            val_loss, y_valid = model.forward(A, node_features, valid_node, valid_target)
            val_f1 = torch.mean(f1_score(torch.argmax(y_valid,dim=1), valid_target, num_classes=3)).cpu().numpy()
            print('Valid - Loss: {}, Macro_F1: {}'.format(val_loss.detach().cpu().numpy(), val_f1))
            test_loss, y_test = model.forward(A, node_features, test_node, test_target)
            test_f1 = torch.mean(f1_score(torch.argmax(y_test,dim=1), test_target, num_classes=3)).cpu().numpy()
            test_acc = accuracy(torch.argmax(y_test,dim=1), test_target)
            print('Test - Loss: {}, Macro_F1: {}, Acc: {}\n'.format(test_loss.detach().cpu().numpy(), test_f1, test_acc))
            if val_f1 > best_val_f1:
                best_val_loss = val_loss.detach().cpu().numpy()
                best_test_loss = test_loss.detach().cpu().numpy()
                best_train_loss = loss.detach().cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1
                best_test_f1 = test_f1
        torch.cuda.empty_cache()
    print('---------------Best Results--------------------')
    print('Train - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_train_f1))
    print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
    print('Test - Loss: {}, Macro_F1: {}'.format(best_test_loss, best_test_f1))


# In[ ]:




