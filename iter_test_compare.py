from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
# from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
import random
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


curretn_path = os.getcwd()
path = f"{curretn_path}/chest_xray_graphs"

embed_dim = 128

def load_all_from_one_folder(path,type = 0):
    all_files = os.listdir(path)
    all_data = []
    k = 0
    for one_g in all_files[0:100]:
        print(one_g)
        name = one_g.split(".")[0]
        G = nx.read_gpickle(f"{path}/{one_g}")
        # print(G.nodes[0]['x'].shape)
        data = from_networkx(G)
        print(data)

        if type:
            data.y = [1]
        else:
            data.y = [0]
        k+= 1
        # print(data.x.shape)
        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
        data.name = name
        # data.x = data.x.type(torch.LongTensor)
        print(k,data)
        all_data.append(data)
    return all_data


def permute_array(array):
    permuted_array = []
    for i in range(len(array)):
        permuted_array.append(array[i])
    return permuted_array



def dataloader():
    """
    load train and test data
    """
    print("loading data")
    train_normal = load_all_from_one_folder(f"{path}/train/NORMAL")
    train_pneumonia = load_all_from_one_folder(f"{path}/train/PNEUMONIA",1)

    all_data = train_normal + train_pneumonia
    # all_data = permute_array(all_data)
    random.shuffle(all_data)

    if True:
        transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
        )
        data = transform(all_data[0])


    train_dataset = all_data[:int(len(all_data)*0.8)]
    val_dataset = all_data[int(len(all_data)*0.8):int(len(all_data)*0.8) + 100]
    test_dataset = all_data[int(len(all_data)*0.8):]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False,drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True,drop_last=True)

    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset



train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data[0].edge_attr)
    print()

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = GCNConv(512, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.lin = Linear(hidden_channels, 2)
        self.conv1 = GCNConv(512, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 64)
        # self.lin1 = Linear(32, 16)
        self.lin = Linear(64, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin1(x)
        # x = x.relu()
        x = self.lin(x)

        return x

model = GCN(hidden_channels=64)
print(model)


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        print('^^^^^'*10)

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        data.y = torch.Tensor(data.y)
        data.y = torch.Tensor(torch.flatten(data.y))
        data.y = data.y.type(torch.LongTensor)
        # print(data.y,"kjhkjdhsfkjhsdkjfhksjdhfkjsdhkjfhsdkjhfjs")
        # print(out,"dsjflkdsjlfkjsdlkfjlkdsjflksdjlfkjsdlkjlkj")
        loss = criterion(out, data.y)
        print(loss.item())
        # loss = F.nll_loss(out, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
compare = []
def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         data.y = torch.Tensor(data.y)
         print("==="*10)
         print(data)
         pred = out.argmax(dim=1).view(-1,1)  # Use the class with highest probability.
         print(pred,"pred here",data.y,"file name is",data.name)
         compare.append([pred,data.name,data.y])
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 10):
    train()
    try:
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    except:
        pass

print("number of paramteres for this model",sum(p.numel() for p in model.parameters()))
# print(compare)
df = pd.DataFrame(compare, columns=["pred","filename","ingraph-label"])
df.to_csv('cheching_preds.csv',index=False)




