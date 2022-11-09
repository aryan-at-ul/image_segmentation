from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
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

from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
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
        G = nx.read_gpickle(f"{path}/{one_g}")
        data = from_networkx(G)

        if type:
            data.y = [1]
        else:
            data.y = [0]
        k+= 1
        
        data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
        # data.x = data.x.type(torch.LongTensor)
        print(k,data)
        all_data.append(data)
    return all_data




def dataloader():
    """
    load train and test data
    """
    print("loading data")
    train_normal = load_all_from_one_folder(f"{path}/train/NORMAL")
    train_pneumonia = load_all_from_one_folder(f"{path}/train/PNEUMONIA",1)

    all_data = train_normal + train_pneumonia

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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset



train_loader, test_loader, train_dataset, test_dataset, val_loader, val_dataset = dataloader()

# dataset = train_dataset # Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
# data = dataset[0]


# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='Cora')
# parser.add_argument('--hidden_channels', type=int, default=16)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--epochs', type=int, default=200)
# parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
# parser.add_argument('--wandb', action='store_true', help='Track experiment')
# args = parser.parse_args()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# init_wandb(name=f'GCN-img', lr=0.01, epochs=2,
#            hidden_channels=16, device=device)

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')


# if True:
#     transform = T.GDC(
#         self_loop_weight=1,
#         normalization_in='sym',
#         normalization_out='col',
#         diffusion_kwargs=dict(method='ppr', alpha=0.05),
#         sparsification_kwargs=dict(method='topk', k=128, dim=0),
#         exact=True,
#     )
#     data = transform(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=not True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=not True)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GCN(512, 16, 2)
model = model.to(device)#, data.to(device)


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, data.edge_weight)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return float(loss)


# @torch.no_grad()
# def test():
#     model.eval()
#     pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
#     return accs


# best_val_acc = final_test_acc = 0
# for epoch in range(1, 2):
#     loss = train()
#     train_acc, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

model = Sequential('x, edge_index, batch', [
    (Dropout(p=0.5), 'x -> x'),
    (GCNConv(512, 64), 'x, edge_index -> x1'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x1, edge_index -> x2'),
    ReLU(inplace=True),
    (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
    (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
    (global_mean_pool, 'x, batch -> x'),
    Linear(2 * 64, 2),
])

# model = GCN(512, 16, 2)
model = model.to(device)#, data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Only perform weight-decay on first convolution.


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        print(data)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = test(test_loader)
    print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
    optimizer.step()
