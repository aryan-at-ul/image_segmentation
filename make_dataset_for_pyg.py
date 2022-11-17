import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_networkx, from_networkx
from torch_geometric.loader import DataLoader
import os
import networkx as nx
import matplotlib.pyplot as plt


path = "/Users/aryansingh/projects/image_segmentation/chest_xray_graphs/train/NORMAL"

all_files = os.listdir(path)
all_data = []
for one_g in all_files:
    print(one_g)
    G = nx.read_gpickle(f"{path}/{one_g}")
    data = from_networkx(G)
    print(len(data.x))
    print(torch.flatten(data.x[0]).shape)
    data.y = [0]
    print(data)
    #data.x = [t.numpy() for t in data.x]
    #print(len([torch.flatten(val) for val in data.x]))
    data.x = torch.Tensor([torch.flatten(val).tolist() for val in data.x])#nx.get_node_attributes(G,'image')
    #ts = torch.stack([val for val in data.values()], dim=1)
    #data.x = ts
    all_data.append(data)
    print(data)
    coragraph = to_networkx(data)
    print("node features",data.num_node_features)
    #plt.figure(1,figsize=(14,12))
    #nx.draw(coragraph, cmap=plt.get_cmap('Set1'),node_size=75,linewidths=6)
    #plt.show()




print(len(all_data))

