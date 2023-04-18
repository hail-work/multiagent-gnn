import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.figure(figsize=(6, 6))

x_size = 5
y_size = 5

G = nx.grid_2d_graph(x_size, y_size)

# Set all weights to 1
for edge in G.edges:
	G.edges[edge]['weight'] = 1

pos = {(x, y): (y, -x) for x, y in G.nodes()}

G.add_edges_from([
					 ((x, y), (x + 1, y + 1))
					 for x in range(x_size - 1)
					 for y in range(y_size - 1)
				 ] + [
					 ((x + 1, y), (x, y + 1))
					 for x in range(x_size - 1)
					 for y in range(y_size - 1)
				 ], weight=1.4)

for count, node in enumerate(G.nodes()):
	if count % 6 == 0:
		#         G.nodes[node]['x']={'x':np.ones(5).tolist()}
		G.nodes[node]['x'] = [1.0]
	else:
		#         G.nodes[node]={'x':np.zeros(5).tolist()}
		G.nodes[node]['x'] = [0.0]

# nx.draw(G, with_labels=True)
nx.draw(G, pos=pos,
		node_color='grey',
		with_labels=False,
		node_size=10)


from torch_geometric.utils.convert import from_networkx

pyg_graph = from_networkx(G)

print(pyg_graph)
# Data(edge_index=[2, 12], x=[5], y=[5])
print(pyg_graph.x)
# tensor([0.5000, 0.2000, 0.3000, 0.1000, 0.2000])
# print(pyg_graph.edge_index)
# tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
#         [1, 3, 4, 0, 2, 3, 1, 4, 0, 1, 0, 2]])


mask=[]
for i in range(len(pyg_graph.x)):
    mask.append(False)
mask[0]=True
mask = np.array(mask)

