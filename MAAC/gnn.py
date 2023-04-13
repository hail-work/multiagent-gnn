import networkx as nx
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from math import ceil

class nxGraph:
	def __init__(self,x_size, y_size):
		G = nx.grid_2d_graph(x_size, y_size)
		# Set all weights to 1
		for edge in G.edges:
			G.edges[edge]['weight'] = 1

		pos = {(x,y):(y,-x) for x,y in G.nodes()}
		G.add_edges_from([
			((x, y), (x+1, y+1))
			for x in range(x_size-1)
			for y in range(y_size-1)
		] + [
			((x+1, y), (x, y+1))
			for x in range(x_size-1)
			for y in range(y_size-1)
		])

		self.G = G
		self.pos = pos



	def render_graph(self, node_color='grey', with_labels=False, node_size=10):
		nx.draw(self.G, pos=self.pos,
				node_color=node_color,
				with_labels=with_labels,
				node_size=node_size)

if __name__=='__main__':
	# test if the diff pool is working with visualization using networkx
	# print the start
	print('start')

	from matplotlib import pyplot as plt

	plt.figure(figsize=(6, 6))

	x_size = 20
	y_size = 20

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

	nx.draw_networkx(G, pos=pos,
			node_color='grey',
			with_labels=False,
			node_size=10)
	print('')




if __name__=='__tutorial0_gen_graph__':
	# model = GNN(10, 10)
	# Make the networkx graph
	G = nx.Graph()
	# Add some cars (just do 4 for now)
	G.add_nodes_from([
		(1, {'x': np.random.rand(10)}),
		(2, {'x': np.random.rand(10)}),
		(3, {'x': np.random.rand(10)}),
		(4, {'x': np.random.rand(10)}),
		(5, {'x': np.random.rand(10)}),
	])
	# Add some edges
	G.add_edges_from([
		(1, 2), (1, 4), (1, 5),
		(2, 3), (2, 4),
		(3, 2), (3, 5),
		(4, 1), (4, 2),
		(5, 1), (5, 3)
	])
	nx.draw(G, with_labels=True)
