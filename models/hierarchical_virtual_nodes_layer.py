from itertools import combinations
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv, GNNModel
from models.mlp import MLP
from torch_geometric.nn import MessagePassing

# components of virtual nodes layers
import torch
class bottom_to_up_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(bottom_to_up_layer, self).__init__()
        self.config = config

        if self.config.bottom_to_up_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'GNNModel':
            self.nn = GNNModel(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'SAGE':
            self.nn = SAGEConv(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'MEAN':
            self.nn = False
            
    def forward(self, embedding, bottom_to_top_paths):
        if type(self.nn) == bool:
            for bottom_to_up_array in bottom_to_top_paths:
                update_message = torch.mm(bottom_to_up_array, embedding)
                embedding = embedding + update_message
                embedding = torch.mul(embedding, 1.0 / (bottom_to_up_array.sum(-1) + 1).unsqueeze(1))
        else:
            embedding = self.nn(embedding, bottom_to_top_paths)

        return embedding


class top_to_bottom_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(top_to_bottom_layer, self).__init__()
        self.config = config
        if self.config.top_to_bottom_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'GNNModel':
            self.nn = GNNModel(in_size, out_size)
        elif self.config.top_to_bottom_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.top_to_bottom_gnn == 'SAGE':
            self.nn = SAGEConv(in_size, out_size)
        
    def forward(self, embedding, top_to_bottom_edge_index):
        embedding = self.nn(embedding, top_to_bottom_edge_index)
        
        return embedding
    

class DEHNN_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size, gnn_type):
        super(DEHNN_layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.config = config

        self.bottom_to_up_layer = bottom_to_up_layer(self.config, self.in_size, self.in_size)
        if gnn_type == 'GCN':
            self.samle_level_layer = GCNConv(in_size, out_size)
        elif self.config.bottom_to_up_gnn == 'GNNModel':
            self.nn = GNNModel(in_size, out_size)
        elif gnn_type == 'SAGE':
            self.samle_level_layer = SAGEConv(in_size, out_size)
        elif gnn_type == 'GAT':
            self.samle_level_layer = GATConv(in_size, out_size)
        elif gnn_type == 'GIN':
            self.same_level_lnn_layer = torch.nn.Linear(in_size, out_size)
            self.samle_level_layer = GINConv(self.same_level_lnn_layer)
        self.top_to_bottom_layer = top_to_bottom_layer(self.config, self.out_size, self.out_size)
        
    def forward(self, embedding, down2up_path, same_level_edge_index, up2down_edge_index):
        # going up
        embed = self.bottom_to_up_layer(embedding=embedding, down2up_paths=down2up_path)
        # same level
        embed = self.samle_level_layer(embed, same_level_edge_index)
        # going down
        embed = self.top_to_bottom_layer(embed, up2down_edge_index)
        
        return embed