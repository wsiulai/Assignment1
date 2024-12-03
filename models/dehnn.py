# components of virtual nodes layers
class bottom_to_up_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(bottom_to_up_layer, self).__init__()
        self.config = config

        if self.config.down2up_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.down2up_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.down2up_gnn == 'SAGE':
            self.nn = SAGEConv(in_size, out_size)
        elif self.config.down2up_gnn == 'MEAN':
            self.nn = False
            
    def forward(self, embedding, bottom_to_top_paths):
        if type(self.nn) == bool:
            for down2up_array in bottom_to_top_paths:
                update_message = torch.mm(down2up_array, embedding)
                embedding = embedding + update_message
                embedding = torch.mul(embedding, 1.0/(down2up_array.sum(-1)+1).unsqueeze(1))
        else:
            embedding = self.nn(embedding, bottom_to_top_paths)

        return embedding


class top_to_bottom_layer(torch.nn.Module):
    def __init__(self, config, in_size, out_size):
        super(top_to_bottom_layer, self).__init__()
        self.config = config
        if self.config.up2down_gnn == 'GAT':
            self.nn = GATConv(in_size, out_size)
        elif self.config.up2down_gnn == 'GCN':
            self.nn = GCNConv(in_size, out_size)
        elif self.config.up2down_gnn == 'SAGE':
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

        self.down2up_layer = Down2Up_layer(self.config, self.in_size, self.in_size)
        if gnn_type == 'GCN':
            self.samle_level_layer = GCNConv(in_size, out_size)
        elif gnn_type == 'SAGE':
            self.samle_level_layer = SAGEConv(in_size, out_size)
        elif gnn_type == 'GAT':
            self.samle_level_layer = GATConv(in_size, out_size)
        elif gnn_type == 'GIN':
            self.same_level_lnn_layer = torch.nn.Linear(in_size, out_size)
            self.samle_level_layer = GINConv(self.same_level_lnn_layer)
        self.up2down_layer = Up2Down_layer(self.config, self.out_size, self.out_size)
        
    def forward(self, embedding, down2up_path, same_level_edge_index, up2down_edge_index):
        # down2up
        embed = self.down2up_layer(embedding=embedding, down2up_paths=down2up_path)
        # same level
        embed = self.samle_level_layer(embed, same_level_edge_index)
        # up2down
        embed = self.up2down_layer(embed, up2down_edge_index)
        
        return embed
    

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


# MLP layer for aggregation
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, *args):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class MLPModel(nn.Module):
    
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of hidden layers
            dp_rate - Dropout rate to apply throughout the network
        """
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [nn.Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, *args, **kwargs):
        """
        Inputs:
            x - Input features per node
        """
        return self.layers(x)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
    
def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        edge_attr=None,
    ):
        """
        Performs a forward pass of the layer.

        Args:
            x (Tensor or OptPairTensor): The input node features.
            edge_index (Adj): The adjacency matrix of the graph.
            size (Size, optional): The size of the graph. Defaults to None.
            edge_attr (Tensor, optional): The edge attributes. Defaults to None.

        Returns:
            output (Tensor): The output node features.
            edge_attr (Tensor): The updated edge attributes.

        """

        # propagate node information
        output = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # propagate edge information
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        if self.node_self_loop:
            # Add the node's own features
            output = torch.cat((x[1], output), dim=1)

        # Node projection
        output = self.lin_l(output)

        # Edge projection
        if self.edge_out_channels != -1 and edge_attr is not None:
            edge_attr = self.edge_layer(edge_attr)

        if self.normalize:
            output = self.node_normalize(output)
            edge_attr = self.edge_normalize(edge_attr)

        return output, edge_attr
    
#base model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, normalize=True, cached=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=normalize, cached=cached))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=normalize, cached=cached))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=normalize, cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


    
