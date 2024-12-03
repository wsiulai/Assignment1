import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv, AvgPooling, SAGEConv
from dgl import LapPE
import dgl
import torch as th
from torch.nn.utils.rnn import pad_sequence
import torch as th
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder
from utils_gmae import create_activation, create_norm
from dgl.utils import expand_as_pair

class MyDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name='my_dataset')
        self.graphs = []
        self.label = []
        
    def add_graph_data(self, dgl_graph, label_data):
        self.graphs.append(dgl_graph)
        self.label.append(label_data)
    
    def __getitem__(self, idx):
        return self.graphs[idx]
    
    def __len__(self):
        return len(self.graphs)

    def convert(self):
        self.label = torch.FloatTensor(self.label)

    def to_device(self, device):
        for i in range(len(self.graphs)):
            self.graphs[i] = self.graphs[i].to(device)
        # self.label = self.label.to(device)

    def collate(self, samples):
        
        graphs = list(samples)

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # Graphormer adds a virual node to the graph, which is connected to
        # all other nodes and supposed to represent the graph embedding. So
        # here +1 is for the virtual node.
        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -th.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
        )

        for i in range(num_graphs):
            # A binary mask where invalid positions are indicated by True.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(
                th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # Path padding to make all paths to the same length "max_len".
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graphs[i].edata["feat"] + 1
            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = th.cat(
                (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)
        del graphs, samples, edata

        return (
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            th.stack(path_data),
            dist
        )


class Graphormer(nn.Module):
    def __init__(
        self,
        out_dim=256,
        edge_dim=12,
        max_degree=256,
        num_spatial=64,
        multi_hop_max_dist=5,
        num_encoder_layers=7,
        embedding_dim=27,
        ffn_embedding_dim=256,
        num_attention_heads=3,
        dropout=0.1,
        pre_layernorm=True,
        activation_fn="gelu",
        encoding=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.encoding = encoding
        # self.atom_encoder = nn.Embedding(
        #     num_atoms + 1, embedding_dim, padding_idx=0
        # )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))
        self.graph_token = nn.Embedding(1, embedding_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=embedding_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_attention_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_attention_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=self.embedding_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    activation=create_activation(activation_fn),
                    norm_first=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = create_activation(activation_fn)
        self.embed_out = nn.Linear(self.embedding_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(th.zeros(out_dim))

        ### add a fc layer to map the graph representation to the output (from embedding_dim to 256)
        self.fc = nn.Linear(self.embedding_dim, out_dim, bias=False)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
        self,
        batched_data,
        mask=None,
    ):  
        (attn_mask,node_feat,in_degree,out_degree,path_data,dist) = batched_data
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(th.stack((in_degree, out_degree)))

        if self.encoding:
            node_feat_comb = node_feat + deg_emb
            graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
                num_graphs, 1, 1
            )
            x = th.cat([graph_token_feat, node_feat_comb], dim=1)

        attn_bias = th.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )

        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        if (mask is not None) and self.encoding:
            x = x[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            attn_bias = attn_bias[:, ~mask, :, :][:, :, ~mask, :] # [n graph, n non-masked nodes, n non-masked nodes, n heads,]
            attn_mask = attn_mask[:, ~mask, :][:, :, ~mask]

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 0, :]
        node_rep = x[:, 1:, :]
        
        node_rep = self.embed_out(node_rep)
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return node_rep, graph_rep




class GSAGE(nn.Module):
    def __init__(self,
                 in_dim=27,
                 num_hidden=512,
                 out_dim=256,
                 num_layers=3,
                 dropout=0,
                 activation="relu",
                 residual=False,
                 norm=None,
                 encoding=False
                 ):
        super(GSAGE, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(SAGEConv(
                in_dim, out_dim, aggregator_type = 'mean', activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(SAGEConv(
                in_dim, num_hidden, aggregator_type = 'mean',  activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(SAGEConv(
                    num_hidden, num_hidden, aggregator_type = 'mean',  activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(SAGEConv(
                num_hidden, out_dim, aggregator_type = 'mean',  activation=last_activation))

        self.head = nn.Identity()

    def forward(self, g, feature, return_hidden=False):
        h = feature
        # h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gcn_layers[l](g, h)
            hidden_list.append(h)


        ### get graph-level embedding ###
        avgpool = AvgPooling()
        hg = avgpool(g, h)

        if return_hidden:
            return h, hidden_list, hg
        else:
            return h, hg

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        
        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        # if norm == "batchnorm":
        #     self.norm = nn.BatchNorm1d(out_dim)
        # elif norm == "layernorm":
        #     self.norm = nn.LayerNorm(out_dim)
        # else:
        #     self.norm = None
        
        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')
            
            feat_src, feat_dst = expand_as_pair(feat, graph)
            
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            
            rst = self.fc(rst)

            # if self._norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class GIN(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 encoding=False,
                 learn_eps=False,
                 aggr="sum",
                 ):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:
            # input projection (no residual)
            self.layers.append(GINConv(
                in_dim, 
                num_hidden, 
                ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                init_eps=0,
                learn_eps=learn_eps,
                residual=residual)
                )
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(GINConv(
                    num_hidden, num_hidden, 
                    ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm), 
                    init_eps=0,
                    learn_eps=learn_eps,
                    residual=residual)
                )
            # output projection
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)

            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            hidden_list.append(h)
        # output projection
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 apply_func,
                 aggregator_type="sum",
                 init_eps=0,
                 learn_eps=False,
                 residual=False,
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func

        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
            
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)
        
