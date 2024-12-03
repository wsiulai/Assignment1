import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import numpy as np
import random, json
import dgl

from models.gnn import GCN
from models.gt import MLP_dec
from models.loss_fn import TripletLoss

def load_config_from_json_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def all_to_device(lst, device):
    return (x.to(device) for x in lst)

def setup_loss_fn(loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        # elif loss_fn == "sce":
        #     criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "nll":
            criterion = nn.NLLLoss()
        elif loss_fn == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == "cos":
            criterion = nn.CosineSimilarity()
        elif loss_fn == "mae":
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError
        return criterion

class Net_Encoder(nn.Module):
    def __init__(self,   
                 config,
                 device,
                 accelerator=None
                 ):
        super(Net_Encoder, self).__init__()

        self.device = device
        self.config = config
        if accelerator:
            self.accelerator = accelerator

        self.embed_dim = config['embed_dim']

        ### loss function  initialization ###
        margin_num = 1.0
        self.loss_cl = TripletLoss(margin = margin_num)
        self.loss_gmae = setup_loss_fn("mse")
        

        ### GNN initialization ###
        self.gnn_config = load_config_from_json_file(config['gnn_config'])
        self.gnn = GCN(
            config=self.gnn_config
        )
        self.graph_proj = nn.Linear(self.gnn_config['embed_dim'], self.embed_dim)
        self.gmae_decoder = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=512,
            num_layers=3,
            output_dim=self.gnn_config['feat_dim'],
            activation="relu",
            norm="batchnorm"
        )
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.gnn_config['feat_dim']))

        self._mask_rate = self.gnn_config['mask_rate']
        self._drop_edge_rate = self.gnn_config['drop_edge_rate']
        self._replace_rate = self.gnn_config['replace_rate']
        self._mask_token_rate = 1 - self._replace_rate


    def forward(self, data, mode = 'pretrain'):
        self.mode = mode
        if self.mode == 'pretrain':
            return self.pretrain_forward(data)
        elif self.mode == 'infer':
            return self.finetune_forward(data)
        else:
            raise ValueError("Invalid mode")

    def graph_encode(self, graph_data):
        node_rep, graph_rep = self.gnn(graph_data.to(self.device), graph_data.ndata['feat'].to(self.device), return_hidden=False)
        embeds = F.normalize(self.graph_proj(graph_rep),dim=-1)

        return node_rep, embeds

    def finetune_forward(self, data):
        _, graph_embeds = self.graph_encode(data)
        return graph_embeds
    
    def pretrain_forward(self, data):
        graph_ori, graph_pos, graph_neg = data[0], data[1], data[2]

        
        loss_cl = self.pretrain_task_cl(graph_ori, graph_pos, graph_neg)

        loss_gmae = self.pretrain_task_gmae(graph_ori)

        # print(f"loss_cl: {loss_cl.item()} loss_gmae: {loss_gmae.item()}")
        loss = 1.0*loss_cl + 0.03*loss_gmae

        return loss, loss_cl, loss_gmae


    def pretrain_task_cl(self, graph_ori, graph_pos, graph_neg):

        _, graph_embeds_ori = self.graph_encode(graph_ori)
        _, graph_embeds_pos = self.graph_encode(graph_pos)
        _, graph_embeds_neg = self.graph_encode(graph_neg)
        loss = self.loss_cl(graph_embeds_ori, graph_embeds_pos, graph_embeds_neg)

        return loss


    def pretrain_task_gmae(self, g):
        x = g.ndata["feat"].to(self.device)
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._drop_edge_rate > 0:
            use_g, masked_edges = self.drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep, all_hidden, graph_emb = self.gnn(use_g.to(self.device), use_x, return_hidden=True)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.gmae_decoder(rep)

        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.loss_gmae(x_rec, x_init)
        return loss

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = 1 if int(self._replace_rate * num_mask_nodes) < 1 else int(self._replace_rate * num_mask_nodes)
            # num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x = out_x.to(self.device)
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)


    def drop_edge(self, graph, drop_rate, return_edges=False):
        if drop_rate <= 0:
            return graph

        n_node = graph.num_nodes()
        edge_mask = self.mask_edge(graph, drop_rate)
        src = graph.edges()[0]
        dst = graph.edges()[1]

        nsrc = src[edge_mask]
        ndst = dst[edge_mask]

        ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
        ng = ng.add_self_loop()

        dsrc = src[~edge_mask]
        ddst = dst[~edge_mask]

        if return_edges:
            return ng, (dsrc, ddst)
        return ng

    def mask_edge(self, graph, mask_prob):
        E = graph.num_edges()
        mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        return mask_idx
