import torch
import numpy as np
import pickle, json, time, re, sys
from DG import Graph, Node
import networkx as nx
from multiprocessing import Pool
import dgl
from dgl import from_networkx
import dgl
import torch as th
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from graph2dgl import node_feat_extra_word, edge_feat_extra_word
import os

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

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = th.stack(labels)

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

        return (
            [labels.reshape(num_graphs, -1),
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            th.stack(path_data),
            dist]
        )

    

def run_one_ep(design_name, ep):
    
    folder_dir = f'/home/coguest5/hdl_fusion/data_collect/dataset/{cmd}/graph'
    with open(f'{folder_dir}/{design_name}/{ep}_{cmd}.pkl', 'rb') as f:
            graph = pickle.load(f)
    with open(f'{folder_dir}/{design_name}/{ep}_{cmd}_node_dict.pkl', 'rb') as f:
            node_dict = pickle.load(f)

    g_nx = nx.DiGraph(graph)

    feat_matrix = []
    # feat_dict = {}
    for node_name in g_nx.nodes():
        if node_name not in node_dict:
            feat_vec = np.array([0 for i in range(27)])
            # print(len(feat_vec))
            # input()
        else:
            node = node_dict[node_name]
            feat_vec = node_feat_extra_word(node_name, node, g_nx, node_dict)
        # feat_dict[node_name] = torch.FloatTensor(feat_vec)
        feat_matrix.append(feat_vec)
    feat_matrix = np.array(feat_matrix)
    feat_matrix = torch.FloatTensor(feat_matrix)

    feat_matrix_edge = []
    for edge_pair in g_nx.edges():
        edge_vec = edge_feat_extra_word(edge_pair, g_nx, node_dict)
        feat_matrix_edge.append(edge_vec)
    feat_matrix_edge = np.array(feat_matrix_edge)
    feat_matrix_edge = torch.FloatTensor(feat_matrix_edge)

    dgl_graph = from_networkx(g_nx)
    dgl_graph.ndata['feat'] = feat_matrix
    dgl_graph.edata['feat'] = feat_matrix_edge
    spd, path = dgl.shortest_dist(dgl_graph, root=None, return_paths=True)
    dgl_graph.ndata['spd'] = spd
    dgl_graph.ndata['path'] = path
    
    dataset.add_graph_data(dgl_graph, [])

    data_dict[len(data_dict)] = [design_name, ep]


def save_dataset(design_lst, tag):
    global dataset, data_dict
    data_dict = {}
    # cmd = "ori"
    # cmd = "pos"
    # cmd = "neg"
    # for cmd in ["ori", "pos", "neg"]:
    if cmd in ['ori', "pos"]:
        print(f'Tag: {tag} CMD: {cmd}')
        dataset = MyDataset()
        for design in design_lst:
            with open (f"/home/coguest5/hdl_fusion/data_collect/label/ep_lst/{design}.json", 'r') as f:
                reg_lst = json.load(f)

            for ep in reg_lst:
                # print(design + ' ' + ep)
                if not os.path.exists(f"/home/coguest5/hdl_fusion/text_enc/llm_extra/rtl_func_ori/{design}/{ep}.txt"):
                    print("Not exist: ", design, ep)
                    continue
                if not os.path.exists(f"/home/coguest5/hdl_fusion/text_enc/llm_extra/rtl_func_pos/{design}/{ep}.txt"):
                    print("Not exist: ", design, ep)
                    continue
                run_one_ep(design, ep)
    
    elif cmd == 'neg':
        print(f'Tag: {tag} CMD: {cmd}')
        with open (f"/home/coguest5/hdl_fusion/text_encbert_cl/neg_map/reg_map.json", 'r') as f:
            reg_map = json.load(f)
        dataset = MyDataset()
        for design in design_lst:
            with open (f"/home/coguest5/hdl_fusion/data_collect/label/ep_lst/{design}.json", 'r') as f:
                reg_lst = json.load(f)

            for ep in reg_lst:
                # print(design + ' ' + ep)
                if not os.path.exists(f"/home/coguest5/hdl_fusion/text_enc/llm_extra/rtl_func_ori/{design}/{ep}.txt"):
                    print("Not exist: ", design, ep)
                    continue
                if not os.path.exists(f"/home/coguest5/hdl_fusion/text_enc/llm_extra/rtl_func_pos/{design}/{ep}.txt"):
                    print("Not exist: ", design, ep)
                    continue
                neg_mapped = reg_map[design][ep]['neg1']
                design_neg, ep_neg = neg_mapped[0], neg_mapped[1]
                run_one_ep(design_neg, ep_neg)


        
        # dataset.convert()
    print(len(dataset))
    with open(f'./data_bench/dataset_{tag}_{cmd}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    with open(f'./data_bench/data_dict_{tag}_{cmd}.json', 'w') as f:
        json.dump(data_dict, f)



if __name__ == '__main__':
    global cmd
    cmd = "ori"
    cmd = "pos"
    cmd = "neg"
        
    with open("/home/coguest5/hdl_fusion/data_collect/dataset/json/design_lst/design_lst.json", 'r') as f:
        design_lst = json.load(f)
    
    with open ("../dataset_js/train_lst.json", 'r') as f:
        train_design_lst = json.load(f)
    with open ("../dataset_js/test_lst.json", 'r') as f:
        test_design_lst = json.load(f)
    with open ("../dataset_js/valid_lst.json", 'r') as f:
        valid_design_lst = json.load(f)

    with open ("../dataset_js/demo_lst.json", 'r') as f:
        demo_lst = json.load(f)
    # with open ("../dataset_js/sft_none_lst.json", 'r') as f:
    #     sft_design_lst_all = json.load(f)

    # save_dataset(train_design_lst, 'train')
    # save_dataset(test_design_lst, 'test')

    # save_dataset(valid_design_lst, 'valid')
    # save_dataset(sft_design_lst_all, 'sft')

    save_dataset(demo_lst, 'demo')

    # with open ("../dataset_js/sft_ft_lst.json", 'r') as f:
    #     sft_design_lst_all = json.load(f)
    # save_dataset(sft_design_lst_all, 'sft')

    # with open ("../dataset_js/sft_ft_lst_4.json", 'r') as f:
    #     sft_design_lst_all = json.load(f)
    # save_dataset(sft_design_lst_all, 'sft_4')

    # with open ("../dataset_js/sft_ft_lst_16.json", 'r') as f:
    #     sft_design_lst_all = json.load(f)
    # save_dataset(sft_design_lst_all, 'sft_16')