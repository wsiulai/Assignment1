import torch
import numpy as np
import pickle, json, time, re, sys, os
from DG import Graph, Node
import networkx as nx
from multiprocessing import Pool
import dgl
from dgl import from_networkx
import dgl


if __name__ == '__main__':
        global dataset, cmd

        with open("/home/coguest5/hdl_fusion/data_collect/dataset/json/design_lst/design_lst.json", 'r') as f:
            design_lst = json.load(f)
        
        for design in design_lst:

            ep_lst = []

            with open (f"/home/coguest5/hdl_fusion/data_collect/label/ep_lst_pt/{design}.json", 'r') as f:
                reg_lst = json.load(f)

            for ep in reg_lst:
                folder_dir = f'/home/coguest5/hdl_fusion/data_collect/dataset/ori/graph'
                pkl_path = f'{folder_dir}/{design}/{ep}_ast.pkl'
                if not os.path.exists(pkl_path):
                    print('ori', design, ep)
                    continue
                folder_dir = f'/home/coguest5/hdl_fusion/data_collect/dataset/pos/graph'
                pkl_path = f'{folder_dir}/{design}/{ep}_ast.pkl'
                if not os.path.exists(pkl_path):
                    print('pos', design, ep)
                    continue
                folder_dir = f'/home/coguest5/hdl_fusion/data_collect/dataset/neg/graph'
                pkl_path = f'{folder_dir}/{design}/{ep}_ast.pkl'
                if not os.path.exists(pkl_path):
                    print('neg', design, ep)
                    continue

                ep_lst.append(ep)
                with open (f"/home/coguest5/hdl_fusion/data_collect/label/ep_lst/{design}.json", 'w') as f:
                    json.dump(ep_lst, f)



