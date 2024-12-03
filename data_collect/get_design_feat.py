import json, pickle, re
import networkx as nx
import DG

def get_node_tpe(node_name, node_dict):
    tpe_ = 0
    if not node_name in node_dict:
        print(node_name)
        tpe_ = 0
        return tpe_
    
    node_class = node_dict[node_name]
    node_tpe_ori = node_class.tpe
    if node_tpe_ori in ['DFF']:
        tpe_ = 1
    elif node_tpe_ori in ['Input', 'Inout', None]:
        tpe_ = 2
    elif node_tpe_ori in ['BUF']:
        tpe_ = 3
    elif node_tpe_ori in ['INV']:
        tpe_ = 4
    elif node_tpe_ori in ['AND']:
        tpe_ = 5
    elif node_tpe_ori in ['OR']:
        tpe_ = 6
    elif node_tpe_ori in ['XOR']:
        tpe_ = 7
    elif node_tpe_ori in ['NOR']:
        tpe_ = 8
    elif node_tpe_ori in ['XNOR']:
        tpe_ = 9
    elif node_tpe_ori in ['NAND']:
        tpe_ = 10
    elif node_tpe_ori in ['AOI']:
        tpe_ = 11
    elif node_tpe_ori in ['OAI']:
        tpe_ = 12
    elif node_tpe_ori in ['MUX']:
        tpe_ = 13
    elif node_tpe_ori in ['DLL']:
        tpe_ = 14
    elif node_tpe_ori in ['HA']:
        tpe_ = 15
    elif node_tpe_ori in ['FA']:
        tpe_ = 16
    # else:
    #     print(node_tpe_ori)
    #     assert False
    
    return tpe_

def run_one_design(design):
    print('Current Design: ', design)
    with open (f"../{cmd}/{design}_graph.pkl", 'rb') as f:
        g = pickle.load(f)
    g_nx = nx.DiGraph(g)

    with open(f"../{cmd}/{design}_node_dict.pkl", 'rb') as f:
        node_dict = pickle.load(f)

    vec = [0 for i in range(17)]
    
    for node in g_nx.nodes():
        # print(node, node_dict[node])
        # break
        tpe = get_node_tpe(node, node_dict)
        vec[tpe] += 1
    vec = vec[1:]
    print(vec)

    with open (f"./{cmd}/{design}_feat.json", 'w') as f:
        json.dump(vec, f)

if __name__ == '__main__':
    global cmd
    cmd = "ori"


    with open("/home/coguest5/rtl_repr/data_collect/dataset/json/design_lst/design_lst.json", 'r') as f:
        design_lst = json.load(f)
    
    # design_lst = ['b20']

    for design in design_lst:
        run_one_design(design)