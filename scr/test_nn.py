from loss_fn_cl import *
from dgl.dataloading import GraphDataLoader
import pickle, os, json
import numpy as np
from stat_ import regression_metrics, classify_metrics
from utils import *
from collections import defaultdict
from models import Graphormer, GSAGE, MyDataset
from net_mae import PreModel

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test_gcl(loader_lst, encoder_rtl, encoder_net, date, epoch):
    rtl_ori_loader, net_ori_loader, rtl_train_ori_loader, net_train_ori_loader = loader_lst
    ### ================== calculate training pool ====================
    net_train_emb = np.empty((0,256))
    for data_train in net_train_ori_loader:
        _, net_train_emb_batch = encoder_net.encoder(data_train, data_train.ndata['feat'])
        net_train_emb = np.append(net_train_emb, net_train_emb_batch.cpu().detach().numpy(), axis=0)
            
    net_train_emb = torch.from_numpy(net_train_emb).to(device)


    ### =================== inference and retrival =====================
    test_dict_all = {}  ## key: design name, val: test & top-k list(dict) [{test_emb, labels (PPA + func)}, {retrival_emb, labels}]

    for i, data in enumerate(zip(rtl_ori_loader, net_ori_loader)):

        # rtl_ori = data[0]
        net_ori = data[1]
        # rtl_ori_emb = encoder_rtl(rtl_ori)
        _, net_ori_emb = encoder_net.encoder(net_ori, net_ori.ndata['feat'])
    
        distance_train = F.pairwise_distance(net_ori_emb, net_train_emb, 2)
        distance_train = distance_train.cpu().detach().numpy()
        top_k_idx = np.argsort(distance_train)[:50]

        ### ground truth ###
        cone_y = data_dict_test[str(i)][0]
        ep_y = data_dict_test[str(i)][1]
        ppa_y = design_ppa_dict[cone_y][ep_y]
        func_y = design_func_dict[cone_y][ep_y]


        ### retrivaled eps ###
        if cone_y not in test_dict_all:
            test_dict_all[cone_y] = {}

        feat_dict = {"emb": list(net_ori_emb.cpu().detach().numpy()[0]), "ppa": ppa_y, "func": func_y, "design": cone_y, "ep": ep_y, "min_dst": np.min(distance_train)}
        
        ep_dict = defaultdict(list)
        ep_dict[ep_y].append(feat_dict)

        for idx in top_k_idx:
            cone = data_dict_train[str(idx)][0]
            ep = data_dict_train[str(idx)][1]
            ppa = design_ppa_dict[cone][ep]
            func = design_func_dict[cone][ep]

            emb = net_train_emb[idx]
            feat_dict = {"emb": list(emb.cpu().detach().numpy()), "ppa": ppa, "func": func, "design": cone, "ep": ep}
            ep_dict[ep_y].append(feat_dict)

        test_dict_all[cone_y].update(ep_dict)

    ### =================== stat =====================
    stat_dict = {}
    for design, dct in test_dict_all.items():
        # ppa_dict = stat_one_design_ppa(design, dct)
        func_dict = stat_one_design_func(design, dct)
        stat_dict[design] = { "func": func_dict}
    
    if not os.path.exists(f"../result/{date}_{epoch}"):
        os.makedirs(f"../result/{date}_{epoch}")
    with open(f"../result/{date}_{epoch}/stat_dict_nn.json", 'w') as f:
        json.dump(stat_dict, f, indent=4)



def test_nn(date, epoch):

    global FP_lst
    FP_lst = []

    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("/home/coguest5/CircuitFusion/data_collect/dataset/json/design_lst/design_lst.json", 'r') as f:
        design_lst = json.load(f)

    global design_ppa_dict, design_func_dict
    design_ppa_dict, design_func_dict = get_all_label_cone(design_lst)

    with open(f'../dataset/data_bench/rtl_test_ori.pkl', 'rb') as f:
        rtl_test_ori = pickle.load(f)
    with open(f'../dataset/data_bench/rtl_train_ori.pkl', 'rb') as f:
        rtl_train_ori = pickle.load(f)
    
    with open(f'../dataset/data_bench/net_test_ori.pkl', 'rb') as f:
        net_test_ori = pickle.load(f)
    with open(f'../dataset/data_bench/net_train_ori.pkl', 'rb') as f:
        net_train_ori = pickle.load(f)
    
    global data_dict_train, data_dict_test
    with open(f'../dataset/data_bench/train_dict.json', 'r') as f:
        data_dict_train = json.load(f)
    with open(f'../dataset/data_bench/test_dict.json', 'r') as f:
        data_dict_test = json.load(f)

    rtl_test_ori.to_device(device)
    rtl_train_ori.to_device(device)
    net_test_ori.to_device(device)
    net_train_ori.to_device(device)

    
    batch_size = 1
    train_size = 512
    rtl_ori_loader = GraphDataLoader(rtl_test_ori, batch_size=batch_size, shuffle=False)
    rtl_train_ori_loader = GraphDataLoader(rtl_train_ori, batch_size=1024, shuffle=False)

    net_ori_loader = GraphDataLoader(net_test_ori, batch_size=batch_size, shuffle=False)
    net_train_ori_loader = GraphDataLoader(net_train_ori, batch_size=1024, shuffle=False)

    


    model_save_path = f"../pretrain_model/{date}/encoder_net_{epoch}.pth"
    encoder_net = torch.load(model_save_path)

    loader_lst = [rtl_ori_loader, net_ori_loader, \
                  rtl_train_ori_loader, net_train_ori_loader]

    test_gcl(loader_lst, "", encoder_net, date, epoch)

    print("finish!")

if __name__ == '__main__':
    date = '0406_w_cmae_cl'
    epoch = '15'
    test_nn(date, epoch)