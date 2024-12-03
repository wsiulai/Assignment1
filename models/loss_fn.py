import torch
import torch.nn.functional as F


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CMAELoss(torch.nn.Module):
    def __init__(self, margin_rr=1.0, margin_nn=1.0, margin_rn=1.0, margin_nr=1.0,
                 lamda_rr=1.0, lamda_nn=1.0, lamda_rn=0.001, lamda_nr=0.001, lamda_mae=1.0):
        super(CMAELoss, self).__init__()
        self.margin_rr = margin_rr
        self.margin_nn = margin_nn
        self.margin_rn = margin_rn
        self.margin_nr = margin_nr
        self.lamda_rr = lamda_rr
        self.lamda_nn = lamda_nn
        self.lamda_rn = lamda_rn
        self.lamda_nr = lamda_nr

        self.lamda_mae = lamda_mae
    
    def forward(self, mae_loss, rtl_emb, net_emb, lamda_rr=None, lamda_nn=None, lambda_cs=None, lamda_mae=None):
        mae_loss_rtl = mae_loss
        rtl_ori, rtl_pos, rtl_neg = rtl_emb
        net_ori, net_pos, net_neg = net_emb
        
        if lamda_rr:
            self.lamda_rr = lamda_rr
        if lamda_nn:
            self.lamda_nn = lamda_nn
        if lambda_cs:
            self.lamda_rn = self.lamda_nr = lambda_cs
        if lamda_mae:
            self.lamda_mae = lamda_mae
            
        loss_rtl = self.triplet_loss(rtl_ori, rtl_pos, rtl_neg, self.margin_rr)
        loss_net = self.triplet_loss(net_ori, net_pos, net_neg, self.margin_nn)
        loss_rn = self.triplet_loss(rtl_ori, net_pos, net_neg, self.margin_rn)
        loss_nr = self.triplet_loss(net_ori, rtl_pos, rtl_neg, self.margin_nr)

        loss_joint = self.lamda_rr * loss_rtl + self.lamda_nn * loss_net +\
                     self.lamda_rn * loss_rn + self.lamda_nr * loss_nr +\
                     0.2*(self.lamda_mae * mae_loss_rtl)
        
        return loss_joint

    def triplet_loss(self, anchor, positive, negative, margin):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()