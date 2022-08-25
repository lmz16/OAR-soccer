import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json


class SAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, att=0, attdim=32):
        super(SAGELayer, self).__init__()
        self.fc = nn.Linear(2*in_dim, out_dim, bias=False)
        self.att = att
        if self.att == 0:
            self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        elif self.att == 3:
            self.attn_fc = nn.Linear(attdim, 1)
    
    def edge_attention(self, edges):
        if self.att == 0:
            z = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            a = self.attn_fc(z)
            return {'e' : torch.sigmoid(a)}
        elif self.att == 1:
            return {'e': torch.ones([len(edges), 1], requires_grad=False).to(edges.src['h'].device)}
        elif self.att == 2:
            return {'e': torch.rand(len(edges), 1, requires_grad=False).to(edges.src['h'].device)}
        elif self.att == 3:
            z = torch.cat([edges.src['t'], edges.dst['t']], dim=1)
            a = self.attn_fc(z)
            return {'e' : torch.sigmoid(a)}

    def message_func(self, edges):
        return {'h' : edges.src['h'], 'e' : edges.data['e']}
    
    def reduce_func(self, nodes):
        f = torch.sum(nodes.mailbox['e'] * nodes.mailbox['h'], dim=1)
        return {'f': f}
    
    def forward(self, g, x, y=None):
        if y is not None:
            g.ndata['t'] = y
        g.ndata['h'] = x
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return self.fc(torch.cat([g.ndata.pop('f'), x], dim=1))

class NodeAttribute(nn.Module):
    def __init__(self,
                lstm_hidden_dim=16,
                lstm_layers=3):
        super(NodeAttribute, self).__init__()
        self.lstm = nn.GRU(4, lstm_hidden_dim, lstm_layers)
        self.init_parameters(lstm_layers)
    
    def init_parameters(self, lstm_layers):
        for i in range(lstm_layers):
            nn.init.orthogonal_(getattr(self.lstm, "weight_ih_l{}".format(i)))
            nn.init.constant_(getattr(self.lstm, "bias_ih_l{}".format(i)), 0.)
            nn.init.orthogonal_(getattr(self.lstm, "weight_hh_l{}".format(i)))
            nn.init.constant_(getattr(self.lstm, "bias_hh_l{}".format(i)), 0.)

    def forward(self, trajs, node_a=None):
        _, hn = self.lstm(trajs.permute(1, 0, 2))
        b = hn.shape[1]
        traj_feat = hn.permute(1, 0, 2).reshape((b, -1))
        if node_a is None:
            return traj_feat
        else:
            return torch.cat([traj_feat, node_a], dim=1)

class TrajAttribute(nn.Module):
    def __init__(self,
                lstm_hidden_dim=16,
                lstm_layers=3,
                input_dim=3):
        super(TrajAttribute, self).__init__()
        self.lstm = nn.GRU(input_dim, lstm_hidden_dim, lstm_layers)
        self.init_parameters(lstm_layers)

    def init_parameters(self, lstm_layers):
        for i in range(lstm_layers):
            nn.init.orthogonal_(getattr(self.lstm, "weight_ih_l{}".format(i)))
            nn.init.constant_(getattr(self.lstm, "bias_ih_l{}".format(i)), 0.)
            nn.init.orthogonal_(getattr(self.lstm, "weight_hh_l{}".format(i)))
            nn.init.constant_(getattr(self.lstm, "bias_hh_l{}".format(i)), 0.)

    def forward(self, trajs):
        _, hn = self.lstm(trajs.permute(1, 0, 2))
        traj_feat = hn[-1, :, :]
        return traj_feat


class PassModel_SAGE(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(PassModel_SAGE, self).__init__()
        self.layer1_1 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_2 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_3 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_4 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
        self.aa = TrajAttribute(att_dim, 3, 4)
        self.na = NodeAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 4)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        att_feat = self.aa(x)
        h1 = self.layer1_1(g, traj_feat, att_feat)
        h2 = self.layer1_2(g, traj_feat, att_feat)
        h3 = self.layer1_3(g, traj_feat, att_feat)
        h4 = self.layer1_4(g, traj_feat, att_feat)
        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = F.elu(h)
        g_feat = self.layer2(g, h)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))


class ShotModel_SAGE(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_SAGE, self).__init__()
        self.layer1_1 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_2 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_3 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_4 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
        self.aa = TrajAttribute(att_dim, 3, 4)
        self.na = NodeAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 4)
        self.predictor = Predictor(na_lstm_hidden_dim+gcn_hidden_dim, linear_hidden_dim)

    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        att_feat = self.aa(x)
        h1 = self.layer1_1(g, traj_feat, att_feat)
        h2 = self.layer1_2(g, traj_feat, att_feat)
        h3 = self.layer1_3(g, traj_feat, att_feat)
        h4 = self.layer1_4(g, traj_feat, att_feat)
        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = F.elu(h)
        g_feat = self.layer2(g, h)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, feature):
        linear_feat = torch.relu(self.linear1(feature))
        return self.linear2(linear_feat)


class BCELossWithWeight(nn.Module):
    def __init__(self, weights=(1, 1)):
        super().__init__()
        assert len(weights) == 2
        self.weights = weights
    
    def forward(self, pred, targ):
        n = pred.shape[0]
        loss = -1/n*torch.sum(self.weights[0]*torch.log(pred)*targ + self.weights[1]*torch.log(1-pred)*(1-targ))
        return loss

class BCELossWithPower(nn.Module):
    def __init__(self, weights=(1, 1)):
        super().__init__()
        assert len(weights) == 2
        self.weights = weights
    
    def forward(self, pred, targ):
        n = pred.shape[0]
        loss = -1/n*torch.sum(-self.weights[0]*(1-pred)**3*targ + self.weights[1]*torch.log(1-pred)*(1-targ))
        return loss

class PowerLoss(nn.Module):
    def __init__(self, weights=(1, 1)):
        super().__init__()
        assert len(weights) == 2
        self.weights = weights
    
    def forward(self, pred, targ):
        n = pred.shape[0]
        loss = -1/n*torch.sum(-self.weights[0]*(1-pred)**3*targ - self.weights[1]*pred**3*(1-targ))
        return loss


if __name__ == "__main__":
    pass