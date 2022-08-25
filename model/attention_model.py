import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json

import dgl
from model import NodeAttribute, Predictor
from video_model import ConvGRU
from dgl.nn import GatedGraphConv as GGNN

class NodeAttribute_S(nn.Module):
    def __init__(self,
                lstm_hidden_dim=16,
                lstm_layers=3,
                feat_dim=3):
        super(NodeAttribute_S, self).__init__()
        self.lstm = nn.GRU(feat_dim, lstm_hidden_dim, lstm_layers)
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


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, att=0, attdim=32):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
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
        h = self.fc(x)
        g.ndata['h'] = h
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('f')


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

class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, att=0, attdim=32):
        super(GINLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.register_parameter('epsilon', torch.nn.Parameter(torch.tensor([0.])))
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
        return self.fc((1+self.epsilon)*x+g.ndata.pop('f'))

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim, att_mode, att_dim=32):
        super(GAT, self).__init__()
        # self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim1, num_heads1)
        # self.layer2 = MultiHeadGATLayer(hidden_dim2 * num_heads1, out_dim, 1)
        self.layer1 = GATLayer(in_dim, hidden_dim1, att_mode, att_dim)
        # self.layer2 = GATLayer(hidden_dim1, hidden_dim2)
        # self.layer3 = GATLayer(hidden_dim2, out_dim)
        if att_mode == 3:
            self.layer2 = GATLayer(hidden_dim1, out_dim, 0)
        else:
            self.layer2 = GATLayer(hidden_dim1, out_dim, att_mode)
    
    def forward(self, g, x, y=None):
        if y is None:
            h = self.layer1(g, x)
        else:
            h = self.layer1(g, x, y)
        h = F.elu(h)
        h = self.layer2(g, h)
        # h = F.elu(h)
        # h = self.layer3(g, h)
        return h

class MultiGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, att_mode, att_dim=32):
        super(MultiGAT, self).__init__()
        self.layer1_1 = GATLayer(in_dim, hidden_dim, att_mode, att_dim)
        self.layer1_2 = GATLayer(in_dim, hidden_dim, att_mode, att_dim)
        self.layer1_3 = GATLayer(in_dim, hidden_dim, att_mode, att_dim)
        self.layer1_4 = GATLayer(in_dim, hidden_dim, att_mode, att_dim)
        if att_mode == 3:
            self.layer2 = GATLayer(hidden_dim*4, out_dim, 0)
        else:
            self.layer2 = GATLayer(hidden_dim*4, out_dim, att_mode)
    
    def forward(self, g, x, y=None):
        if y is None:
            h1 = self.layer1_1(g, x)
            h2 = self.layer1_2(g, x)
            h3 = self.layer1_3(g, x)
            h4 = self.layer1_3(g, x)
        else:
            h1 = self.layer1_1(g, x, y)
            h2 = self.layer1_2(g, x, y)
            h3 = self.layer1_3(g, x, y)
            h4 = self.layer1_4(g, x, y)
        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h

    
class PassModel_GAT(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(PassModel_GAT, self).__init__()
        self.graph_model = GAT(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 3, 6)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        if self.att_mode == 3:
            ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, traj_feat, att_feat)
        else:
            g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class PassModel_GGNN(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256):
        super(PassModel_GGNN, self).__init__()
        self.graph_model = GGNN(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim, 2, 1)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class PassModel_GIN(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(PassModel_GIN, self).__init__()
        self.graph_layer1 = GINLayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim, att_mode)
        if att_mode == 3:
            self.graph_layer2 = GINLayer(gcn_hidden_dim, gcn_hidden_dim, 0)
            self.aa = TrajAttribute(att_dim, 3, 6)
        else:
            self.graph_layer2 = GINLayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.att_mode = att_mode
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        if self.att_mode == 3:
            ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_layer1(g, traj_feat, att_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        else:
            g_feat = self.graph_layer1(g, traj_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class PassModel_SAGE(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(PassModel_SAGE, self).__init__()
        self.graph_layer1 = SAGELayer(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim, att_mode)
        if att_mode == 3:
            self.graph_layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
            self.aa = TrajAttribute(att_dim, 3, 6)
        else:
            self.graph_layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.att_mode = att_mode
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        if self.att_mode == 3:
            ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_layer1(g, traj_feat, att_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        else:
            g_feat = self.graph_layer1(g, traj_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class PassModel_MSAGEV(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=96,
                 linear_hidden_dim=256,
                 video_hidden_dim=9*16,
                 att_dim=16):
        super(PassModel_MSAGEV, self).__init__()
        self.video_hidden_dim = video_hidden_dim
        self.layer1_1 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_2 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_3 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_4 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
        self.aa = TrajAttribute(att_dim, 3, 6)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.videomodel = ConvGRU(3, [32, 32, 16], [3, 5, 3], 3)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)
        self.meanpooling = nn.AdaptiveMaxPool2d(3)

    def forward(self, g, x, q_from, q_to, img, imginds):
        img = img.permute([0, 4, 2, 3, 1])
        hidden = [None, None, None]
        for i in range(8):
            hidden = self.videomodel(img[:, :, :, :, i], hidden)
        video_feat = self.meanpooling(hidden[-1]).reshape([4, -1])
        v_feat = torch.zeros([23, self.video_hidden_dim]).to(video_feat.device)
        v_feat[imginds, :] = video_feat
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
        h1 = self.layer1_1(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h2 = self.layer1_2(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h3 = self.layer1_3(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h4 = self.layer1_4(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = F.elu(h)
        g_feat = self.layer2(g, h)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))



class PassModel_MGAT(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16):
        super(PassModel_MGAT, self).__init__()
        self.graph_model = MultiGAT(na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, gcn_hidden_dim, att_mode)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 3, 6)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)
    
    def forward(self, g, x, q_from, q_to):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        if self.att_mode == 3:
            ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, traj_feat, att_feat)
        else:
            g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_GAT(nn.Module):
    def __init__(self, 
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_GAT, self).__init__()
        self.graph_model = GAT(gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 2, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)
    
    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        if self.att_mode == 3:
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, traj_feat, att_feat)
        else:
            g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_MGAT(nn.Module):
    def __init__(self, 
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_MGAT, self).__init__()
        self.graph_model = MultiGAT(gcn_hidden_dim, gcn_hidden_dim//4, gcn_hidden_dim, att_mode)
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 2, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)
    
    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        if self.att_mode == 3:
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, traj_feat, att_feat)
        else:
            g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_GGNN(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128):
        super(ShotModel_GGNN, self).__init__()
        self.graph_model = GGNN(gcn_hidden_dim, gcn_hidden_dim, 2, 1)
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)
    
    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_GIN(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_GIN, self).__init__()
        self.graph_layer1 = GINLayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        if att_mode == 3:
            self.graph_layer2 = GINLayer(gcn_hidden_dim, gcn_hidden_dim, 0)
            self.aa = TrajAttribute(att_dim, 2, 6)
        else:
            self.graph_layer2 = GINLayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.att_mode = att_mode
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)

    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        if self.att_mode == 3:
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_layer1(g, traj_feat, att_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        else:
            g_feat = self.graph_layer1(g, traj_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        union_feat = torch.cat([g_feat[q_from, :],  
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_SAGE(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_SAGE, self).__init__()
        self.graph_layer1 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        if att_mode == 3:
            self.graph_layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
            self.aa = TrajAttribute(att_dim, 2, 6)
        else:
            self.graph_layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.att_mode = att_mode
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)

    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        if self.att_mode == 3:
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_layer1(g, traj_feat, att_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        else:
            g_feat = self.graph_layer1(g, traj_feat)
            g_feat = F.elu(g_feat)
            g_feat = self.graph_layer2(g, g_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_MSAGEV(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=96,
                 linear_hidden_dim=128,
                 video_hidden_dim=9*16,
                 att_dim=16):
        super(ShotModel_MSAGEV, self).__init__()
        self.video_hidden_dim = video_hidden_dim
        self.layer1_1 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_2 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_3 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer1_4 = SAGELayer(video_hidden_dim+na_lstm_hidden_dim*na_lstm_hidden_layer, gcn_hidden_dim//4, 3, 2*att_dim)
        self.layer2 = SAGELayer(gcn_hidden_dim, gcn_hidden_dim, 0)
        self.aa = TrajAttribute(att_dim, 3, 6)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.videomodel = ConvGRU(3, [32, 32, 16], [3, 5, 3], 3)
        self.predictor = Predictor(na_lstm_hidden_dim+gcn_hidden_dim, linear_hidden_dim)
        self.meanpooling = nn.AdaptiveMaxPool2d(3)

    def forward(self, g, x, q_from, img, imginds):
        img = img.permute([0, 4, 2, 3, 1])[[0, 1], :, :, :, :]
        hidden = [None, None, None]
        for i in range(8):
            hidden = self.videomodel(img[:, :, :, :, i], hidden)
        video_feat = self.meanpooling(hidden[-1]).reshape([2, -1])
        v_feat = torch.zeros([23, self.video_hidden_dim]).to(video_feat.device)
        v_feat[imginds, :] = video_feat
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
        h1 = self.layer1_1(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h2 = self.layer1_2(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h3 = self.layer1_3(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h4 = self.layer1_4(g, torch.cat([traj_feat, v_feat], dim=1), att_feat)
        h = torch.cat([h1, h2, h3, h4], dim=1)
        h = F.elu(h)
        g_feat = self.layer2(g, h)
        union_feat = torch.cat([g_feat[q_from, :],
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class PassModel_MM(nn.Module):
    def __init__(self,
                 na_lstm_hidden_dim=16,
                 na_lstm_hidden_layer=3,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=256,
                 att_mode=0,
                 att_dim=16,
                 y_dim=32):
        super(PassModel_GAT, self).__init__()
        self.graph_model = GAT(na_lstm_hidden_dim*na_lstm_hidden_layer+y_dim, gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.na = NodeAttribute_S(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 3, 6)
        self.predictor = Predictor(2*(na_lstm_hidden_dim+gcn_hidden_dim), linear_hidden_dim)

    def forward(self, g, x, q_from, q_to, y):
        traj_feat = self.na(x)
        traj_feat2 = self.ta(x)
        if self.att_mode == 3:
            ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, torch.cat([traj_feat, y], dim=1), att_feat)
        else:
            g_feat = self.graph_model(g, torch.cat([traj_feat, y], dim=1))
        union_feat = torch.cat([g_feat[q_from, :], 
                                g_feat[q_to, :], 
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))

class ShotModel_MM(nn.Module):
    def __init__(self, 
                 na_lstm_hidden_dim=128,
                 na_lstm_hidden_layer=2,
                 gcn_hidden_dim=64,
                 linear_hidden_dim=128,
                 att_mode=0,
                 att_dim=16):
        super(ShotModel_GAT, self).__init__()
        self.graph_model = GAT(gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, gcn_hidden_dim, att_mode)
        self.na = TrajAttribute(gcn_hidden_dim, 1)
        self.ta = TrajAttribute(na_lstm_hidden_dim, na_lstm_hidden_layer, 6)
        self.att_mode = att_mode
        if att_mode == 3:
            self.aa = TrajAttribute(att_dim, 2, 6)
        self.predictor = Predictor(gcn_hidden_dim+na_lstm_hidden_dim, linear_hidden_dim)
    
    def forward(self, g, x, q_from):
        traj_feat = self.na(x)
        ball_traj = torch.repeat_interleave(x[0, :, :].unsqueeze(0), repeats=x.shape[0], dim=0)
        traj_feat2 = self.ta(torch.cat([ball_traj, x], dim=2))
        if self.att_mode == 3:
            att_feat = self.aa(torch.cat([ball_traj, x], dim=2))
            g_feat = self.graph_model(g, traj_feat, att_feat)
        else:
            g_feat = self.graph_model(g, traj_feat)
        union_feat = torch.cat([g_feat[q_from, :], 
                                traj_feat2[q_from, :]], dim=1)
        return torch.sigmoid(self.predictor(union_feat))