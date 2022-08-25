import numpy as np
import torch
from torch import nn
from attention_model import TrajAttribute
from model import Predictor

class SVM(nn.Module):
    def __init__(self, dim):
        super(SVM, self).__init__()
        self.layer = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

def svm_loss(scores, labels):
    loss = 1-labels*scores
    loss[loss<=0] = 0
    return torch.sum(loss)

def sign(x):
    x[x>=0] = 1
    x[x<0] = -1
    return x

def feature_extract(trajs, ind):
    v = trajs[ind, 1::2, [0,1]] - trajs[ind, 0::2, [0,1]]
    d = trajs[ind, ::2, [0,1]] - trajs[0, ::2, [0,1]]
    d2 = d[:, 0]*d[:, 0] + d[:, 1]*d[:, 1]
    l = trajs[ind, ::4, [0,1]]
    return torch.cat([v.reshape(1, -1), d.reshape(1, -1), l.reshape(1, -1), d2.reshape(1, -1)], dim=1).reshape(1, -1)

class MLP_Pass(nn.Module):
    def __init__(self, inp_dim=40):
        super(MLP_Pass, self).__init__()
        self.layer1 = nn.Linear(inp_dim, 256)
        self.layer2 = nn.Linear(256, 32)
        self.layer3 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)
        # nn.init.uniform_(self.layer1.bias, a=0.01, b=0.99)
        # nn.init.uniform_(self.layer2.bias, a=0.01, b=0.99)
        # nn.init.uniform_(self.layer3.bias, a=0.01, b=0.99)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        # x = torch.clamp(torch.sigmoid(x)+0.01, max=0.9999)
        x = torch.sigmoid(x)
        return x

class LSTM_MLP_Pass(nn.Module):
    def __init__(self, lstm_hidden_dim=48, lstm_hidden_layer=1, inp_dim=3):
        super(LSTM_MLP_Pass, self).__init__()
        self.ta1 = TrajAttribute(lstm_hidden_dim, lstm_hidden_layer, inp_dim)
        self.ta2 = TrajAttribute(lstm_hidden_dim, lstm_hidden_layer, inp_dim)
        self.layer1 = nn.Linear(4*lstm_hidden_dim, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)

    def forward(self, x, q_from, q_to):
        traj_feat = self.ta1(x)
        traj_feat2 = self.ta2(x-x[0, :, :])
        union_feat = torch.cat([traj_feat[q_from, :], 
                                traj_feat[q_to, :],
                                traj_feat2[q_from, :],
                                traj_feat2[q_to, :]], dim=1)
        f = self.layer1(union_feat)
        f = torch.relu(f)
        f = self.layer2(f)
        f = torch.relu(f)
        return torch.sigmoid(self.layer3(f))


class LSTM_MLP_Shot(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_hidden_layer=2, inp_dim=4):
        super(LSTM_MLP_Shot, self).__init__()
        self.ta = TrajAttribute(lstm_hidden_dim, lstm_hidden_layer, inp_dim)
        self.layer1 = nn.Linear(lstm_hidden_dim, 32)
        self.layer2 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)

    def forward(self, x):
        traj_feat = self.ta(x)
        f = self.layer1(traj_feat)
        f = torch.relu(f)
        return torch.sigmoid(self.layer2(f))

class LSTM_SC_Pass(nn.Module):
    def __init__(self, lstm_hidden_dim=48, lstm_hidden_layer=1, inp_dim=3):
        super(LSTM_SC_Pass, self).__init__()
        self.ta = TrajAttribute(lstm_hidden_dim, lstm_hidden_layer, inp_dim)
        self.layer1 = nn.Linear(2*lstm_hidden_dim, 128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
        nn.init.xavier_normal_(self.layer3.weight, gain=1)
    
    def forward(self, x, q_from, q_to):
        traj_feat = self.ta(x)
        union_feat = torch.cat([traj_feat[q_from, :], 
                                traj_feat[q_to, :],], dim=1)
        f = self.layer1(union_feat)
        f = torch.relu(f)
        f = self.layer2(f)
        f = torch.relu(f)
        return torch.sigmoid(self.layer3(f))


class LSTM_SC_Shot(nn.Module):
    def __init__(self, lstm_hidden_dim=48, lstm_hidden_layer=1, inp_dim=4):
        super(LSTM_SC_Shot, self).__init__()
        self.ta = TrajAttribute(lstm_hidden_dim, lstm_hidden_layer, inp_dim)
        self.layer1 = nn.Linear(lstm_hidden_dim, 32)
        self.layer2 = nn.Linear(32, 1)
        nn.init.xavier_normal_(self.layer1.weight, gain=1)
        nn.init.xavier_normal_(self.layer2.weight, gain=1)
    
    def forward(self, x, q_from):
        traj_feat = self.ta(x)
        f = self.layer1(traj_feat[q_from, :])
        f = torch.relu(f)
        return torch.sigmoid(self.layer2(f))
        