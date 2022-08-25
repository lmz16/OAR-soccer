from model.model import PassModel_SAGE, BCELossWithWeight
from dataset import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import dgl
import os
import time
import random
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='data/matches', required=False)
parser.add_argument("--task", type=str, default='recog', required=False)
parser.add_argument("--log_dir", type=str, default='logs/', required=False)
parser.add_argument("--na_lstm_hidden_dim", type=int, default=16, required=False)
parser.add_argument("--na_lstm_hidden_layer", type=int, default=3, required=False)
parser.add_argument("--gcn_hidden_dim", type=int, default=64, required=False)
parser.add_argument("--linear_hidden_dim", type=int, default=256, required=False)

parser.add_argument("--gpu_id", type=int, default=4, required=False)
parser.add_argument("--batch_size", type=int, default=8, required=False)
parser.add_argument("--epochs", type=int, default=30, required=False)
parser.add_argument("--lr", type=float, default=3e-4, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-7, required=False)
parser.add_argument("--save_interval", type=int, default=50, required=False)
parser.add_argument("--save_dir", type=str, default='checkpoints', required=False)
parser.add_argument("--attmode", type=int, default=3, required=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    writer = SummaryWriter('{0}/{1}_pass_sc_{2}'.format(args.log_dir, time.strftime("%m%d%H%M"), args.task))
    train_model = PassModel_SAGE(
        args.na_lstm_hidden_dim,
        args.na_lstm_hidden_layer,
        args.gcn_hidden_dim,
        args.linear_hidden_dim,
        args.attmode
        ).to(device)
    train_dataloader = DataLoader("{0}/train_{1}_pass.hdf5".format(args.data_dir, args.task), device)
    test_dataloader = DataLoader("{0}/test_{1}_pass.hdf5".format(args.data_dir, args.task), device)
    optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1, 1]))
    # criterion = torch.nn.BCELoss()
    criterion = BCELossWithWeight(weights=(2, 1))
    for e in range(args.epochs):
        loss_total = 0

        train_model.train()
        TP, TN, FP, FN = 0, 0, 0, 0
        for b in tqdm(range(0, len(train_dataloader), args.batch_size)):
            loss = 0
            for i in range(b, min(len(train_dataloader), b+args.batch_size)):
                ptraj, esrc, edst, scalar, p_num = train_dataloader[i]
                pass_from, pass_to = scalar[0], scalar[1]
                if pass_from == pass_to:
                    continue
                g = dgl.graph((esrc, edst)).to(device)
                cands = list(range(p_num))
                cands.remove(pass_from)
                cands.remove(pass_to)
                fake_from, fake_to = random.sample(cands, 2)
                q_from = [pass_from, pass_from, fake_from]
                q_to = [pass_to, random.choice(cands), fake_to]

                pred = train_model(g, ptraj, q_from, q_to)
                gt = np.array([[1], [0], [0]])
                gt = torch.from_numpy(gt).to(device)

                loss += criterion(pred, gt)

                pred_ = pred.detach().cpu().numpy()
                if pred_[0] > 0.5:
                    TP += 1
                else:
                    FN += 1
                if pred_[1] > 0.5:
                    FP += 1
                else:
                    TN += 1
                if pred_[2] > 0.5:
                    FP += 1
                else:
                    TN += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            loss_total += loss.item()
        print("epoch:{0}/{1}, total loss:{2:.5f}, train acc:{3:.3f}, train recall:{4:.3f}".format(e+1, args.epochs, loss_total, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN+0.0001)))
        writer.add_scalar('loss', loss_total, e)
        writer.add_scalar('train_acc', (TP+TN)/(TP+TN+FP+FN), e)
        writer.add_scalar('train_recall', TP/(TP+FN+0.0001), e)

        train_model.eval()
        TP, TN, FP, FN = 0, 0, 0, 0
        start_time = time.time()
        for i in tqdm(range(len(test_dataloader))):
            with torch.no_grad():
                ptraj, esrc, edst, scalar, p_num = test_dataloader[i]

                if pass_from == pass_to:
                    continue
                g = dgl.graph((esrc, edst)).to(device)

                pass_from, pass_to = scalar[0], scalar[1]
                cands = list(range(p_num))
                cands.remove(pass_from)
                cands.remove(pass_to)

                fake_from, fake_to = random.sample(cands, 2)
                q_from = [pass_from, pass_from, fake_from]
                q_to = [pass_to, random.choice(cands), fake_to]
                pred = train_model(g, ptraj, q_from, q_to)

                pred_ = pred.cpu().numpy()
                if pred_[0] > 0.5:
                    TP += 1
                else:
                    FN += 1
                if pred_[1] > 0.5:
                    FP += 1
                else:
                    TN += 1
                if pred_[2] > 0.5:
                    FP += 1
                else:
                    TN += 1
        end_time = time.time()

        print("epoch:{0}/{1}, valid acc:{2:.4f}, valid recall:{3:.4f}, valid prec:{4:.4f}, inference time:{5:.4f}".format(e+1, args.epochs, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN+0.0001), TP/(TP+FP+0.0001),(end_time-start_time)/len(test_dataloader)))
        writer.add_scalar('valid_acc', (TP+TN)/(TP+TN+FP+FN), e)
        writer.add_scalar('valid_recall', TP/(TP+FN+0.0001), e)
        writer.add_scalar('valid_prec', TP/(TP+FP+0.0001), e)

            
if __name__ == "__main__":
    train(args)