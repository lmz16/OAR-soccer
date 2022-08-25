from model.baseline_model import MLP_Pass, LSTM_SC_Shot
from model.model import PowerLoss, BCELossWithWeight
from dataset import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import random
import numpy as np
import os
import time
from utils import count_TFPN
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='data/matches', required=False)
parser.add_argument("--log_dir", type=str, default='logs/', required=False)
parser.add_argument("--task", type=str, default='recog', required=False)

parser.add_argument("--gpu_id", type=int, default=7, required=False)
parser.add_argument("--batch_size", type=int, default=16, required=False)
parser.add_argument("--epochs", type=int, default=20, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-8, required=False)
parser.add_argument("--save_interval", type=int, default=50, required=False)
parser.add_argument("--save_dir", type=str, default='checkpoints', required=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    writer = SummaryWriter('{0}/{1}_shot_lstm_sc_{2}'.format(args.log_dir, time.strftime("%m%d%H%M"), args.task))
    # train_model = MLP_Pass(76).to(device)
    train_model = LSTM_SC_Shot(inp_dim=4).to(device)
    train_data_path = "{0}/train_{1}_shot.hdf5".format(args.data_dir, args.task)
    test_data_path = "{0}/test_{1}_shot.hdf5".format(args.data_dir, args.task)
    train_dataloader = DataLoader(train_data_path, device)
    test_dataloader = DataLoader(test_data_path, device, flip1=False, flip2=False)
    optimizer = optim.SGD(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1, 1]))
    # criterion = torch.nn.BCELoss()
    # criterion = PowerLoss(weights=(3, 1))
    criterion = BCELossWithWeight(weights=(3, 1))
    for e in range(args.epochs):
        loss_total = 0

        train_model.train()
        TP, TN, FP, FN = 0, 0, 0, 0
        for b in tqdm(range(0, len(train_dataloader), args.batch_size)):
            loss = 0
            for i in range(b, min(len(train_dataloader), b+args.batch_size)):
                ptraj, _, _, scalar, p_num = train_dataloader[i]
                label, shot_from = scalar[0], scalar[2]
                label = 0 if label < 0 else 1
                if shot_from == -1:
                    continue
                cands = list(range(p_num))
                cands.remove(shot_from)
                fake_from = random.choice(cands)

                # x1 = ptraj[shot_from, :, [0, 1, 3]].reshape(-1)
                # t1 = torch.from_numpy(np.array([ptraj[shot_from, 0, 2].cpu().numpy()])).to(device)
                # f1 = torch.cat([x1, t1])

                # x2 = ptraj[fake_from, :, [0, 1, 3]].reshape(-1)
                # t2 = torch.from_numpy(np.array([ptraj[fake_from, 0, 2].cpu().numpy()])).to(device)
                # f2 = torch.cat([x2, t2])

                # pred = train_model(torch.stack([f1, f2], dim=0))
                pred = train_model(ptraj, [shot_from, fake_from])
                gt = np.array([[label], [0]])
                gt = torch.from_numpy(gt).to(device)
                loss += criterion(pred, gt)
                
                pred_ = pred.detach().cpu().numpy()
                tp, tn, fp, fn = count_TFPN(pred_, [label, 0], 'nn')
                TP += tp
                TN += tn
                FP += fp
                FN += fn

            optimizer.zero_grad()
            loss = loss / args.batch_size
            loss.backward()
            optimizer.step()  
            loss_total += loss.item()
        print("epoch:{0}/{1}, total loss:{2:.5f}, train acc:{3:.3f}, train recall:{4:.3f}".format(e+1, args.epochs, loss_total, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN+0.01)))
        writer.add_scalar('loss', loss_total, e)
        writer.add_scalar('train_acc', (TP+TN)/(TP+TN+FP+FN), e)
        writer.add_scalar('train_recall', TP/(TP+FN+0.01), e)

        train_model.eval()
        TP, TN, FP, FN = 0, 0, 0, 0
        start_time = time.time()
        for i in tqdm(range(len(test_dataloader))):
            with torch.no_grad():
                ptraj, _, _, scalar, p_num= test_dataloader[i]
                label, shot_from = scalar[0], scalar[2]
                label = 0 if label < 0 else 1
                if shot_from == -1:
                    continue
                cands = list(range(p_num))
                cands.remove(shot_from)
                fake_from = random.choice(cands)

                # x1 = ptraj[shot_from, :, [0, 1, 3]].reshape(-1)
                # t1 = torch.from_numpy(np.array([ptraj[shot_from, 0, 2].cpu().numpy()])).to(device)
                # f1 = torch.cat([x1, t1])

                # x2 = ptraj[fake_from, :, [0, 1, 3]].reshape(-1)
                # t2 = torch.from_numpy(np.array([ptraj[fake_from, 0, 2].cpu().numpy()])).to(device)
                # f2 = torch.cat([x2, t2])

                # pred = train_model(torch.stack([f1, f2], dim=0))
                pred = train_model(ptraj, [shot_from, fake_from])
                pred_ = pred.detach().cpu().numpy()
                tp, tn, fp, fn = count_TFPN(pred_, [label, 0], 'nn')
                TP += tp
                TN += tn
                FP += fp
                FN += fn
        end_time = time.time()
        print("epoch:{0}/{1}, valid acc:{2:.3f}, valid recall:{3:.3f}, valid prec:{4:.3f}, inference time:{5:.4f}".format(e+1, args.epochs, (TP+TN)/(TP+TN+FP+FN), TP/(TP+FN+0.01), TP/(TP+FP+0.01), (end_time-start_time)/len(test_dataloader)))
        writer.add_scalar('valid_acc', (TP+TN)/(TP+TN+FP+FN), e)
        writer.add_scalar('valid_recall', TP/(TP+FN+0.01), e)
        writer.add_scalar('valid_prec', TP/(TP+FP+0.01), e)

if __name__ == "__main__":
    train(args)