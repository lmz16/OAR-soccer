from model.attention_model import PassModel_MSAGEV
from model.model import BCELossWithWeight
from dataset import DataLoader_V
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
parser.add_argument("--data_dir", type=str, default='data', required=False)
parser.add_argument("--log_dir", type=str, default='logs/', required=False)
parser.add_argument("--na_lstm_hidden_dim", type=int, default=16, required=False)
parser.add_argument("--na_lstm_hidden_layer", type=int, default=3, required=False)
parser.add_argument("--gcn_hidden_dim", type=int, default=96, required=False)
parser.add_argument("--linear_hidden_dim", type=int, default=256, required=False)

parser.add_argument("--gpu_id", type=int, default=0, required=False)
parser.add_argument("--batch_size", type=int, default=4, required=False)
parser.add_argument("--epochs", type=int, default=40, required=False)
parser.add_argument("--lr", type=float, default=5e-4, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-7, required=False)
parser.add_argument("--save_interval", type=int, default=50, required=False)
parser.add_argument("--save_dir", type=str, default='checkpoints', required=False)
parser.add_argument("--delay", type=int, default=0, required=False)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    writer = SummaryWriter('{0}/pass_video_{1}_{2}'.format(args.log_dir, args.delay, time.strftime("%m%d%H%M")))
    train_model = PassModel_MSAGEV(
        args.na_lstm_hidden_dim,
        args.na_lstm_hidden_layer,
        args.gcn_hidden_dim,
        args.linear_hidden_dim,
    ).to(device)
    train_data_path = "{0}/train_pass_delay_{1}_mode_0.hdf5".format(args.data_dir, args.delay)
    test_data_path = "{0}/test_pass_delay_{1}_mode_0.hdf5".format(args.data_dir, args.delay)
    train_dataloader = DataLoader_V(train_data_path, "data/crops/pass_{}".format(args.delay), device)
    test_dataloader = DataLoader_V(test_data_path, "data/crops/pass_{}".format(args.delay), device)
    optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = BCELossWithWeight(weights=(2, 1))
    for e in range(args.epochs):
        if e < 8:
            for params in optimizer.param_groups:
                params['lr'] *= 0.8
        loss_total = 0
        train_model.train()
        TP, TN, FP, FN = 0, 0, 0, 0
        for b in tqdm(range(0, len(train_dataloader), args.batch_size)):
            loss = 0
            for i in range(b, min(len(train_dataloader), b+args.batch_size)):
                ptraj, esrc, edst, true_from, true_to, fake_from, fake_to, imgs = train_dataloader[i]
                if true_from is None:
                    continue
                if true_from == true_to:
                    continue
                g = dgl.graph((esrc, edst)).to(device)

                q_from = [true_from, true_from, fake_from]
                q_to = [true_to, fake_to, fake_to]
                pred = train_model(g, ptraj, q_from, q_to, imgs, [true_from, true_to, fake_from, fake_to])
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
                ptraj, esrc, edst, true_from, true_to, fake_from, fake_to, imgs = test_dataloader[i]
                if true_from is None:
                    continue
                if true_from == true_to:
                    continue
                g = dgl.graph((esrc, edst)).to(device)

                q_from = [true_from, true_from, fake_from]
                q_to = [true_to, fake_to, fake_to]
                pred = train_model(g, ptraj, q_from, q_to, imgs, [true_from, true_to, fake_from, fake_to])

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