from model.baseline_model import SVM, svm_loss, feature_extract, MLP_Pass, LSTM_MLP_Shot
from model.model import BCELossWithWeight
from dataset import DataLoader_S
from utils import count_TFPN
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import os
import time
import random
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='data', required=False)
parser.add_argument("--log_dir", type=str, default='logs/', required=False)

parser.add_argument("--gpu_id", type=int, default=7, required=False)
parser.add_argument("--batch_size", type=int, default=1, required=False)
parser.add_argument("--epochs", type=int, default=20, required=False)
parser.add_argument("--lr", type=float, default=1e-3, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-7, required=False)
parser.add_argument("--save_interval", type=int, default=50, required=False)
parser.add_argument("--save_dir", type=str, default='checkpoints', required=False)
parser.add_argument("--ignore", type=int, default=0, required=False)
parser.add_argument("--edgemode", type=int, default=0, required=False)
parser.add_argument("--delay", type=int, default=1, required=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    writer = SummaryWriter('{0}/{1}_shot_compare_delay_{2}'.format(args.log_dir, time.strftime("%m%d%H%M"), args.delay))
    train_model1 = SVM(120).to(device)
    # train_model1 = MLP_Pass(160).to(device)
    # train_model1 = LSTM_MLP_Shot().to(device)
    train_data_path = "{0}/train_shot_delay_{1}_mode_{2}.hdf5".format(args.data_dir, args.delay, args.edgemode)
    test_data_path = "{0}/test_shot_delay_{1}_mode_{2}.hdf5".format(args.data_dir, args.delay, args.edgemode)
    train_dataloader = DataLoader_S(train_data_path, device, ignore=args.ignore)
    test_dataloader = DataLoader_S(test_data_path, device, ignore=args.ignore, flip1=False, flip2=False)
    optimizer1 = optim.Adam(train_model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion1 = svm_loss
    # criterion1 = BCELossWithWeight(weights=(3, 1))
    for e in range(args.epochs):
        loss_total = [0]

        train_model1.train()
        TP, TN, FP, FN = [1], [1], [1], [1]
        for b in tqdm(range(0, len(train_dataloader), 8)):
            batch_x, batch_y = [], []
            for i in range(b, min(b+8, len(train_dataloader))):
                ptraj, esrc, edst, label, pre, shot_from, delay = train_dataloader[i]
                if shot_from is None:
                    continue

                if shot_from < 12:
                    cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    cands.remove(shot_from)
                else:
                    cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                    cands.remove(shot_from)
                q_from = [shot_from, random.choice(cands)]

                f1 = feature_extract(ptraj, q_from[0])
                f2 = feature_extract(ptraj, q_from[1])
                # f3 = feature_extract(ptraj, random.choice(cands))
                # f4 = feature_extract(ptraj, fake_from)
                # f5 = feature_extract(ptraj, fake_to)
                # x1 = torch.cat([f1, f2], dim=1)
                # x2 = torch.cat([f1, f3], dim=1)
                # x3 = torch.cat([f4, f5], dim=1)
                # x1 = torch.cat([ptraj[q_from[0], :, [0, 1]], ptraj[0, :, [0, 1]]], dim=1)
                # x2 = torch.cat([ptraj[q_from[1], :, [0, 1]], ptraj[0, :, [0, 1]]], dim=1)
                # x1 = torch.cat([ptraj[q_from[0], :, [0, 1]], ptraj[0, :, [0, 1]]]).reshape(-1)
                # x2 = torch.cat([ptraj[q_from[1], :, [0, 1]], ptraj[0, :, [0, 1]]]).reshape(-1)
                if i % 2 == 0:
                    batch_x.append(f1)
                    batch_y.append([1])
                elif i % 4 == 1:
                    batch_x.append(f1)
                    batch_y.append([-1])
                else:
                    batch_x.append(f2)
                    batch_y.append([-1])
                # batch_x.append(f1)
                # batch_x.append(f2)
                # batch_y.append([label])
                # batch_y.append([0])
            pred1 = train_model1(torch.stack(batch_x, dim=0))
            # pred1 = train_model1(ptraj, [pass_from, pass_from, fake_from], [pass_to, random.choice(cands), fake_to])
            gt1 = np.array(batch_y)
            gt1 = torch.from_numpy(gt1).to(device)

            loss1 = criterion1(pred1, gt1)
            loss_total[0] += loss1.item()

            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()  
            pred_ = pred1.detach().cpu().numpy()
            tp, tn, fp, fn = count_TFPN(pred_, [x[0] for x in batch_y], 'svm')
            TP[0] += tp
            TN[0] += tn
            FP[0] += fp
            FN[0] += fn
        print("epoch:{0}/{1}, total loss:{2:.5f}, train acc:{3:.3f}, train recall:{4:.3f}".format(e+1, args.epochs, loss_total[0], (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0]), TP[0]/(TP[0]+FN[0]+0.01)))
        writer.add_scalar('loss', loss_total[0], e)
        writer.add_scalar('train_acc', (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0]), e)
        writer.add_scalar('train_recall', TP[0]/(TP[0]+FN[0]+0.01), e)

        train_model1.eval()
        TP, TN, FP, FN = [0], [0], [0], [0]
        start_time = time.time()
        for i in tqdm(range(len(test_dataloader))):
            with torch.no_grad():
                ptraj, esrc, edst, label, pre, shot_from, delay = test_dataloader[i]
                if shot_from is None:
                    continue

                if shot_from < 12:
                    cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    cands.remove(shot_from)
                else:
                    cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                    cands.remove(shot_from)
                q_from = [shot_from, random.choice(cands)]

                f1 = feature_extract(ptraj, q_from[0])
                f2 = feature_extract(ptraj, q_from[1])
                # f1 = feature_extract(ptraj, pass_from)
                # f2 = feature_extract(ptraj, pass_to)
                # f3 = feature_extract(ptraj, random.choice(cands))
                # f4 = feature_extract(ptraj, fake_from)
                # f5 = feature_extract(ptraj, fake_to)
                # x1 = torch.cat([f1, f2], dim=1)
                # x2 = torch.cat([f1, f3], dim=1)
                # x3 = torch.cat([f4, f5], dim=1)
                # x1 = torch.cat([ptraj[q_from[0], :, [0, 1]], ptraj[0, :, [0, 1]]]).reshape(-1)
                # x2 = torch.cat([ptraj[q_from[1], :, [0, 1]], ptraj[0, :, [0, 1]]]).reshape(-1)
                # x1 = torch.cat([ptraj[q_from[0], :, [0, 1]], ptraj[0, :, [0, 1]]], dim=1)
                # x2 = torch.cat([ptraj[q_from[1], :, [0, 1]], ptraj[0, :, [0, 1]]], dim=1)
                # pred1 = train_model1(torch.stack([x1, x2], dim=0))
                pred1 = train_model1(torch.stack([f1, f2], dim=0))

                pred_ = pred1.cpu().numpy()
                tp, tn, fp, fn = count_TFPN(pred_, [label, -1], 'svm')
                TP[0] += tp
                TN[0] += tn
                FP[0] += fp
                FN[0] += fn
        end_time = time.time()

        print("epoch:{0}/{1}, valid acc:{2:.3f}, valid recall:{3:.3f}, valid prec:{4:.3f}, inference time:{5:.4f}".format(e+1, args.epochs, (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0]), TP[0]/(TP[0]+FN[0]+0.01), TP[0]/(TP[0]+FP[0]+0.01), (end_time-start_time)/len(test_dataloader)))
        writer.add_scalar('valid_acc', (TP[0]+TN[0])/(TP[0]+TN[0]+FP[0]+FN[0]), e)
        writer.add_scalar('valid_recall', TP[0]/(TP[0]+FN[0]+0.01), e)
        writer.add_scalar('valid_prec', TP[0]/(TP[0]+FP[0]+0.01), e)
        # if (e+1) % args.save_interval == 0:
        #     torch.save(train_model, "{0}/pass_epoch_{1}.pkl".format(args.save_dir, e+1))

if __name__ == "__main__":
    train(args)