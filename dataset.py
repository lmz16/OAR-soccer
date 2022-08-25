import json
import cv2
import torch
import numpy as np
import json
import h5py
import random

class DataLoader():
    def __init__(self, datapath, device=None, flip1=True, flip2=True):
        self.datapath = datapath
        self.dataset = h5py.File(datapath)
        self.keys = list(self.dataset.keys())
        random.shuffle(self.keys)
        self.device = device
        self.ind = -1
        self.flip1 = flip1
        self.flip2 = flip2
    
    def __len__(self):
        return len(self.dataset.keys())
    
    def __next__(self):
        if self.ind >= self.__len__():
            raise StopIteration
        else:
            self.ind += 1
            return self.__getitem__(self.ind)

    def __getitem__(self, ind):
        temp = self.dataset[self.keys[ind]]
        ptraj = np.array(temp['trajs'], np.float32)
        esrc = np.array(temp['srcs'], np.int32)
        edst = np.array(temp['dsts'], np.int32)
        scalar = temp['scalar']
        if self.flip1 and np.random.randint(2) == 1:
            ptraj[:, :, 1] = -ptraj[:, :, 1]
        if self.flip2 and np.random.randint(2) == 1:
            ptraj[:, :, 0] = - ptraj[:, :, 0]
            ptraj[:, :, 3] = (3 - ptraj[:, :, 3])*(ptraj[:, :, 3] > 0)
        pass_from, pass_to, shot_from, ballid, eventtype = int(scalar[0]), int(scalar[1]), int(scalar[2]), int(scalar[3]), int(scalar[-1])
        if self.device:
            ptraj = torch.from_numpy(ptraj).to(self.device)
        return ptraj, esrc, edst, (pass_from, pass_to, shot_from, ballid, eventtype), len(temp['pids'])

class DataLoader_S:
    def __init__(self, datapath, device=None, flip1=True, flip2=True, ignore=0):
        self.datapath = datapath
        self.dataset = h5py.File(datapath)
        self.keys = list(self.dataset.keys())
        random.shuffle(self.keys)
        self.device = device
        self.ind = -1
        self.flip1 = flip1
        self.flip2 = flip2
        self.ignore = ignore
    
    def __len__(self):
        return len(self.dataset.keys())
    
    def __next__(self):
        if self.ind >= self.__len__():
            raise StopIteration
        else:
            self.ind += 1
            return self.__getitem__(self.ind)

    def __getitem__(self, ind):
        if self.keys[ind][0] == 'p' or self.keys[ind][0] == 's':
            temp = self.dataset[self.keys[ind]]
            ptraj = np.array(temp['trajs'], np.float32)
            esrc = np.array(temp['srcs'], np.int32)
            edst = np.array(temp['dsts'], np.int32)
            scalar = temp['scalar']

            ptraj[:, :, 0] = ptraj[:, :, 0]/55.0 - 1.0
            ptraj[:, :, 1] = ptraj[:, :, 1]/36.0 - 1.0
            if self.flip1 and np.random.randint(2) == 1:
                ptraj[:, :, 1] = -ptraj[:, :, 1]
            if self.flip2 and np.random.randint(2) == 1:
                ptraj[:, :, 0] = - ptraj[:, :, 0]
                ptraj[:, :, 2] = (3 - ptraj[:, :, 2])*(ptraj[:, :, 2] > 0)
            
            pass_from, pass_to, shot_from, delay = int(scalar[0]), int(scalar[1]), int(scalar[2]), int(scalar[3])
            if self.device:
                ptraj = torch.from_numpy(ptraj).to(self.device)
            return ptraj, esrc, edst, pass_from, pass_to, shot_from, delay
        else:
            return None, None, None, None, None, None, None

class DataLoader_V:
    def __init__(self, datapath, img_dir, device=None, flip=True):
        self.datapath = datapath
        self.dataset = h5py.File(datapath)
        self.keys = list(self.dataset.keys())
        random.shuffle(self.keys)
        self.device = device
        self.ind = -1
        self.flip = flip
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.dataset.keys())
    
    def __next__(self):
        if self.ind >= self.__len__():
            raise StopIteration
        else:
            self.ind += 1
            return self.__getitem__(self.ind)

    def __getitem__(self, ind):
        if self.keys[ind][0] == 'p' or self.keys[ind][0] == 's':
            temp = self.dataset[self.keys[ind]]
            crop = cv2.imread("{0}/{1}.jpg".format(self.img_dir, self.keys[ind]))
            imgs = [[], [], [], []]

            ptraj = np.array(temp['trajs'], np.float32)
            esrc = np.array(temp['srcs'], np.int32)
            edst = np.array(temp['dsts'], np.int32)
            players = temp['players']
            scalar = temp['scalar']

            ptraj[:, :, 0] = ptraj[:, :, 0]/55.0 - 1.0
            ptraj[:, :, 1] = ptraj[:, :, 1]/36.0 - 1.0
            if self.flip and np.random.randint(2) == 1:
                ptraj[:, :, 0] = - ptraj[:, :, 0]
                ptraj[:, :, 2] = (3 - ptraj[:, :, 2])*(ptraj[:, :, 2] > 0)
                for i in range(4):
                    for j in range(8):
                        imgs[i].append(np.flip(crop[i*96:(i+1)*96, j*96:(j+1)*96, :], axis=1))
            else:
                for i in range(4):
                    for j in range(8):
                        imgs[i].append(crop[i*96:(i+1)*96, j*96:(j+1)*96, :])
            imgs = np.array(imgs).astype(np.float32) / 255
            if len(players) == 4:
                true_from, true_to, fake_from, fake_to = int(players[0]), int(players[1]), int(players[2]), int(players[3])
            elif len(players) == 2:
                true_from, true_to, fake_from, fake_to = int(players[0]), int(scalar[0]), int(players[1]), -1
            if self.device:
                ptraj = torch.from_numpy(ptraj).to(self.device)
                imgs = torch.from_numpy(imgs).to(self.device)
            return ptraj, esrc, edst, true_from, true_to, fake_from, fake_to, imgs
        else:
            return None, None, None, None, None, None, None, None

    