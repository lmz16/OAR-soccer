import h5py
import cv2
import numpy as np
import random

def train_pass_process(delay=0):
    h5f = h5py.File('train_pass_delay_{}_mode_0.hdf5'.format(delay), 'r+')
    img_lists = []
    for i in range(12):
        img_lists.append(np.array(h5f['{}_imgs'.format(i)]))
    img_lists[10] = []
    for en in h5f.keys():
        if en[0] == 'p':
            canvas = np.zeros([384, 768, 3], dtype='uint8')
            gameid = int(en.split('_')[1])
            event = h5f[en]
            scalar = np.array(event['scalar'])
            img_inds = np.array(event['img_trajs'])[0, 4::5, 2]
            imgs = []
            for img_ind in img_inds:
                flag = False
                for offset in [0, 1, -1, 2, -2]:
                    if int(img_ind + offset) in img_lists[gameid]:
                        img = cv2.imread('videos/pass_{0}_{1}/{2}.jpg'.format(delay, gameid, int(img_ind + offset)))
                        if img is not None:
                            imgs.append(img)
                            flag = True
                            break
                if not flag:
                    imgs.append(np.zeros([1080, 1920, 3], dtype='uint8'))

            pass_from, pass_to = scalar[0], scalar[1]
            if pass_from < 12:
                cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                cands.remove(pass_from)
                cands.remove(pass_to)
            else:
                cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                cands.remove(pass_from)
                cands.remove(pass_to)
            fake_from, fake_to = random.sample(cands, 2)
            h5f['{0}/players'.format(en)] = [pass_from, pass_to, fake_from, fake_to]
            img_trajs = np.array(event['img_trajs'])[[pass_from, pass_to, fake_from, fake_to], 4::5, :2]
            for i in range(8):
                for j in range(4):
                    x, y = int(img_trajs[j, i, 0]), int(img_trajs[j, i, 1])
                    if 48 < x < 1920-48 and 1080 > y > 96:
                        canvas[j*96:(j+1)*96, i*96:(i+1)*96, :] = imgs[i][y-96:y, x-48:x+48, :]
            cv2.imwrite('crops/pass_{0}/{1}.jpg'.format(delay, en), canvas)


def test_pass_process(delay=0):
    h5f = h5py.File('test_pass_delay_{}_mode_0.hdf5'.format(delay), 'r+')
    img_lists = []
    for i in range(12, 16):
        img_lists.append(np.array(h5f['{}_imgs'.format(i)]))
    for en in h5f.keys():
        if en[0] == 'p':
            canvas = np.zeros([384, 768, 3], dtype='uint8')
            gameid = int(en.split('_')[1])
            event = h5f[en]
            scalar = np.array(event['scalar'])
            img_inds = np.array(event['img_trajs'])[0, 4::5, 2]
            imgs = []
            for img_ind in img_inds:
                flag = False
                for offset in [0, 1, -1, 2, -2]:
                    if int(img_ind + offset) in img_lists[gameid-12]:
                        img = cv2.imread('videos/pass_{0}_{1}/{2}.jpg'.format(delay, gameid, int(img_ind + offset)))
                        if img is not None:
                            imgs.append(img)
                            flag = True
                            break
                if not flag:
                    imgs.append(np.zeros([1080, 1920, 3], dtype='uint8'))

            pass_from, pass_to = scalar[0], scalar[1]
            if pass_from < 12:
                cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                cands.remove(pass_from)
                cands.remove(pass_to)
            else:
                cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                cands.remove(pass_from)
                cands.remove(pass_to)
            fake_from, fake_to = random.sample(cands, 2)
            h5f['{0}/players'.format(en)] = [pass_from, pass_to, fake_from, fake_to]
            img_trajs = np.array(event['img_trajs'])[[pass_from, pass_to, fake_from, fake_to], 4::5, :2]
            for i in range(8):
                for j in range(4):
                    x, y = int(img_trajs[j, i, 0]), int(img_trajs[j, i, 1])
                    if 48 < x < 1920-48 and 1080 > y > 96:
                        canvas[j*96:(j+1)*96, i*96:(i+1)*96, :] = imgs[i][y-96:y, x-48:x+48, :]
            cv2.imwrite('crops/pass_{0}/{1}.jpg'.format(delay, en), canvas)

def train_shot_process(delay=0):
    h5f = h5py.File('train_shot_delay_{}_mode_0.hdf5'.format(delay), 'r+')
    img_lists = []
    for i in range(12):
        img_lists.append(np.array(h5f['{}_imgs'.format(i)]))
    img_lists[10] = []
    for en in h5f.keys():
        if en[0] == 's':
            canvas = np.zeros([384, 768, 3], dtype='uint8')
            gameid = int(en.split('_')[1])
            event = h5f[en]
            scalar = np.array(event['scalar'])
            img_inds = np.array(event['img_trajs'])[0, 4::5, 2]
            imgs = []
            for img_ind in img_inds:
                flag = False
                for offset in [0, 1, -1, 2, -2]:
                    if int(img_ind + offset) in img_lists[gameid]:
                        img = cv2.imread('videos/shot_{0}_{1}/{2}.jpg'.format(delay, gameid, int(img_ind + offset)))
                        if img is not None:
                            imgs.append(img)
                            flag = True
                            break
                if not flag:
                    imgs.append(np.zeros([1080, 1920, 3], dtype='uint8'))

            shot_from = scalar[2]
            if shot_from < 12:
                cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                cands.remove(shot_from)
            else:
                cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                cands.remove(shot_from)
            fake_from = random.choice(cands)
            h5f['{0}/players'.format(en)] = [shot_from, fake_from]
            img_trajs = np.array(event['img_trajs'])[[shot_from, fake_from], 4::5, :2]
            for i in range(8):
                for j in range(2):
                    x, y = int(img_trajs[j, i, 0]), int(img_trajs[j, i, 1])
                    if 48 < x < 1920-48 and 1080 > y > 96:
                        canvas[j*96:(j+1)*96, i*96:(i+1)*96, :] = imgs[i][y-96:y, x-48:x+48, :]
            cv2.imwrite('crops/shot_{0}/{1}.jpg'.format(delay, en), canvas)

def test_shot_process(delay=0):
    h5f = h5py.File('test_shot_delay_{}_mode_0.hdf5'.format(delay), 'r+')
    img_lists = []
    for i in range(12, 16):
        img_lists.append(np.array(h5f['{}_imgs'.format(i)]))
    for en in h5f.keys():
        if en[0] == 's':
            canvas = np.zeros([384, 768, 3], dtype='uint8')
            gameid = int(en.split('_')[1])
            event = h5f[en]
            scalar = np.array(event['scalar'])
            img_inds = np.array(event['img_trajs'])[0, 4::5, 2]
            imgs = []
            for img_ind in img_inds:
                flag = False
                for offset in [0, 1, -1, 2, -2]:
                    if int(img_ind + offset) in img_lists[gameid-12]:
                        img = cv2.imread('videos/shot_{0}_{1}/{2}.jpg'.format(delay, gameid, int(img_ind + offset)))
                        if img is not None:
                            imgs.append(img)
                            flag = True
                            break
                if not flag:
                    imgs.append(np.zeros([1080, 1920, 3], dtype='uint8'))

            shot_from = scalar[2]
            if shot_from < 12:
                cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                cands.remove(shot_from)
            else:
                cands = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
                cands.remove(shot_from)
            fake_from = random.choice(cands)
            h5f['{0}/players'.format(en)] = [shot_from, fake_from]
            img_trajs = np.array(event['img_trajs'])[[shot_from, fake_from], 4::5, :2]
            for i in range(8):
                for j in range(2):
                    x, y = int(img_trajs[j, i, 0]), int(img_trajs[j, i, 1])
                    if 48 < x < 1920-48 and 1080 > y > 96:
                        canvas[j*96:(j+1)*96, i*96:(i+1)*96, :] = imgs[i][y-96:y, x-48:x+48, :]
            cv2.imwrite('crops/shot_{0}/{1}.jpg'.format(delay, en), canvas)

if __name__ == "__main__":
    # train_pass_process(delay=2)
    # test_pass_process(delay=2)
    test_shot_process(delay=0)
    test_shot_process(delay=1)
