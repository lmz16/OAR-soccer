import cv2
import numpy as np
import h5py
import os
import shutil

fns = ['Match_2019_08_30_#001', 'Match_2019_09_09_#001', 'Match_2019_09_10_#001', 'Match_2019_09_11_#001', 'Match_2019_09_11_#002',
    'Match_2019_09_12_#001', 'Match_2019_09_12_#002', 'Match_2019_09_13_#001', 'Match_2019_09_13_#002', 'Match_2019_09_14_#001', 'Match_2019_09_14_#002',
    'Match_2019_09_15_#001', 'Match_2019_09_15_#002', 'Match_2019_09_16_#001', 'Match_2019_09_16_#002', 'Match_2019_09_16_#003']

starts = [2133, 4466, 154, 591, -394, 289, -493, -487, -525, -415, -415, 495, -893, -1861, 827, -215]

if __name__ == "__main__":
    delay = 1
    event = 'shot'
    train_f = h5py.File('train_{0}_delay_{1}_mode_0.hdf5'.format(event, delay), 'r')
    test_f = h5py.File('test_{0}_delay_{1}_mode_0.hdf5'.format(event, delay), 'r')

    for i in range(12):
        if os.path.exists('videos/{0}_{1}_{2}/'.format(event, delay, i)):
            shutil.rmtree('videos/{0}_{1}_{2}/'.format(event, delay, i))
        os.mkdir('videos/{0}_{1}_{2}/'.format(event, delay, i))
        imgs = np.array(train_f['{0}_imgs'.format(i)]).tolist()
        imgs.sort()
        cap = cv2.VideoCapture('videos/{0}.mkv'.format(fns[i]))
        success = True
        count = 0
        p = 0
        while success:
            success, frame = cap.read()
            count += 1
            while -20 < count*40 - starts[i] - imgs[p] <= 20 and success:
                cv2.imwrite('videos/{0}_{1}_{2}/{3}.jpg'.format(event, delay, i, imgs[p]), frame)
                p += 1
                if p == len(imgs):
                    success = False
                    break
                if p % 400 == 0:
                    print('{0}/16: {1}/{2}'.format(i, p, len(imgs)))
    
    for i in range(12, 16):
        if os.path.exists('videos/{0}_{1}_{2}/'.format(event, delay, i)):
            shutil.rmtree('videos/{0}_{1}_{2}/'.format(event, delay, i))
        os.mkdir('videos/{0}_{1}_{2}/'.format(event, delay, i))
        imgs = np.array(test_f['{0}_imgs'.format(i)]).tolist()
        imgs.sort()
        cap = cv2.VideoCapture('videos/{0}.mkv'.format(fns[i]))
        success = True
        count = 0
        p = 0
        while success:
            success, frame = cap.read()
            count += 1
            while -20 < count*40 - starts[i] - imgs[p] <= 20 and success:
                cv2.imwrite('videos/{0}_{1}_{2}/{3}.jpg'.format(event, delay, i, imgs[p]), frame)
                p += 1
                if p == len(imgs):
                    success = False
                    break
                if p % 400 == 0:
                    print('{0}/16: {1}/{2}'.format(i, p, len(imgs)))