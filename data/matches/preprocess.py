import json
import cv2
import numpy as np
import collections
import h5py
import random

DATALENGTH = 25
WIDTH = 128
HEIGHT = 256
GROUND_LENGTH = 106
GROUND_WIDTH = 70.4
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def homography_trans(x, y, H):
    v = np.array([[x/GROUND_LENGTH], [-y/GROUND_WIDTH], [1]])
    v_ = np.matmul(H, v)
    return IMAGE_WIDTH*(v_[0]/(v_[2]+1e-8)+0.5), IMAGE_HEIGHT*(v_[1]/(v_[2]+1e-8)+0.5)

def shotp_process_h5(h5f, dir_name, split, scale=0.02, b=0):
    with open("{0}/shot.json".format(dir_name), "r") as f:
        sinfo = json.load(f)
    with open("{}/match_data.json".format(dir_name), "r") as f:
        info = json.load(f)
        team_info = {}
        for p in info['players']:
            team_info[p['trackable_object']] = p['team_id']
        ball_trackid = info['ball']['trackable_object']
    with open("{}/structured_data.json".format(dir_name), "r") as f:
        data = json.load(f)
    events = sinfo[split]

    for i in range(len(events)):
        p_duration = collections.defaultdict(int)
        # endtime = max(events[i][1]-b+2, DATALENGTH+1)
        endtime = max(events[i][1]-random.randint(5, 15), DATALENGTH+1)
        p_traj = collections.defaultdict(lambda: [[-1000, -1000, 0, 0] for _ in range(DATALENGTH)])
        for j in range(endtime-DATALENGTH, endtime):
            frame = data[sinfo[split+"_start"]+2*j]
            for p in frame['data']:
                if 'trackable_object' in p:
                    if p['trackable_object'] in team_info:
                        p_duration[p['trackable_object']] += 1
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
                        if p['trackable_object'] == frame['possession']['trackable_object']:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][2] = 1
                        if (split == "first")^(team_info[p['trackable_object']] == sinfo['right_team']):
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 1
                        else:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 2
                    elif p['trackable_object'] == ball_trackid:
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
        p_duration[ball_trackid] = DATALENGTH
        pids, trajs, types, srcs, dsts = [], [], [], [], []
        shot_from, ball_id = -1, -1
        starttime = endtime-DATALENGTH
        for pid in p_duration.keys():
            if abs(p_duration[pid]) > 5:
                pids.append(pid)
                trajs.append(p_traj[pid])
                if pid == ball_trackid:
                    ball_id = len(pids)-1
                types.append(p_traj[pid][0][3])
                if pid == events[i][0]:
                    shot_from = len(pids)-1
        if len(pids) > 5 and shot_from != -1:
            edge_cands = []
            try:
                for p1 in range(len(pids)):
                    if p1 == ball_id:
                        continue
                    dist = []
                    for p2 in range(len(pids)):
                        if p1 != p2 and p2 != ball_id:
                            dist.append([(trajs[p1][-1][0]-trajs[p2][-1][0])**2+(trajs[p1][-1][1]-trajs[p2][-1][1])**2, p2])
                    dist.sort()
                    for n in range(4):
                        edge_cands.append([dist[n][1], p1])
                for p2 in range(len(pids)):
                    if ball_id != p2 and ball_id != -1:
                        edge_cands.append([ball_id, p2])
            except IndexError:
                print("IndexError")
                continue
            for src, dst in edge_cands:
                srcs.append(src)
                dsts.append(dst)
            
            dset = h5f.create_group("shotp_{0}_{1}_{2}_{3}".format(dir_name, split, i, b))
            dset["pids"], dset["trajs"], dset["srcs"], dset["dsts"] = pids, trajs, srcs, dsts
            dset["scalar"] = [1, 1, shot_from, ball_id, starttime, endtime]


def shotn_process_h5(h5f, dir_name, split, scale=0.02, b=0):
    with open("{0}/shot.json".format(dir_name), "r") as f:
        sinfo = json.load(f)
    with open("{}/match_data.json".format(dir_name), "r") as f:
        info = json.load(f)
        team_info = {}
        for p in info['players']:
            team_info[p['trackable_object']] = p['team_id']
        ball_trackid = info['ball']['trackable_object']
    with open("{}/structured_data.json".format(dir_name), "r") as f:
        data = json.load(f)
    with open("{0}/shotn_{1}.json".format(dir_name, split), "r") as f:
        events = json.load(f)

    for i in range(len(events)):
        p_duration = collections.defaultdict(int)
        # endtime = max(events[i][1]-b+2, DATALENGTH+1)
        endtime = max(events[i][1]-random.randint(5, 15), DATALENGTH+1)
        p_traj = collections.defaultdict(lambda: [[-1000, -1000, 0, 0] for _ in range(DATALENGTH)])
        for j in range(endtime-DATALENGTH, endtime):
            frame = data[sinfo[split+"_start"]+2*j]
            for p in frame['data']:
                if 'trackable_object' in p:
                    if p['trackable_object'] in team_info:
                        p_duration[p['trackable_object']] += 1
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
                        if p['trackable_object'] == frame['possession']['trackable_object']:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][2] = 1
                        if (split == "first")^(team_info[p['trackable_object']] == sinfo['right_team']):
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 1
                        else:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 2
                    elif p['trackable_object'] == ball_trackid:
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
        p_duration[ball_trackid] = DATALENGTH
        pids, trajs, types, srcs, dsts = [], [], [], [], []
        shot_from, ball_id = -1, -1
        starttime = endtime-DATALENGTH
        for pid in p_duration.keys():
            if abs(p_duration[pid]) > 5:
                pids.append(pid)
                trajs.append(p_traj[pid])
                if pid == ball_trackid:
                    ball_id = len(pids)-1
                types.append(p_traj[pid][0][3])
                if pid == events[i][0]:
                    shot_from = len(pids)-1
        if len(pids) > 5 and shot_from != -1:
            edge_cands = []
            try:
                for p1 in range(len(pids)):
                    if p1 == ball_id:
                        continue
                    dist = []
                    for p2 in range(len(pids)):
                        if p1 != p2 and p2 != ball_id:
                            dist.append([(trajs[p1][-1][0]-trajs[p2][-1][0])**2+(trajs[p1][-1][1]-trajs[p2][-1][1])**2, p2])
                    dist.sort()
                    for n in range(4):
                        edge_cands.append([dist[n][1], p1])
                for p2 in range(len(pids)):
                    if ball_id != p2 and ball_id != -1:
                        edge_cands.append([ball_id, p2])
            except IndexError:
                print("IndexError")
                continue
            for src, dst in edge_cands:
                srcs.append(src)
                dsts.append(dst)
            
            dset = h5f.create_group("shotn_{0}_{1}_{2}_{3}".format(dir_name, split, i, b))
            dset["pids"], dset["trajs"], dset["srcs"], dset["dsts"] = pids, trajs, srcs, dsts
            dset["scalar"] = [-1, -1, shot_from, ball_id, starttime, endtime]


def pass_process_h5(h5f, dir_name, split, scale=0.02, b=0):
    with open("{0}/shot.json".format(dir_name), "r") as f:
        sinfo = json.load(f)
    with open("{}/match_data.json".format(dir_name), "r") as f:
        info = json.load(f)
        team_info = {}
        for p in info['players']:
            team_info[p['trackable_object']] = p['team_id']
        ball_trackid = info['ball']['trackable_object']
    with open("{}/structured_data.json".format(dir_name), "r") as f:
        data = json.load(f)
    with open("{0}/pass_{1}.json".format(dir_name, split), "r") as f:
        events = json.load(f)
        # print(len(events))

    for i in range(len(events)):
        p_duration = collections.defaultdict(int)
        # endtime = max(events[i][3]-random.randint(5, 15), DATALENGTH+1) #pred
        endtime = max(events[i][3]-b+2, DATALENGTH+1)
        p_traj = collections.defaultdict(lambda: [[-1000, -1000, 0, 0] for _ in range(DATALENGTH)])
        for j in range(endtime-DATALENGTH, endtime):
            frame = data[sinfo[split+"_start"]+2*j]
            for p in frame['data']:
                if 'trackable_object' in p:
                    if p['trackable_object'] in team_info:
                        p_duration[p['trackable_object']] += 1
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
                        if p['trackable_object'] == frame['possession']['trackable_object']:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][2] = 1
                        if (split == "first")^(team_info[p['trackable_object']] == sinfo['right_team']):
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 1
                        else:
                            p_traj[p['trackable_object']][j-endtime+DATALENGTH][3] = 2
                    elif p['trackable_object'] == ball_trackid:
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][0] = p['x']*scale
                        p_traj[p['trackable_object']][j-endtime+DATALENGTH][1] = p['y']*scale
        p_duration[ball_trackid] = DATALENGTH
        pids, trajs, srcs, dsts = [], [], [], []
        pass_from, pass_to, shot_from, ball_id = -1, -1, -1, -1
        starttime = endtime-DATALENGTH
        for pid in p_duration.keys():
            if abs(p_duration[pid]) > 5:
                pids.append(pid)
                trajs.append(p_traj[pid])
                if pid == ball_trackid:
                    ball_id = len(pids)-1
                if pid == events[i][0]:
                    pass_from = len(pids)-1
                elif pid == events[i][1]:
                    pass_to = len(pids)-1
        if len(pids) > 5 and pass_from != -1 and pass_to != -1:
            edge_cands = []
            for p1 in range(len(pids)):
                if p1 == ball_id:
                    continue
                dist = []
                for p2 in range(len(pids)):
                    if p1 != p2 and p2 != ball_id:
                        dist.append([(trajs[p1][-1][0]-trajs[p2][-1][0])**2+(trajs[p1][-1][1]-trajs[p2][-1][1])**2, p2])
                dist.sort()
                for n in range(4):
                    edge_cands.append([dist[n][1], p1])
            for p2 in range(len(pids)):
                if ball_id != p2 and ball_id != -1:
                    edge_cands.append([ball_id, p2])
            for src, dst in edge_cands:
                srcs.append(src)
                dsts.append(dst)
            try:
                dset = h5f.create_group("pass_{0}_{1}_{2}_{3}".format(dir_name, split, i, b))
                dset["pids"], dset["trajs"], dset["srcs"], dset["dsts"] = pids, trajs, srcs, dsts
                dset["scalar"] = [pass_from, pass_to, shot_from, ball_id, starttime, endtime]
            except ValueError:
                print("pass_{0}_{1}_{2}".format(dir_name, split, i))

if __name__ == "__main__":
    train_f = h5py.File("train_pred_shot.hdf5","w")
    test_f = h5py.File("test_pred_shot.hdf5","w")
    for b in range(5):
    # train_f = None
        for fn in ["2068", "2269", "2417", "2440", "2841", "3442", "3518"]:
            # pass_process_h5(train_f, fn, "first", b=b)
            # pass_process_h5(train_f, fn, "second", b=b)
            shotp_process_h5(train_f, fn, "first", b=b)
            shotp_process_h5(train_f, fn, "second", b=b)
            shotn_process_h5(train_f, fn, "first", b=b)
            shotn_process_h5(train_f, fn, "second", b=b)
    
        for fn in ["4039", "3749"]:
            # pass_process_h5(test_f, fn, "first", b=b)
            # pass_process_h5(test_f, fn, "second", b=b)
            shotp_process_h5(test_f, fn, "first", b=b)
            shotp_process_h5(test_f, fn, "second", b=b)
            shotn_process_h5(test_f, fn, "first", b=b)
            shotn_process_h5(test_f, fn, "second", b=b)
