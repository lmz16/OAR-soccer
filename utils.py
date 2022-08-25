def count_TFPN(pred, target, mode='nn'):
    TP, TN, FP, FN = 0, 0, 0, 0
    thres = 0 if mode == 'svm' else 0.5
    for i in range(len(target)):
        if pred[i] > thres:
            if target[i] > thres:
                TP += 1
            else:
                FP += 1
        else:
            if target[i] > thres:
                FN += 1
            else:
                TN += 1
    return TP, TN, FP, FN

