import torch
import numpy as np
import csv
import scipy.io
from sklearn import preprocessing
from PenaltyMethod import EPFM
from predict import *
from sklearn.preprocessing import Normalizer

if __name__ == '__main__':
    data = scipy.io.loadmat(r'datasets/Yeast_spoem.mat')
    data1 = scipy.io.loadmat(r'result_mll/Yeast_spoem_mll.mat')
    features = data['features']
    pri_features = data1['distribution']
    labels = data['labels']
    y_log = data['y_log']

    features = preprocessing.scale(features)
    pri_features = preprocessing.scale(pri_features)

    samples_num = len(features)
    features_dim = len(features[0])
    pri_dim = len(pri_features[0])
    labels_dim = len(labels[0])

    lam1 = 0.1
    lam2 = 0.1
    # evaluate_amounts = 5 #评价指标个数

    X1 = torch.from_numpy(features).float()  # data
    X2 = torch.from_numpy(pri_features).float()  # privilege data
    Y = torch.from_numpy(y_log).float()

    W_1 = torch.zeros((features_dim, labels_dim), requires_grad=True)
    W_2 = torch.zeros((pri_dim, labels_dim), requires_grad=True)
    W_3 = torch.zeros((pri_dim, labels_dim), requires_grad=True)
    b_1 = torch.zeros((samples_num, 1), requires_grad=True)
    b_2 = torch.zeros((samples_num, 1), requires_grad=True)
    b_3 = torch.zeros((samples_num, 1), requires_grad=True)

    i_loss =  torch.ones((samples_num, labels_dim)) * .0001


    W = [W_1, W_2,W_3, b_1, b_2,b_3]


    def f(W):
        a = torch.sum(torch.norm(W[0], dim=0)) + lam1 * (torch.sum(torch.norm(W[1], dim=0)) + torch.sum(torch.norm(W[2], dim=0)))
        b = torch.sum(X2.mm(W[1]) + W[4]) + torch.sum(X2.mm(W[2]) + W[5])
        return 1 / 2 * a + lam2 * b


    def ieq(W):
        ieq1 =(i_loss + (X2.mm(W[1]) + W[4]) - Y + (X1.mm(W[0]) + W[3])).reshape(-1, 1)
        ieq2 = (i_loss + (X2.mm(W[2]) + W[5]) + Y - (X1.mm(W[0]) + W[3])).reshape(-1, 1)
        ieq3 = (X2.mm(W[2]) + W[5]).reshape(-1, 1)
        ieq4 = (X2.mm(W[1]) + W[4]).reshape(-1, 1)

        return torch.cat([ieq1, ieq2,ieq3,ieq4], dim=0)
        return torch.cat([ieq1, ieq2,ieq3,ieq4], dim=0)
    cons = {
        'ieq': [
            ieq
        ]
    }

    opter = EPFM(f, W, cons, Epoches=4, epoches=5000, is_plot=True, record=False, fast_mode=True)
    opter.optimize()
    weights = opter.parameters[0].detach().numpy()
    b = opter.parameters[3].detach().numpy()
    res_once, pre_value = predict(weights, b, features, labels)
    print("Cheb               Clark                Can                 Kl                 Cos               Intersec")
    print(res_once)