import numpy as np
import csv
import pandas as pd
import scipy.io
from sklearn import preprocessing
from evaluation_metrics import *


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1,keepdims=True)
    s = x_exp / x_sum
    return s

def predict(weights, b, X_test, Y_test):
    pre_label = X_test.dot(weights)+b
    pre_value = softmax(pre_label)

    Cheb = chebyshev(Y_test + 10 ** -6., pre_value + 10 ** -6)
    Clark = clark(Y_test + 10 ** -6, pre_value + 10 ** -6)
    Can = canberra(Y_test + 10 ** -6, pre_value + 10 ** -6)
    Kl = kl(Y_test + 10 ** -6, pre_value + 10 ** -6)
    Cos = cosine(Y_test + 10 ** -6, pre_value + 10 ** -6)
    Intersec = intersection(Y_test + 10 ** -6, pre_value + 10 ** -6)

    return [Cheb, Clark, Can, Kl, Cos,Intersec], pre_value

    
