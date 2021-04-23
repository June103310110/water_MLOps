import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from utils import get_data


def sig_disturb(data2func, percent):
    percent = percent
    y = data2func.copy()
    np.random.seed(1)

    nums = int(len(y)/100*percent)
    if nums == 0:
        nums = 1
    rand = np.random.choice(range(len(y)), nums, replace=False)
    noise = []
    for i in y[rand]:
        a = np.random.normal(loc=i, scale=y.std())
        noise.append(a)
    noise = np.array(noise)
    noise *= 0.01
    y[rand] += noise
    return y, rand

def sig_transmittion_noise(data2func, percent):
    percent = percent
    y = data2func.copy()
    np.random.seed(1)

    nums = int(len(y)/100*percent)
    if nums == 0:
        nums = 1
    rand = np.random.choice(range(len(y)), nums, replace=False)
#     noise = np.ones(nums)
#     noise*=1000
    y[rand] = 1000
    
    return y, rand

def sig_drop(data2func, percent):
    
    y = data2func.copy()
    x_axis = np.array(range(len(label)))
    percent = percent
    np.random.seed(1)
    nums = int(len(y)/100*percent)
    if nums == 0:
        nums = 1
    print(nums)

    # drop element
    rand = np.random.choice(range(len(y)), nums, replace=False)
    drop_label = y.copy()
    drop_label[rand] = 0 

    return drop_label, rand
# 對一整個區間dropout 


def plt_drop(data2func, percent):
    
    y = data2func.copy()
    x_axis = np.array(range(len(label)))
    drop_label, rand = sig_drop(y, percent)
    # plot
    fig=plt.figure(figsize = (16, 5))

    fig_place = [1,2]

    subplot1 = fig.add_subplot(fig_place[0], fig_place[1], 1)
    plt.scatter(x_axis, y, marker='o')
    plt.scatter(x_axis[rand], drop_label[rand], marker='o', c = 'w')
    plt.title("origin signal of label", y=1)

    subplot2 = fig.add_subplot(fig_place[0], fig_place[1], 2)
    plt.title("origin signal of label with with sensitivity disturb")
    plt.scatter(x_axis, drop_label, marker='o')
    plt.scatter(x_axis[rand], drop_label[rand], marker='o', c = 'w')
