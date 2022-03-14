import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report

def sigmode(z):
    return 1 / (1 + np.exp(-z))

def cost(theta,X,y,learningRate):
    X = np.matrix(X);y = np.matrix(y);theta = np.matrix(theta)
    first = np.multiply(-y, np.log(sigmode(X * theta.T)))
    second = np.multiply(1 - y,np.log(1 - sigmode(X * theta.T)))
    reg = (learningRate * np.sum(np.power(theta[:,1:theta.shape[1]],2))) / (2 * len(X))
    return (np.sum(first - second) / len(X)) + reg



data = loadmat("ex3data1.mat")
sample_idx = np.random.choice(np.array(data["X"].shape[0]),100)
sample_images = data["X"][sample_idx,:]
fig, ax_array = plt.subplots(nrows=10,ncols=10,sharex=True,sharey=True,figsize = (15,15))
for r in range(10):
    for c in range(10):
        ax_array[r,c].matshow(np.array(sample_images[10 * r + c].reshape(20,20)).T,cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()