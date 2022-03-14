import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmode(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    X = np.matrix(X);y = np.matrix(y);theta = np.matrix(theta)
    first = np.multiply(-y, np.log(sigmode(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmode(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):
    X = np.matrix(X);y = np.matrix(y);theta = np.matrix(theta)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmode(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

def data_show(result,*parameters):
    x1 = np.linspace(30,100,100)
    fun = (-result[0][0] - result[0][1] * x1) / result[0][2]   # 不理解
    plt.figure(figsize=(15,7), dpi = 80)
    plt.plot(x1, fun, c = "y", label = "Prediction");plt.xlabel("Exam_1");plt.ylabel("Exam_2")
    plt.scatter(parameters[0]["Exam_1"],parameters[0]["Exam_2"],s = 50, c="b", marker = "o", label ="Admitted")
    plt.scatter(parameters[1]["Exam_1"],parameters[1]["Exam_2"],s = 50, c= "r",marker = "x", label = "Not Admitted")
    plt.legend();plt.show()

def data_operation(data):
    data.insert(0, "Ones", 1)
    X = np.array(data.iloc[:,:-1].values)
    y = np.array(data.iloc[:,data.shape[1] - 1 :].values)
    theta = np.zeros(3)
    result = opt.fmin_tnc(func=cost, x0 = theta, fprime = gradient, args = (X,y))
    return result

if __name__ == "__main__":
    #数据读取
    data = pd.read_csv("ex2data2.csv",header=None,names=["Exam_1","Exam_2","Admitted"])
    positive = data[data["Admitted"].isin([1])]
    negative = data[data["Admitted"].isin([0])]
    #数据学习
    result = data_operation(data)
    #数据展示
    data_show(result,positive,negative)
    