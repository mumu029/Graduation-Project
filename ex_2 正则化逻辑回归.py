import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

#逻辑函数
def sigmode(z):
    return 1 / (1 + np.exp(-z))

#数据展示
def data_show(result,*parameters):
    plt.figure(figsize = (15,7), dpi = 80)
    plt.xlabel("Test_1");plt.ylabel("Test_2")
    plt.scatter(parameters[0]["Test_1"],parameters[0]["Test_2"],s = 50,c = 'b', marker = 'o', label = "Accept")
    plt.scatter(parameters[1]["Test_1"],parameters[1]["Test_2"],s = 50,c = "r", marker = "x", label = "Reject")
    x,y = find_decision_boundary(result)
    plt.scatter(x,y,c='y',s = 10,label = "Prediction")
    plt.legend();plt.show()

#代价函数
def costReg(theta,X,y,learningRate):
    X = np.matrix(X);y = np.matrix(y);theta = np.matrix(theta)
    first = np.multiply(-y, np.log(sigmode(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmode(X * theta.T)))
    reg = (learningRate * np.sum(np.power(theta[:,1:theta.shape[1]],2))) / (2 * len(X)) 
    return (np.sum(first - second) / len(X)) + reg

#计算梯度
def gradientReg(theta,X,y,learningRate):
    X = np.matrix(X);y = np.matrix(y);theta = np.matrix(theta)
    parameters = int(theta.shape[1])
    grad = np.zeros(parameters)

    error = sigmode(X * theta.T) - y
    for i in range(parameters):
        flag = np.multiply(error, X[:,i])
        if i == 0:
            grad[i] = np.sum(flag) / len(X)
        else:
            grad[i] = (np.sum(flag) / len(X)) + ((learningRate * theta[:,i]) / len(X))
    return grad

#决策曲线
def hfunc2(theta,x1,x2):
    temp = theta[0][0]
    place = 0
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            temp += np.power(x1, i - j) * np.power(x2,j) * theta[0][place + 1]
            place += 1
    return temp
def find_decision_boundary(theta):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({"x1" : x_cord, "x2" : y_cord})
    h_val['hval'] = hfunc2(theta,h_val["x1"],h_val["x2"])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2

#数据读取
data_init = pd.read_csv("data.csv", header= None, names=["Test_1","Test_2","Accepted"])
positive = data_init[data_init["Accepted"].isin([1])]
negative = data_init[data_init["Accepted"].isin([0])]

#特征映射
degree = 6
data2 = data_init
x1 = data2["Test_1"]
x2 = data2["Test_2"]

data2.insert(3,"Ones",1)

for i in range(1,degree + 1):
    for j in range(0, i + 1):
        data2["F" + str(i -j) + str(j)] = np.power(x1,i - j) * np.power(x2,j)
data2.drop("Test_1", axis = 1, inplace = True)
data2.drop("Test_2", axis = 1, inplace = True)
#数据处理
cols = data2.shape[1]
x2 = data2.iloc[:,1:cols].to_numpy()
y2 = data2.iloc[:,0:1].to_numpy()
theta = np.zeros(cols - 1)
learningRate = 1
result = opt.fmin_tnc(func = costReg,x0=theta,fprime=gradientReg,args=(x2,y2,learningRate))
data_show(result,positive,negative)
