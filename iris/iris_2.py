'''
向量化实现逻辑回归
使用pca包实现降维（3D)
'''
import numpy as np
import pandas as pd
import random as rd
import time
import matplotlib.pyplot as plt
import sklearn.decomposition as sd
from mpl_toolkits.mplot3d import Axes3D


def load_data(path):
    '''
    加载数据
    '''
    global num_train
    Data_All = pd.read_csv(path,delimiter=',',header=None)
    Data_All_arr = np.array(Data_All)
    #two class
    np.random.shuffle(Data_All_arr)
    positive = Data_All_arr[np.where(Data_All_arr[:,4]=='Iris-setosa')]
    negetive = Data_All_arr[np.where(Data_All_arr[:,4]=='Iris-versicolor')]
    negetive[:,4] = 0#打上数字标记
    positive[:,4] = 1
    #train sample
    positive_tr = positive[0:num_train//2]
    negetive_tr = negetive[0:num_train//2]
    #test sample
    positive_te = positive[num_train//2:50]
    negetive_te = negetive[num_train//2:50]
    return positive_tr,negetive_tr,positive_te,negetive_te

def train_gen(positive_tr,negetive_tr):
    '''
    生成训练集
    '''
    global num_train
    #属性
    X_train_pos = positive_tr.T[0:4]
    X_train_neg = negetive_tr.T[0:4]
    X_train = np.hstack((X_train_neg,X_train_pos))#水平把数组拼接起来
    #分离标签
    y_train_pos = positive_tr.T[4]
    y_train_neg = negetive_tr.T[4]
    y_train = np.hstack((y_train_neg,y_train_pos))
    y_train = y_train.reshape(1,num_train)
    return X_train,y_train

def test_gen(positive_te,negetive_te):
    '''
    生成测试集
    '''
    global num_test
    #属性
    X_test_pos = positive_te.T[0:4]
    X_test_neg = negetive_te.T[0:4]
    X_test = np.hstack((X_test_neg,X_test_pos))#水平把数组拼接起来
    #分离标签
    y_test_pos = positive_te.T[4]
    y_test_neg = negetive_te.T[4]
    y_test = np.hstack((y_test_neg,y_test_pos))
    y_test = y_test.reshape(1,num_test)
    return X_test,y_test

def sigmoid(W,X,b):
    '''
    sigmoid 函数
    '''
    #expression
    z = np.dot(W.T,X)+b
    z = z.astype(float)#如果不进行类型转换会发生异常
    return 1/(1+np.exp(-z))

def cost(y_train,sig):
    '''
    损失函数
    '''
    return np.sum(-y_train*np.log(sig)-(1-y_train)*np.log(1-sig))

def initial_params():
    '''
    参数初始化
    '''
    W = np.zeros((4,1))
    b = 0
    return W,b

def gradient_descent(sig,X_train,b,y_train,W):
    #learning rate
    alpha = 0.001
    W_temp = W - alpha * np.dot(X_train,(sig-y_train).T)
    b_temp = b - alpha *np.sum(sig-y_train)
    return b_temp, W_temp

def draw(cost_list,number):

    plt.title("cost function curve")
    plt.xlabel('frequency')
    plt.ylabel('cost function')
    plt.plot(range(number),cost_list,'g.-')
    plt.show()

def PCA(X):
    pca = sd.PCA(n_components=3)
    pca.fit(X.T)
    X_PCA = pca.fit_transform(X.T)
    return X_PCA.T

def scatter3D(x,y,z):
    global num_train

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x[0:num_train//2],y[0:num_train//2],z[0:num_train//2],c='r')
    ax.scatter(x[num_train//2:num_train],y[num_train//2:num_train],z[num_train//2:num_train],c='g',marker='^')
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')
    ax.set_zlabel('zlabel')
    plt.show()

if __name__=='__main__':
    print('start……')
    time_start = time.time()
    #the number of train/test set
    num_train = 80
    num_test = 100 - num_train
    #parameters
    W,b = initial_params()
    #file path
    path ='iris.data'
    #train\test set
    positive_tr,negetive_tr,positive_te,negetive_te=load_data(path)

    #生成训练集并训练
    X_train,y_train = train_gen(positive_tr,negetive_tr)
    number = 0
    cost_list = []
    while(True):
        number+= 1
        sig = sigmoid(W,X_train,b)
        b,W = gradient_descent(sig,X_train,b,y_train,W)
        temp = cost(y_train,sig)
        cost_list.append(temp)
        if(temp<0.1):
            break
    time_end = time.time()
    print('time:',time_end-time_start,'number',number)
    print('final:','\n','W:',W,'\n','b',b,'cost',cost(y_train,sig))

    #画出损失函数下降曲线
    draw(cost_list[:100],len(cost_list[:100]))

    #PCA并显示
    X_train_PCA = PCA(X_train)
    scatter3D(X_train_PCA[0],X_train_PCA[1],X_train_PCA[2])

    #生成测试集测试
    X_test,y_test = test_gen(positive_te,negetive_te)
    y_hat = sigmoid(W,X_test,b)
    y_hat = y_hat * 2 //1#把预测值变为0，1
    error = np.sum(abs(y_hat-y_test))/100
    print('error',error)

