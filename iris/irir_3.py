'''
@time:2018/11/26
@version:3.0
classification with svm 
'''

from sklearn import svm
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

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



if __name__ == '__main__':
    #number of training set 
    num_train = 80
    num_test = 100 - num_train
    path ='iris.data'
    #train\test set
    positive_tr,negetive_tr,positive_te,negetive_te=load_data(path)

    X_train,y_train = train_gen(positive_tr,negetive_tr)

    # print(list(y_train[0]))
    clf = svm.SVC(gamma='auto')
    clf.fit(X_train.T,list(y_train[0]))                 #label need to trans to list
    print(X_train.T)

    X_test,y_test = test_gen(positive_te,negetive_te)

    correct = (clf.predict(X_test.T)==y_test).sum()
    print('Accuracy of the 20 test set %d %%'%(100* correct/num_test))
    
