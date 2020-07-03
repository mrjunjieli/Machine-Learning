#-*- encoding:utf-8 -*-
'''iris的对数几率模型
两个属性，二分类
70%训练 30%测试'''

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random as rd

Data = np.loadtxt(".\\iris.data",delimiter=',',dtype=str)
##把花种类用0或1替换
flag = Data[:,4]
flag[flag=="Iris-setosa"]=1
flag[flag=='Iris-versicolor']=0

data = rd.sample(list(Data),90)#随机获取70个数据
data = np.array(data)
x1 = data[:,0]#第一个属性值
x2 = data[:,1]#第二个属性值
y = data[:,4]#iris的分类
#change数据类型为float
x1 = x1.astype(float)
x2 = x2.astype(float)
y= y.astype(int)#把0,1转成可以参与计算的整型
m = len(x1)#数据的个数

#three parameters
theta2_num = 0
theta1_num = 0
b_num = 0

theta2,theta1,b = sp.symbols("theta2 theta1 b")

alpha = 0.1#learning rate

#hypothesis function
def h_liner(x1,x2):#linear
    return theta1*x1+theta2*x2+b
def h(x1,x2):#sigmoid function
    return 1/(1+np.e**(-h_liner(x1,x2)))
h = h(x1,x2)

#cost function
def cost():
    return np.array(list(map(lambda m,n:m*sp.log(n)+(1-m)*sp.log(1-n),y,h)))
def cost_func():
    return -1/m*np.sum(cost())
cost_function = cost_func()

#theta1的偏导数
def gra_des1(theta1):
    return sp.diff(cost_function,theta1)
grades1 = gra_des1(theta1)

#theta2偏导数
def gra_des2(theta2):
    return sp.diff(cost_function,theta2)
grades2 = gra_des2(theta2)

#b的偏导数
def gra_desb(b):
    return sp.diff(cost_function,b)
gradesb = gra_desb(b)

number = 0#梯度下降的次数
cost_list = []#所有代价函数的值的列表
#cost_last=0#上一次的代价函数值
while 1:
    number += 1
    temp1 = theta1_num - alpha * grades1.subs([(theta1,theta1_num),(theta2,theta2_num),(b,b_num)])
    temp2 = theta2_num - alpha * grades2.subs([(theta1,theta1_num),(theta2,theta2_num),(b,b_num)])
    tempb = b_num - alpha * gradesb.subs([(theta1,theta1_num),(theta2,theta2_num),(b,b_num)])
    theta2_num = temp2
    theta1_num = temp1
    b_num = tempb
    cost_now = cost_function.subs([(theta1,theta1_num),(theta2,theta2_num),(b,b_num)])#这一次的代价函数值
    cost_list.append(cost_now)
    if number>10/alpha:
        if abs(cost_list[-2]-cost_list[-1])<0.1:#当两次代价函数之差小于特定值时，就认为取到全局最优点
            break
print("cost_function:%f"%(cost_now))
print('theta1=%f,theta2=%f,b=%f'%(theta1_num,theta2_num,b_num))


#建立测试集
set1 = set([tuple(num) for num in Data])
set2 = set([tuple(num) for num in data])
test_data = np.array(list(map(lambda x:list(x),list(set1-set2))))#去掉训练集中重复的数据
x1_test = test_data[:,0]#保证test_data要为narray类型才能使用
x2_test = test_data[:,1]
y_test= test_data[:,4]
#更改数据类型为float
x1_test = x1_test.astype(float)
x2_test = x2_test.astype(float)
#把种类用0或1替换
y_test= y_test.astype(int)
m_test = len(x1_test) #测试集个数


#统计正确率
error = 0#错误的个数
h_test = 1/(np.e**(-(theta1_num*x1_test+theta2_num*x2_test+b_num))+1)#测试集的预测输出
h_test[h_test>=0.5]=1#输出大于0.5都认为是1
h_test[h_test<0.5]=0
for x in (h_test-y_test):
    error +=abs(x)
print("正确率为：%f%%"%((1-error/m_test)*100))

#绘制坐标图
plt.title("red:Iris-setosa green:Iris-versicolor")#标题
plt.xlabel("sepal length in cm")#萼片长度
plt.ylabel("sepal width in cm")#萼片长度
plt.axis([4,7,2,5])#确定坐标轴的坐标分布
#绘制函数
element =np.arange(4,8)#确定函数的x范围
hx2 = (theta1_num*element+b_num)/(-theta2_num)#分割线函数（当y=0)
plt.plot(element,hx2,color='blue')#打印函数图形
#给不同的点分类
y = list(y)
y_test=list(y_test)
x1_1=x1[list([i for i,x in enumerate(y) if x ==1])]#获取训练集中分类为1的属性值
x2_1=x2[list([i for i,x in enumerate(y) if x ==1])]#该函数返回索引和值组成的元组
x1_0=x1[list([i for i,x in enumerate(y) if x ==0])]#获取训练集中分类为0的属性值
x2_0=x2[list([i for i,x in enumerate(y) if x ==0])]
x1_test_1=x1_test[list([i for i,x in enumerate(y_test) if x ==1])]#获取测试集集中分类为1的属性值
x2_test_1=x2_test[list([i for i,x in enumerate(y_test) if x ==1])]
x1_test_0=x1_test[list([i for i,x in enumerate(y_test) if x ==0])]#获取测试集集中分类为0的属性值
x2_test_0=x2_test[list([i for i,x in enumerate(y_test) if x ==0])]
#打印点
#Iris-setosa
plt.plot(x1_1,x2_1,'ro')#红色
plt.plot(x1_test_1,x2_test_1,'mo')#品红
#Iris-versicolor
plt.plot(x1_0,x2_0,'go')#绿色
plt.plot(x1_test_0,x2_test_0,'co')#青色
plt.show()
#绘制cost function随次数的增加的变化

plt.title("cost function curve")
plt.xlabel('frequency')
plt.ylabel('cost function')
plt.plot(range(number),cost_list)
plt.show()

