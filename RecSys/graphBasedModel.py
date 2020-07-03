#-*- coding:utf-8 -*-
'''
@time:2019/1/3
@author:Jay
@vision:2.0
《推荐系统实践》中关于基于图的物品物品推荐算法
PersonalRank及其改进
数据：MovieLens ml-100k

需要优化的地方：
2、用户的评分对图的影响
'''

import math
import os
import pandas as pd
from sklearn import model_selection as cv
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import time
from operator import itemgetter

class PersonalRank:
    '''
    TopN recommendation
    '''
    def __init__(self,recommend):
        self.train_data =[]                 #训练集中压缩矩阵的值
        self.train_row_ind = []             #训练集中压缩矩阵的行
        self.train_col_ind = []             #训练集中压缩矩阵的列
        
        self.test_data=pd.DataFrame()       #测试集数据

        self.cmap_userID = {}               #{UserID:row_num}
        self.cmap_movieID = {}              #{MovieID:row/col}  
        self.cmap_userID_un={}              #{row/col:UserID}	         
        self.cmap_movieID_un={}             #{row/col:MovieID}

        self.n_users = 0	                #用户数量
        self.n_movies = 0	                #电影个数

        self.Graph_UserID = defaultdict(dict)#{UserID:{MovieID:rate}} 存储训练数据中的图模型 

        self.num_recom_movies = recommend	#选取recommend个电影推荐给用户

        self.popular = {}                   #记录电影的流行度 {MovieID,popular} 

        print("recommend movie number =",recommend)
        print('-'*20)

    def loadfile(self,filename):
        '''load a file'''
        print(">>>loading ratins.dat.....")
        header = ['UserID','MovieID','Rating','Timestamp']
        df = pd.read_csv(filename,sep='\t',names=header,engine='python')
        print("<<<load success!!!")

        return df

    def generate_dataset(self,filename,test_size):
        '''split train || test dataset'''
        #test_size [0,1]
        train_data,self.test_data = cv.train_test_split(self.loadfile(filename),shuffle=True,test_size=test_size)

        #calculate the numbers of (usr and movie)
        self.n_users = train_data.UserID.unique().shape[0]
        self.n_movies = train_data.MovieID.unique().shape[0]

        print("-"*20)
        print("the numbers of the Users is:",self.n_users)
        print("the nubmers of the movies is:",self.n_movies)
        print("-"*20)

        print("the numbers of the train samples:",train_data.shape[0])
        print("the numbers of the test samples:",self.test_data.shape[0])

        self.generateGraph(train_data)


    def generateGraph(self,train_set):
        '''根据训练数据生成图模型和计算矩阵'''
        print("-"*20)
        print(">>>generate Graph Matrix......")

        Graph_MovieID = defaultdict(dict)     
        #生成图 用多值字典存储
        for line in train_set.itertuples():
            self.Graph_UserID[int(line[1])-1][int(line[2])-1] = int(line[3])    #{UserID:{MovieID:rate}}
            Graph_MovieID[int(line[2]-1)][int(line[1]-1)] = int(line[3])        #{MovieID:{UserID:rate}}
            #统计电影的流行度 
            try:
                self.popular[int(line[2]-1)] += 1                               #{MovieID:num}
            except:                                                             #出现一次流行度加一
                self.popular[int(line[2])-1] = 1

        #映射UserID 与矩阵中的行列数字
        for i,key in enumerate(self.Graph_UserID):
            self.cmap_userID[key] =i	                                        #{UserID:row/col}
        #映射MovieID 与矩阵中的行列 定义电影排在用户后面
        for i,key in enumerate(Graph_MovieID):
            self.cmap_movieID[key] =i+self.n_users	                            #{MovieID:row/col}
        
        #self.cmap_movieID解码
        self.cmap_movieID_un = dict(zip(self.cmap_movieID.values(),self.cmap_movieID.keys()))
        #self.cmap_userID解码
        self.cmap_userID_un = dict(zip(self.cmap_userID.values(),self.cmap_userID.keys()))

        #根据图计算矩阵
        Matrix = np.zeros((self.n_movies+self.n_users,self.n_movies+self.n_users))
        for key,value in self.Graph_UserID.items():
            # total =0
            for x in value:
                Matrix[self.cmap_userID[key],self.cmap_movieID[x]] = 1/len(value)
                # total+=value[x]
            # for x in value:
                # Matrix[self.cmap_userID[key],self.cmap_movieID[x]] /= total
        for key,value in Graph_MovieID.items():
            for x in value:
                Matrix[self.cmap_movieID[key],self.cmap_userID[x]] = 1/len(value)
        #压缩训练矩阵
        for row in range(self.n_movies+self.n_users):
            for col in range(self.n_movies+self.n_users):
                if Matrix[row,col]!=0:
                    self.train_data.append(Matrix[row,col])
                    self.train_row_ind.append(row)
                    self.train_col_ind.append(col)

        print("<<<success!!!")
    
    def recommend(self,alpha,testUsers):
        ''' 推荐K个用户可能喜欢的电影'''
        # r0 = np.zeros((self.n_movies+self.n_users,1))
        # r0[self.cmap_userID[user],0] = 1                                    #将user作为根
        
        #根据压缩的数据恢复矩阵
        Matrix = csr_matrix((self.train_data,(self.train_row_ind,self.train_col_ind)),\
        shape=(self.n_users+self.n_movies,self.n_movies+self.n_users)).toarray()
        #方程中的系数
        A = np.mat(np.eye(self.n_movies+self.n_users)-alpha*Matrix.T)           #alpha 条件转移的概率

        recommend =defaultdict(list)                                            #{UserID:[MovieID]}
        D = A.I
        #生成所有用户的推荐电影ID
        print("<<<start recommend movies for users......")
        for i in testUsers:
            score = {}                                                          #{MovieID:num}
            for j in range(self.n_users,self.n_movies+self.n_users):
                try:
                    self.Graph_UserID[i][self.cmap_movieID_un[j]]               #已经看过的电影不推荐
                except:
                    score[self.cmap_movieID_un[j]] = D[j,self.cmap_userID_un[i]]
            temp = sorted(score.items(),key = itemgetter(1),reverse = True)[:self.num_recom_movies]
            for k in range(len(temp)):
                recommend[i].append(temp[k][0])
        print(">>>success!!!")
        return recommend	                                                   #返回MovieID

    def evaluate(self,alpha):
        ''' 对推荐系统算法进行评估: 准确率, 召回率, 覆盖度，流行度 '''
        print("-"*20)
        print(">>>start evaluating......")

        popular_sum = 0	             #流行度的和
        hit = 0                      #代表命中的个数
        user_watch_count = 0	     #用户实际观看的电影个数
        recommend_allUsers_count = 0 #推荐系统推荐给用户看的电影个数
        recommend_set = set()        #推荐系统所有的推荐结果

        #测试集中的数据
        testUsers = self.test_data.UserID.unique()-1                            #userID
        # testUsers = [0,1,2]
        
        watch_allUsers =defaultdict(list)                                       #{UserID:movieId} 用户实际观看的电影
        for line in self.test_data.itertuples():
            watch_allUsers[int(line[1])-1].append(int(line[2])-1)

        for i in watch_allUsers:
            user_watch_count += len(watch_allUsers[i])

        recommend_allUsers = self.recommend(alpha,testUsers)                     #记录推荐系统给每个用户的推荐结果

        recommend_allUsers_count = self.num_recom_movies * len(testUsers)        #推荐系统推荐给用户看的电影个数
        
        for i in range(len(testUsers)):
            for movie in recommend_allUsers[testUsers[i]]:
                recommend_set.add(movie)
                try:
                    popular_sum += math.log(1+self.popular[movie])
                except:
                    pass
                if movie in watch_allUsers[testUsers[i]]:
                    hit =hit +1


        recall = hit/user_watch_count	                #召回率
        precision = hit/recommend_allUsers_count	    #准确率
        coverage = len(recommend_set)/self.n_movies	    #覆盖率
        popular =  popular_sum/recommend_allUsers_count #流行度

        print('precision = %.4f%%\trecall = %.4f%%\tcoverage = %.4f%%\tpopular = %.4f' %
               (precision*100, recall*100, coverage*100, popular))

if __name__ == '__main__':

    #文件路径
    ratingsFile = os.path.join('ml-100k','u.data')
    test_size = 0.3                                   #测试集占的比例
    Prank = PersonalRank(15)
    alpha = 0.99                                     #选择游走的概率
    Prank.generate_dataset(ratingsFile,test_size)
    Prank.evaluate(alpha)