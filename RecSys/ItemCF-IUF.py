#-*- coding:utf-8 -*-

'''
@time:2018/12/31
@author:Jay
@vision:2.0
based on Movielens 10M/100k dataset

《推荐系统实践》 一书中ItemCF-IUF算法的实现
‘基于物品的协同过滤算法’
1、加入了用户的活跃度惩罚
2、加入了热门物品的权重惩罚
3、较快的计算速度
'''

from scipy.sparse import csr_matrix
import os
import pandas as pd
from sklearn import model_selection as cv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from operator import itemgetter
import math


class ItemBasedCF(object):
    '''
    TopN recommendation
    '''
    def __init__(self,similar,recommend):
        self.test_data=pd.DataFrame()       #测试集数据

        self.n_users = 0	                #用户数量
        self.n_movies = 0	                #电影个数

        self.train_data = []                #相似度矩阵的数据
        self.train_row_ind =[]              #行
        self.train_col_ind = []             #列

        self.num_sim_moives = similar	    #选取similar个最相似的电影
        self.num_recom_movies = recommend	#选取recommend个电影推荐给用户

        self.cmap_MovieID = {}              #将MovieID 映射成连续的数字
        self.cmap_MovieID_un = {}           #反向映射

        self.User_Movie_dict=defaultdict(dict)   #记录用户看过的影片 #{userID:{MovieID:rate}}

        self.popular = {}                   #记录电影的流行度  {MovieID:num}
        self.active = {}                    #记录用户活跃度    {UserID:num}

        print("similar movie number =",similar)
        print("recommend movie number =",recommend)
        print('-'*20)

    def loadfile(self,filename):
        '''load a file'''
        print(">>>loading ratins.dat.....")
        header = ['UserID','MovieID','Rating','Timestamp']
        df = pd.read_csv(filename,sep='::',names=header,engine='python')
        print("<<<load success!!!")
        return df

    def generate_dataset(self,filename,test_size,beta):
        '''split train || test dataset'''
        
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

        self.cal_sim_matrix(train_data,beta)

    def cal_sim_matrix(self,train_set,beta):
        '''calculate movie similarity matirx'''
        print("-"*20)
        print(">>calculate itme similarity matrix......")
    
        self.User_Movie_dict = defaultdict(dict)    #{userID:{MovieID:rate}}记录用户看过哪些电影
        Movie_User_dict = defaultdict(dict)         #{MovieID:{UserID:rate}}

        for line in train_set.itertuples():
            self.User_Movie_dict[int(line[1])-1][int(line[2])-1] = int(line[3])
            Movie_User_dict[int(line[2])-1][int(line[1])-1] = int(line[3])
            #统计电影的流行度
            try:
                self.popular[int(line[2]-1)] += 1                               #{MovieID:num}
                self.active[int(line[1])-1] +=1 
            except:                                                             #出现一次流行度加一
                self.popular[int(line[2])-1] = 1
                self.active[int(line[1])-1] =1
        #映射MovieID 成连续的值
        for i,key in enumerate(Movie_User_dict):
            self.cmap_MovieID[key] = i
        #解码cmap_MovieID[key]       
        self.cmap_MovieID_un = dict(zip(self.cmap_MovieID.values(),self.cmap_MovieID.keys()))

        #生成【物品，用户】矩阵
        train_data_matrix = np.zeros((self.n_movies,self.n_users))
        for key,value in Movie_User_dict.items():
            for user in value:
                train_data_matrix[self.cmap_MovieID[key],user] = value[user]/(self.active[user] ** beta)    #加入用户活跃度惩罚

        #calculate 余弦相似度
        #一行为一个向量组
        #当两个向量组完全相似的时候相似度为1
        item_similarity_cosine = cosine_similarity(train_data_matrix)
        for i in range(self.n_movies):
            #对物品j和物品j的相似度置为0
            item_similarity_cosine[i,i] = 0

            #对相似度归一化  大幅提高覆盖率 但是准确率和召回率会有所下降
            # maxInMatrix = np.argmax(self.item_similarity_cosine[i])
            # maxInMatrix = self.item_similarity_cosine[i][maxInMatrix]
            # try:
            #     self.item_similarity_cosine[i] = self.item_similarity_cosine[i]/maxInMatrix
            # except:
            #     pass
        #压缩训练矩阵
        for row in range(self.n_movies):
            for col in range(self.n_movies):
                if item_similarity_cosine[row,col]!=0:
                    self.train_data.append(item_similarity_cosine[row,col])
                    self.train_row_ind.append(row)
                    self.train_col_ind.append(col)

        print("<<<calculate success!!!")
        print("-"*20)

    def recommend(self,testUsers,alpha):
        '''Find K similar movies and recommend N movies(hasn't watched) '''
        print("-"*20)
        print("<<<start recommend for users......")
        #恢复矩阵
        item_similarity_cosine = csr_matrix((self.train_data,(self.train_row_ind,self.train_col_ind)),\
        shape=(self.n_movies,self.n_movies)).toarray()

        recommend = defaultdict(list)                                                #推荐列表
        for user in testUsers:
            recommend_temp = []                                                      #临时的推荐列表
            for movie,rate in self.User_Movie_dict[user].items():   
                intrest = item_similarity_cosine[self.cmap_MovieID[movie]]     	#取出对应的相似度 
                index = [self.cmap_MovieID_un[i] for i in range(self.n_movies)]         #存储intrest对应的MovieID
                temp = sorted(dict(zip(index,intrest)).items(),key=itemgetter(1),reverse=True)[:self.num_sim_moives]
                for i in temp:
                    i = (i[0],i[1]*rate/(self.popular[i[0]]**alpha ))                     #计算兴趣度 加入了评分 惩罚了热门物品权重
                                                                                        #会牺牲准确率而提高覆盖率
                    recommend_temp.append(i)
            temp = sorted(recommend_temp,key=itemgetter(1),reverse=True)
            k = 0
            while len(recommend[user]) != self.num_recom_movies:
                try:
                    self.User_Movie_dict[user][temp[k][0]]
                except:
                    recommend[user].append(temp[k][0])
                finally:
                    k+=1 
        print(">>>success!!!")
        # print(recommend)
        return recommend	                                                            #返回MovieID

    def evaluate(self,alpha):
        ''' 对推荐系统算法进行评估: 准确率, 召回率, 覆盖度，流行度 '''
        print("-"*20)
        print(">>>start evaluating......")

        popular_sum = 0	             #流行度的和
        hit = 0                      #代表命中的个数
        user_watch_count = 0	     #用户实际观看的电影个数
        recommend_allUsers_count = 0 #推荐系统推荐给用户看的电影个数
        recommend_set = set()        #推荐系统所有的推荐结果

        testUsers = self.test_data.UserID.unique()-1                            #userID
        # testUsers = [0,1,2,3]

        watch_allUsers =defaultdict(list)           #[movieId]  illegal    用户实际观看的电影
        for line in self.test_data.itertuples():
            watch_allUsers[int(line[1])-1].append(int(line[2])-1)

        for i in watch_allUsers:
            user_watch_count += len(watch_allUsers[i])

        recommend_allUsers = self.recommend(testUsers,alpha)#记录推荐系统给每个用户的推荐结果

        for i in range(len(testUsers)):
            recommend_allUsers_count += self.num_recom_movies
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
    ratingsFile = os.path.join('ml-1m','ratings.dat')
    test_size = 0.3                                 #测试集占的比例
    alpha = 0.01	                                    #对热门物品的惩罚度 越大惩罚越高 0.1在我的算法里 已经比较大了
    beta = 0.00	                                        #对活跃用户的惩罚系数 越大惩罚越高 
    itemCF = ItemBasedCF(20,15)
    itemCF.generate_dataset(ratingsFile,test_size,beta)
    itemCF.evaluate(alpha)

