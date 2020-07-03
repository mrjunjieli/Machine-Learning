#-*- coding:utf-8 -*-
'''
@time:2019/1/9
@author:Jay
结合标签数据给用户进行物品推荐的简单算法
'''

import pandas as pd
import re
import os
from sklearn import model_selection as cv
from collections import defaultdict
from operator import itemgetter



class SimpleTag:
    '''
    Top N recommendation
    '''
    def __init__(self,recommend):
        # self.data = pd.DataFrame()      #存储电影用户评分标签数据

        self.n_users = 0
        self.n_movies =0	    
        self.n_tags = 0

        self.genres2int={}              #电影标签转数字

        self.test_data = pd.DataFrame()

        self.popular = {}               #电影流行度
        self.active = {}                #用户活跃度

        self.User_Movie_dict = defaultdict(dict) #记录用户看过那些电影

        self.UserTag = defaultdict(dict) #存储用户对标签的喜爱度

        self.num_recom_movies = recommend

        print("recommend movie number =",recommend)
        print('-'*20)

    def load_data(self,filename):
        '''
        load Dataset from File
        '''
        #读取movies 数据集
        movies_title= ['MovieID','Title','Genres']
        movies_dat = pd.read_table(filename+'movies.dat',sep='::',header=None,names=movies_title,engine='python')
        movies_dat = movies_dat.filter(regex='MovieID|Genres')
        #电影类型转数字字典
        genres_set = set()
        for val in movies_dat['Genres'].str.split('|'):
            genres_set.update(val)
        self.genres2int= {val:ii for ii,val in enumerate(genres_set)}
        self.n_tags = len(self.genres2int)
        #将电影类型转成数字列表
        genres_map = {val:[self.genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies_dat['Genres']))}
        movies_dat['Genres'] = movies_dat['Genres'].map(genres_map)

        #获取评分数据集
        ratings_title=['UserID','MovieID','ratings','timestamps']   
        ratings_dat = pd.read_table(filename+'ratings.dat',sep='::',header=None,names = ratings_title,engine='python')
        ratings_dat = ratings_dat.filter(regex='UserID|MovieID|ratings')

        #合并两个表
        data = pd.merge(ratings_dat,movies_dat) #UserID  MovieID  ratings   Genres
        return data

    def generate_dataset(self,filename,test_size):
        '''split train || test dataset'''
        #test_size [0,1]
        train_data,self.test_data = cv.train_test_split(self.load_data(filename),shuffle=True,test_size=test_size)

        #calculate the numbers of (usr and movie)
        self.n_users = train_data.UserID.unique().shape[0]
        self.n_movies = train_data.MovieID.unique().shape[0]

        print("-"*20)
        print("the numbers of the Users is:",self.n_users)
        print("the nubmers of the movies is:",self.n_movies)
        print("the numbers of the tag :",self.n_tags)
        print("-"*20)

        print("the numbers of the train samples:",train_data.shape[0])
        print("the numbers of the test samples:",self.test_data.shape[0])

        print("-"*20)
        print("tag to int map:")
        print(self.genres2int)
        self.generateUserTag(train_data)

    def generateUserTag(self,trainset):
        '''
        生成用户标签矩阵 代表这用户对含有该标签的电影的喜爱程度
        '''
        
        for line in trainset.itertuples():
            self.User_Movie_dict[int(line[1])-1][int(line[2])-1] = int(line[3])
            for tag in line[4]:
                try:
                    self.UserTag[int(line[1])-1][int(tag)] += int(line[3])
                except:
                    self.UserTag[int(line[1])-1][int(tag)] = int(line[3])
            #统计电影的流行度
            try:
                self.popular[int(line[2]-1)] += 1                               #{MovieID:num}
                self.active[int(line[1])-1] +=1 
            except:                                                             #出现一次流行度加一
                self.popular[int(line[2])-1] = 1
                self.active[int(line[1])-1] =1

        self.evaluate(trainset)

    def recommend(self,testUsers,trainset):
        '''Find K similar movies and recommend N movies(hasn't watched) '''
        print("-"*20)
        print("<<<start recommend for users......")

        recommend = defaultdict(list)                                                #推荐列表
        for user in testUsers:
            Movie_tagscore = {}                                                 #记录电影基于标签的评分 {MovieID:score}
            for line in trainset.itertuples():
                try:
                    self.User_Movie_dict[user][int(line[2])-1]
                except:
                    for tag in line[4]:
                        if tag in self.UserTag[user]:
                            try:
                                Movie_tagscore[int(line[2])-1] += self.UserTag[user][tag]
                            except:
                                Movie_tagscore[int(line[2])-1] = self.UserTag[user][tag]
            temp = sorted(Movie_tagscore.items(),key = itemgetter(1),reverse=True)[:self.num_recom_movies]
            for movie in temp:
                recommend[user].append(movie)
        print(">>>success!!!")
        return recommend	            

    def evaluate(self,trainset):
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

        recommend_allUsers = self.recommend(testUsers,trainset)#记录推荐系统给每个用户的推荐结果

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



if __name__ =='__main__':
    filename = os.path.join('./ml-1m/')
    recommend = 15	            #每个用户推荐的物品个数
    test_size = 0.3
    STag = SimpleTag(recommend)
    STag.generate_dataset(filename,test_size)
