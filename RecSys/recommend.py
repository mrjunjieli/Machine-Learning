'''
an example based off the movieslens 100k dataset
'''

from math import sqrt

import numpy as np
import pandas as pd
from sklearn import model_selection as cv
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

#基于内容的协同过滤

#load u.data
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

#calculate the numbers of the (user and movies)
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

#split train || test data
train_data, test_data = cv.train_test_split(df, test_size=0.25,shuffle = True)
print('number of train samples:',train_data.shape[0],'test samples:',test_data.shape[0])

#!create two USER-ITEM matrices, one for training another for testing!
train_data_matrix = np.zeros((n_users,n_items))

for line in train_data.itertuples():
    train_data_matrix[int(line[1])-1,int(line[2])-1]= int(line[3])

test_data_matrix = np.zeros((n_users,n_items))
for line in test_data.itertuples():
    test_data_matrix[int(line[1])-1,int(line[2])-1] = int(line[3])


#calculate 余弦相似度
#一行为一个向量组
#该函数当两个向量完全相似的时候计算出的相似度为0
#from sklearn.metrics.pairwise import cosine_similarity 相似度为1
user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T,metric='cosine')
print(item_similarity)

def predict(ratings,similarity,type = 'user'):
    if type == 'user':
        #避免不同用户的评分标准不同 因此去中心化
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings-mean_user_rating[:,np.newaxis])#去中心化
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)/ np.array([np.abs(similarity).sum(axis=1)]).T#axis=1,行数不变
    elif type=='item':
        pred=ratings.dot(similarity)/np.array([np.abs(similarity).sum(axis=0)])
    else:
        print("Error arguments 'type'")
    return pred

user_prediction = predict(train_data_matrix,user_similarity,'user')
item_prediction = predict(train_data_matrix,item_similarity,'item')


#模型评估
def rmse(prediction,ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()#返回非零元素索引 把数组展平
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction,ground_truth))

print("User-based CF RMSE:",rmse(user_prediction,test_data_matrix))
print('Item-based CF RMSE:',rmse(item_prediction,test_data_matrix))

#计算稀疏度
sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')
