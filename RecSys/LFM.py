'''
@time:2019/1/5
@author:yu yuanqiang
隐语义模型  
待学习改进……
'''
    
import random
import numpy
import math
import sys
from operator import itemgetter

# 置随机数种子
random.seed(0)


# 打印分界线
def print_line():
    print("-"*80)


class Data:
    def __init__(self, dataset='ml-100k'):
        """ 无上下文信息的隐性反馈数据集 """

        path = None
        separator = None
        if dataset == 'ml-100k':
            path = 'ml-100k/u.data'
            separator = '\t'

        print('开始读取数据...')

        # 从源文件读数据
        self.data = []
        count = 0
        for line in open(path, 'r'):
            count += 1
            # 这一步是为了使运算速度更快，读取部分数据
            if count == 1001:
                break
            data_line = line.split(separator)
            userID = int(data_line[0])
            movieID = int(data_line[1])

            self.data.append([userID, movieID])
        self.num_user = self.compress(self.data, 0)
        self.num_item = self.compress(self.data, 1)

        # 训练集和测试集
        self.train, self.test = self.__split_data()
        print('总共有{}条数据，训练集{}，测试集{}，用户{}，物品{}'.format(len(self.data), len(self.train), len(self.test), self.num_user,
                                                      self.num_item))
        print_line()

    def compress(self, data, col):
        """ 压缩数据data第col列的数据 """
        e_rows = dict()  # 键是data数据第col列数据，值是一个存放键出现在的每一个行号的列表
        for i in range(len(data)):
            e = data[i][col]
            if e not in e_rows:
                e_rows[e] = []
            e_rows[e].append(i)

        for rows, i in zip(e_rows.values(), range(len(e_rows))):
            for row in rows:
                data[row][col] = i

        return len(e_rows)

    def __split_data(self):
        ''' 划分训练集与测试集 '''
        test = []
        train = []
        for user, item in self.data:
            if random.randint(1, 8) == 1:
                test.append([user, item])
            else:
                train.append([user, item])
        return train, test


class LFMBasedCF:
    def __init__(self, data):
        """ 隐语义模型算法 """
        self.data = data
        self.ratio = None  # 负正样本比例
        self.max_iter = None  # 学习迭代次数
        self.F = None  # 隐类个数
        self.N = None  # 每个用户最多推荐物品数量

        self.P = None  # P[u][k]是用户u和第k个隐类的关系
        self.Q = None  # Q[k][i]是物品i和第k个隐类的关系
        self.recommendation = None

    def compute_recommendation(self, ratio=20, max_iter=30, F=100, N=10):
        """ 开始计算推荐列表 """
        self.ratio = ratio
        self.max_iter = max_iter
        self.F = F
        self.N = N

        print('开始计算P,Q矩阵（ratio=' + str(self.ratio) + ', max_iter=' + str(self.max_iter) + ', F=' + str(self.F) + '）')
        self.P, self.Q = self.__latent_factor_model()

        print('开始计算推荐列表（N=' + str(self.N) + '）')
        self.recommendation = self.__get_recommendation()

    def __select_negative_sample(self):
        """ 对每个用户分别进行负样本采集 """
        train_user_items = [set() for u in range(self.data.num_user)]
        item_pool = []  # 候选物品列表，每个物品出现的次数和其流行度成正比
        for user, item in self.data.train:
            train_user_items[user].add(item)
            item_pool.append(item)

        user_samples = []
        for user in range(self.data.num_user):
            sample = dict()
            for i in train_user_items[user]:  # 设置用户user所有正反馈物品为正样本（值为1）
                sample[i] = 1
            n = 0  # 已取负样本总量
            max_n = int(len(train_user_items[user]) * self.ratio)  # 根据正样本数量和负正样本比例得到负样本目标数量
            for i in range(max_n * 3):
                item = random.choice(item_pool)
                if item in sample:
                    continue
                sample[item] = 0
                n += 1
                if n >= max_n:
                    break
            user_samples.append(sample)
        return user_samples

    def __latent_factor_model(self, alpha=0.02, lam_bda=0.01):
        print('对每个用户采集负样例')
        user_samples = self.__select_negative_sample()
        samples_ui = [[], []]
        samples_r = []
        for user in range(self.data.num_user):
            for item, rui in user_samples[user].items():
                samples_ui[0].append(user)
                samples_ui[1].append(item)
                samples_r.append(rui)
        samples_ui = numpy.array(samples_ui, numpy.int)
        samples_r = numpy.array(samples_r, numpy.double)

        k = 1 / math.sqrt(self.F)
        P = numpy.array([[random.random() * k for f in range(self.F)] for u in range(self.data.num_user)])
        Q = numpy.array([[random.random() * k for i in range(self.data.num_item)] for f in range(self.F)])
        return self.gradient_decsent(alpha, lam_bda, self.max_iter, P, Q, samples_ui, samples_r)

    def gradient_decsent(self, alpha, lam_bda, max_iter, P, Q, samples_ui, samples_r):
        n = samples_r.shape[0]
        F = P.shape[1]

        for step in range(max_iter):
            print('随机梯度下降法学习P,Q矩阵中... ', step + 1, '/', max_iter)
            for i in range(n):
                user = samples_ui[0, i]
                item = samples_ui[1, i]
                rui = samples_r[i]

                eui = rui - P[user, :].dot(Q[:, item])
                for f in range(F):
                    P[user, f] += alpha * (eui * Q[f, item] - lam_bda * P[user, f])
                    Q[f, item] += alpha * (eui * P[user, f] - lam_bda * Q[f, item])

            alpha *= 0.9

        return P, Q

    def __recommend(self, user_PQ, user_item_set):
        """ 给用户user推荐最多N个物品 """

        rank = dict()
        for i in set(range(self.data.num_item)) - user_item_set:
            rank[i] = user_PQ[i]
        return [items[0] for items in sorted(rank.items(), key=itemgetter(1), reverse=True)[:self.N]]

    def __get_recommendation(self):
        """ 得到所有用户的推荐物品列表 """
        # 得到训练集中每个用户所有有过正反馈物品集合
        train_user_items = [set() for u in range(self.data.num_user)]
        for user, item in self.data.train:
            train_user_items[user].add(item)

        PQ = self.P.dot(self.Q)

        # 对每个用户推荐最多N个物品
        recommendation = []
        for user_PQ, user_item_set in zip(PQ, train_user_items):
            recommendation.append(self.__recommend(user_PQ, user_item_set))
        return recommendation


class Evaluation:
    def __init__(self, recommend_algorithm):
        """ 对推荐算法recommend_algorithm计算各种评测指标 """
        self.rec_alg = recommend_algorithm

        self.precision = None
        self.recall = None
        self.coverage = None
        self.popularity = None

    def evaluate(self):
        """
        评测指标的计算。
        """
        # 准确率和召回率
        self.precision, self.recall = self.__precision_recall()

        # 覆盖率
        self.coverage = self.__coverage()

        # 流行度
        self.popularity = self.__popularity()

        print('准确率 = %.4f\t召回率 = %.4f\t覆盖率 = %.4f\t流行度 = %.4f' %
              (self.precision, self.recall, self.coverage, self.popularity), file=sys.stderr)
        print_line()

    def __precision_recall(self):
        """ 计算准确率和召回率 """
        # 得到测试集用户与其所有有正反馈物品集合的映射
        test_user_items = dict()
        for user, item in self.rec_alg.data.test:
            if user not in test_user_items:
                test_user_items[user] = set()
            test_user_items[user].add(item)

        # 计算准确率和召回率
        hit = 0
        all_ru = 0
        all_tu = 0
        for user, items in test_user_items.items():
            ru = set(self.rec_alg.recommendation[user])
            tu = items

            hit += len(ru & tu)
            all_ru += len(ru)
            all_tu += len(tu)
        return hit / all_ru, hit / all_tu

    def __coverage(self):
        """ 计算覆盖率 """
        recommend_items = set()
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                recommend_items.add(item)
        return len(recommend_items) / self.rec_alg.data.num_item

    def __popularity(self):
        """ 计算新颖度 """
        item_popularity = [0 for i in range(self.rec_alg.data.num_item)]
        for user, item in self.rec_alg.data.train:
            item_popularity[item] += 1

        ret = 0
        n = 0
        for user in range(self.rec_alg.data.num_user):
            for item in self.rec_alg.recommendation[user]:
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / n


if __name__ == '__main__':
    # 初始化隐语义模型类
    recommend = LFMBasedCF(Data())
    # 根据此算法进行推荐
    recommend.compute_recommendation()
    # 评估推荐结果
    eva = Evaluation(recommend)
    eva.evaluate()
