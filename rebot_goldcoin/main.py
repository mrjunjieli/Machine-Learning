#迷宫
import gym
import random
from grid_mdp import GridEnv
from grid_mdp import GridEnv_com
import time
import matplotlib.pyplot as plt

def Qlearn(alpha,epsilon):
    #alpah 折扣因子
    #epsilon 部分贪婪

    # grid = GridEnv()
    grid = GridEnv_com()
    
    # x = []
    # y = []

    qfunc = dict()
    #初始化行为值函数为0或者reward
    for s in grid.states:
        for a in grid.actions:
            key = "%d_%s"%(s,a)
            try:
                qfunc[key] = grid.rewards[key]
            except:
                qfunc[key] = 0
    

    # for iter1 in range(num_iter1):
    #     x.append(iter1)
    #     y.append(compute_error(qfunc))
    


    #初始化初始状态
    s = grid.reset()  #state
    # s=3
    #，根据部分贪婪策略在状态s选择a
    a = epsilon_greedy(qfunc,s,epsilon,grid.actions)

    # a = 's'
    is_terminal = False
    count_last = 0
    count = 0
    step = 0
    count_list = []

    while True:

        # if count ==count_last:
        #     break
        a = epsilon_greedy(qfunc,s,epsilon,grid.actions)

        if step >100:
            plt.figure()
            plt.plot([x for x in range(len(count_list))],count_list)
            plt.show()
            break
        if int(s)==14:
            print('find!!!!!!!!!!!!!!!','count:',count)
            print('寻找成功 ')
            step+=1
            # print('迭代次数',step)
            count_list.append(count)
            count_last = count 
            count = 0
            for x,y in qfunc.items():
                if y != 0:
                    print('index:',x,'value:',y)
            # break

        if True==is_terminal:
            #碰到黑洞则重新开始
            # print('----')
            s = grid.reset()
            a = epsilon_greedy(qfunc,s,epsilon,grid.actions)
            is_terminal = False
            step+=1
            print('掉入黑洞 ^^^^^^^^^^^^^^')


        grid.render()

        key = "%d_%s"%(s,a)
        # print(s,'->',a)
        grid.state = int(s)
        
        #与环境进行一次交互，从环境中得到新的状态及回报
        new_state,r,is_terminal,info = grid.step(a)
        
        #new_state处的最大动作
        temp_key = ["%d_%s"%(new_state,x) for x in grid.actions]
        random.shuffle(temp_key)
        temp_value = [qfunc[x] for x in temp_key]
        # if len(set(temp_value)) ==1:
        #     a_new = list(temp_key[int(random.random()*len(temp_key))])[-1]
        # else:
        a_new = list(temp_key[temp_value.index(max(temp_value))])[-1]


        key_new = "%d_%s"%(new_state,a_new)

        #利用qlearing方法更新值函数
        qfunc[key] = qfunc[key]+alpha*(r+grid.gamma*qfunc[key_new]-qfunc[key])
        
        #转到下一个状态
        s = new_state
        a = a_new

        count += 1
        # time.sleep(0.1)

    # return qfunc


def epsilon_greedy(qfunc,s,epsilon,actions):

    temp_key = ["%d_%s"%(s,x) for x in actions]
    random.shuffle(temp_key)
    temp_value = [qfunc[x] for x in temp_key]
    # if len(set(temp_value)) ==1:
    #     a = list(temp_key[int(random.random()*len(temp_key))])[-1]
        
    # else:
    a = list(temp_key[temp_value.index(max(temp_value))])[-1]

    ran = random.random()
    if ran>epsilon:
        return actions[int(random.random()*len(actions))]
    else:
        return a

def Sarsa(alpha,epsilon):
    #alpah 折扣因子
    #epsilon 部分贪婪

    # grid = GridEnv()
    grid = GridEnv_com()
    
    # x = []
    # y = []

    qfunc = dict()
    #初始化行为值函数为0或者reward
    for s in grid.states:
        for a in grid.actions:
            key = "%d_%s"%(s,a)
            try:
                qfunc[key] = grid.rewards[key]
            except:
                qfunc[key] = 0
    

    # for iter1 in range(num_iter1):
    #     x.append(iter1)
    #     y.append(compute_error(qfunc))
    


    #初始化初始状态
    s = grid.reset()  #state
    # s=3
    #，根据部分贪婪策略在状态s选择a
    a = epsilon_greedy(qfunc,s,epsilon,grid.actions)

    # a = 's'
    is_terminal = False
    count_last = 0
    count = 0
    step = 0
    count_list = []

    while True:

        # if count ==count_last:
        #     break
        a = epsilon_greedy(qfunc,s,epsilon,grid.actions)

        if step >100:
            plt.figure()
            plt.plot([x for x in range(len(count_list))],count_list)
            plt.show()
            break
        if int(s)==14:
            print('find!!!!!!!!!!!!!!!','count:',count)
            print('寻找成功 ')
            step+=1
            # print('迭代次数',step)
            count_list.append(count)
            count_last = count 
            count = 0
            for x,y in qfunc.items():
                if y != 0:
                    print('index:',x,'value:',y)
            # break

        if True==is_terminal:
            #碰到黑洞则重新开始
            # print('----')
            s = grid.reset()
            a = epsilon_greedy(qfunc,s,epsilon,grid.actions)
            is_terminal = False
            step+=1
            print('掉入黑洞 ^^^^^^^^^^^^^^')


        grid.render()

        key = "%d_%s"%(s,a)
        # print(s,'->',a)
        grid.state = int(s)
        
        #与环境进行一次交互，从环境中得到新的状态及回报
        new_state,r,is_terminal,info = grid.step(a)
        
        ran = random.random()
        if ran>epsilon:
            a_new = grid.actions[int(random.random()*len(grid.actions))]
        else:
            #new_state处的最大动作
            temp_key = ["%d_%s"%(new_state,x) for x in grid.actions]
            random.shuffle(temp_key)
            temp_value = [qfunc[x] for x in temp_key]
            # if len(set(temp_value)) ==1:
            #     a_new = list(temp_key[int(random.random()*len(temp_key))])[-1]
            # else:
            a_new = list(temp_key[temp_value.index(max(temp_value))])[-1]


        key_new = "%d_%s"%(new_state,a_new)

        qfunc[key] = qfunc[key]+alpha*(r+grid.gamma*qfunc[key_new]-qfunc[key])
        
        #转到下一个状态
        s = new_state
        a = a_new

        count += 1
        # time.sleep(0.1)

    # return qfunc


if __name__ == "__main__":
    
    Qlearn(0.1,0.9)
    # Sarsa(0.1,0.95)
