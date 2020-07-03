import logging
import numpy as np
import random
from gym import spaces
import gym

logger = logging.getLogger(__name__)

class GridEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        #状态空间
        self.states = [1,2,3,4,5,6,7,8] 
        #动作空间
        self.actions = ['e','s','w','n']
        #系统状态
        self.state = None

        #每个状态出机器人位置的中心坐标
        self.x=[140,220,300,380,460,140,300,460]
        self.y=[250,250,250,250,250,150,150,150]

        #终止状态为字典格式
        self.terminate_states = dict()  
        self.terminate_states[6] = 1
        self.terminate_states[7] = 1
        self.terminate_states[8] = 1

        #回报的数据结构为字典
        self.rewards = dict()        
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        #状态转移
        self.t = dict()
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma = 0.8         #折扣因子
        self.viewer = None

    def reset(self):
        #随机选取初始位置
        # self.state = self.states[int(random.random() * len(self.states))]
        self.state = 1
        return self.state

    def step(self, action):
        #系统当前状态
        state = self.state
        #判断当前状态是否为中止状态
        if state in self.terminate_states:
            return state, 0, True,{}

        
        #将状态和动作组成字典的键值
        key = "%d_%s"%(state, action)

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        
        self.state = next_state     #更新状态
        # print('state: ' + str(self.state))
        # print('action: ',action)
        
        #判断是否终态
        is_terminal = False
        if next_state in self.terminate_states:
            is_terminal = True

        #获取回报奖励
        try:
            r = self.rewards[key]
        except:
            r = 0
        
        return next_state, r, is_terminal,{}

    def render(self, mode='human', close=False):
        #绘制图像
        #
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            #创建窗口
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            
            
            #创建网格世界
            self.line1 = rendering.Line((100,300),(500,300))
            self.line2 = rendering.Line((100, 200), (500, 200))
            self.line3 = rendering.Line((100, 300), (100, 100))
            self.line4 = rendering.Line((180, 300), (180, 100))
            self.line5 = rendering.Line((260, 300), (260, 100))
            self.line6 = rendering.Line((340, 300), (340, 100))
            self.line7 = rendering.Line((420, 300), (420, 100))
            self.line8 = rendering.Line((500, 300), (500, 100))
            self.line9 = rendering.Line((100, 100), (180, 100))
            self.line10 = rendering.Line((260, 100), (340, 100))
            self.line11 = rendering.Line((420, 100), (500, 100))
            
            
            
            #创建第一个骷髅
            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(140,150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0,0,0)
            #创建第二个骷髅
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(460, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)
            #创建金条
            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(300, 150))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            #创建机器人
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            #给直线设置颜色
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        
        #根据当前状态确定机器人中心坐标
        self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')



class GridEnv_com(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        #状态空间
        self.states = [x for x in range(25)] 
        #动作空间
        self.actions = ['e','s','w','n']
        #系统状态
        self.state = None

        #每个状态出机器人位置的中心坐标
        self.x = [150,250,350,450,550]*5
        self.y = [550,450,350,250,150]*5


        # self.x=[140,220,300,380,460,140,300,460]
        # self.y=[250,250,250,250,250,150,150,150]

        #终止状态为字典格式
        self.terminate_states = dict()  
        self.terminate_states[3] = 1
        self.terminate_states[8] = 1
        self.terminate_states[16] = 1
        self.terminate_states[17] = 1
        self.terminate_states[22] = 1
        self.terminate_states[23] = 1
        self.terminate_states[24] = 1
        self.terminate_states[14] = 1


        #回报的数据结构为字典
        self.rewards = dict() 
        self.rewards['2_e'] = -1.0
        self.rewards['7_e'] = -1.0
        self.rewards['4_w'] = -1.0
        self.rewards['9_w'] = -1.0
        self.rewards['5_s'] = -1.0
        self.rewards['6_s'] = -1.0
        self.rewards['12_w'] = -1.0
        self.rewards['15_n'] = -1.0
        self.rewards['16_n'] = -1.0
        self.rewards['21_e'] = -1.0
        self.rewards['17_s'] = -1.0
        self.rewards['18_s'] = -1.0
        self.rewards['19_s'] = -1.0
        self.rewards['13_n'] = -1.0
        self.rewards['19_n'] = 1.0
        self.rewards['9_s'] = 1.0
        self.rewards['13_e'] = 1.0

        #状态转移
        self.t = dict()
        for s in range(25):
            key_0 = "%d_%s"%(s,'e')
            key_1 = "%d_%s"%(s,'s')
            key_2 = "%d_%s"%(s,'w')
            key_3 = "%d_%s"%(s,'n')
            
            if s not in [0,1,2,3,4]:
                self.t[key_3] = s-5
            if s not in [0,5,10,15,20]:
                self.t[key_2] = s-1
            if s not in [20,21,22,23,24]:
                self.t[key_1] = s+5
            if s not in [4,9,14,19,24]:
                self.t[key_0] = s+1

        self.gamma = 0.8         #折扣因子
        self.viewer = None

    def reset(self):
        #随机选取初始位置
        # self.state = self.states[int(random.random() * len(self.states))]
        self.state = 0
        return self.state

    def step(self, action):
        #系统当前状态
        state = self.state
        #判断当前状态是否为中止状态
        if state in self.terminate_states:
            return state, 0, True,{}

        
        #将状态和动作组成字典的键值
        key = "%d_%s"%(state, action)

        #状态转移
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state
        
        self.state = next_state     #更新状态
        # print('state: ' + str(self.state))
        # print('action: ',action)
        
        #判断是否终态
        is_terminal = False
        if next_state in self.terminate_states:
            is_terminal = True

        #获取回报奖励
        try:
            r = self.rewards[key]
        except:
            r = 0
        
        return next_state, r, is_terminal,{}

    def render(self, mode='human', close=False):
        #绘制图像
        #
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 700
        screen_height = 700

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            #创建窗口
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            
            
            #创建网格世界
            self.line1 = rendering.Line((100,300),(600,300))
            self.line2 = rendering.Line((100, 200), (600, 200))
            self.line3 = rendering.Line((100, 100), (600, 100))
            self.line4 = rendering.Line((100, 400), (600, 400))
            self.line5 = rendering.Line((100, 500), (600, 500))
            self.line6 = rendering.Line((100, 600), (600, 600))

            self.line7 = rendering.Line((100, 100), (100, 600))
            self.line8 = rendering.Line((200, 100), (200, 600))
            self.line9 = rendering.Line((300, 100), (300, 600))
            self.line10 = rendering.Line((400, 100), (400, 600))
            self.line11 = rendering.Line((500, 100), (500, 600))
            self.line12 = rendering.Line((600, 100), (600, 600))
            
            # self.line = dict()
            # for pos in range(25):
            #     self.line[pos] = rendering.Line((),())

            
            #创建第一个骷髅
            self.kulo1 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(350,150))
            self.kulo1.add_attr(self.circletrans)
            self.kulo1.set_color(0,0,0)
            #创建第二个骷髅
            self.kulo2 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(450, 150))
            self.kulo2.add_attr(self.circletrans)
            self.kulo2.set_color(0, 0, 0)
            #threee
            self.kulo3 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(550, 150))
            self.kulo3.add_attr(self.circletrans)
            self.kulo3.set_color(0, 0, 0)
            #four
            self.kulo4 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(150, 350))
            self.kulo4.add_attr(self.circletrans)
            self.kulo4.set_color(0, 0, 0)
            #five
            self.kulo5 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(250, 350))
            self.kulo5.add_attr(self.circletrans)
            self.kulo5.set_color(0, 0, 0)
            #six
            self.kulo6 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(450, 450))
            self.kulo6.add_attr(self.circletrans)
            self.kulo6.set_color(0, 0, 0)
            #seven
            self.kulo7 = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(450, 550))
            self.kulo7.add_attr(self.circletrans)
            self.kulo7.set_color(0, 0, 0)
            #创建金条
            self.gold = rendering.make_circle(40)
            self.circletrans = rendering.Transform(translation=(550, 350))
            self.gold.add_attr(self.circletrans)
            self.gold.set_color(1, 0.9, 0)
            #创建机器人
            self.robot= rendering.make_circle(30)
            self.robotrans = rendering.Transform()
            self.robot.add_attr(self.robotrans)
            self.robot.set_color(0.8, 0.6, 0.4)

            #给直线设置颜色
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)
            self.line4.set_color(0, 0, 0)
            self.line5.set_color(0, 0, 0)
            self.line6.set_color(0, 0, 0)
            self.line7.set_color(0, 0, 0)
            self.line8.set_color(0, 0, 0)
            self.line9.set_color(0, 0, 0)
            self.line10.set_color(0, 0, 0)
            self.line11.set_color(0, 0, 0)
            self.line12.set_color(0,0,0)

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(self.line4)
            self.viewer.add_geom(self.line5)
            self.viewer.add_geom(self.line6)
            self.viewer.add_geom(self.line7)
            self.viewer.add_geom(self.line8)
            self.viewer.add_geom(self.line9)
            self.viewer.add_geom(self.line10)
            self.viewer.add_geom(self.line11)
            self.viewer.add_geom(self.line12)
            self.viewer.add_geom(self.kulo1)
            self.viewer.add_geom(self.kulo2)
            self.viewer.add_geom(self.kulo3)
            self.viewer.add_geom(self.kulo4)
            self.viewer.add_geom(self.kulo5)
            self.viewer.add_geom(self.kulo6)
            self.viewer.add_geom(self.kulo7)
            self.viewer.add_geom(self.gold)
            self.viewer.add_geom(self.robot)

        if self.state is None:
            return None
        
        #根据当前状态确定机器人中心坐标
        self.robotrans.set_translation(self.x[self.state], self.y[self.state])
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')



if __name__ == '__main__':
    env = GridEnv_com()

    env.reset()
    while True:
        env.render()