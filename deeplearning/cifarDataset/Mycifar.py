# -*-coding:utf-8
'''
@time: 2018/11/22 
@version:2.0
========================
cifar10数据集
官网：https://www.cs.toronto.edu/~kriz/cifar.html
10个分类：'plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'
训练集
5*10000(3,32,32)
测试集
1*10000(3,32,32)
'''
'''
2019/3/31修改部分代码  论文数据
'''
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from torchsummary import summary
import time
from torchviz import make_dot, make_dot_from_trace

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)



#hyper parameters
EPOCH = 10                   
BATCH = 10                   
LR = 0.001                                          #learning rate

# def unpickle(fileName):
#     with open(fileName,'rb') as fo:
#         data_dict = pickle.load(fo,encoding='bytes')
#     return data_dict


# #open file and return a dictionary
# path = r'data\cifar-10-batches-py\batches.meta'
# data_dict = unpickle(path)
# classes = data_dict[b'label_names']                 #a dictionary stores class name


#0~255 to 0~1.0       0~1 to -1~1      (image-mean)/std
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH,shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH,shuffle=False, num_workers=2)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Modules are added to the container in the order they are passed in
        self.conv1 = nn.Sequential(                 #input shape(3,32,32)
            nn.Conv2d(                              #Convolution layer
                in_channels = 3,                     
                out_channels = 32,
                kernel_size = 3,
                stride=1,                           #step size of movement
                padding=1
            ),                                      #output shape(32,32,32) 
            nn.BatchNorm2d(32),
            nn.ReLU()                              
            )                                           
                                                   
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32,5,1,2),                 #output shape(32,32,32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)                       #output shape(32,16,16)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,7,1,3),                 #output shape(64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,64,9,1,4),                 #output shape(64,16,16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)                       #output shape(64,8,8)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64,128,11,1,5),                 #output shape(128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128,128,13,1,6),                #output shape(128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)                       #output shape(128,4,4)
        )

        #dense
        self.fc1 = nn.Linear(128 * 4 * 4, 200)       #input_feature = 128*4*4 
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(-1, 128 * 4 * 4)               #reshape tensor (16*5*5) -1 :infer from other dim
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out   


def train(optimizer,loss_function,cnn):

    print("start train <50000 samples> >>>")
    for epoch in range(EPOCH):
        time_start = time.time()
        running_loss = 0

        #定义列表存储均值和标准差
        cnn_mean_var = []

        for step, data in enumerate(trainloader,1):#index from 1

            inputs, labels = data
            # inputs= Variable(inputs)                #moder input
            # labels = Variable(labels)               #image real labels
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.cuda()                  #use gpu to compute
            # labels = labels.cuda()

            

            optimizer.zero_grad()                   #clear gradients for this training step
            # print(inputs.shape)
            outputs = cnn(inputs)                   #the nural network output

            loss = loss_function(outputs,labels)    #calculate the error    
            loss.backward()

            optimizer.step()                        #update all parameters

            running_loss += loss.item()             #sum loss data
            if step%1000 ==0:
                time_end = time.time()
                print('epoch:%d/%d | step:%5d | loss :%.3f | time_cost:%.2fs' %(epoch+1,EPOCH,step,running_loss/1000,time_end-time_start))
                loss_list.append(running_loss/1000)
                running_loss = 0
                time_start = time.time()


        print('start compute gradient')   
        #打印参数
        for index,f in enumerate(cnn.parameters()):
            if index==0:
                temp= {}
                temp['type'] = 'conv1'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index ==4:
                temp= {}
                temp['type'] = 'conv2'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index ==8:
                temp= {}
                temp['type'] = 'conv3'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index ==12:
                temp= {}
                temp['type'] = 'conv4'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index ==16:
                temp= {}
                temp['type'] = 'conv5'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index == 20:
                temp= {}
                temp['type'] = 'conv6'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index==24:
                temp= {}
                temp['type'] = 'fc1'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index == 26:
                temp= {}
                temp['type'] = 'fc2'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)
            elif index == 28:
                temp= {}
                temp['type'] = 'fc3'
                temp['mean'] = abs(float(torch.mean(f.grad)))
                temp['var'] = float(torch.var(f.grad))
                cnn_mean_var.append(temp)

            for x in cnn_mean_var:
                save_to_db(epoch,x)



    


    print("Finished train! >>>")

def test(cnn):
    print("start test <10000 samples> <<<")
    correct = 0
    total = 0
    time_start = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # images = images.cuda()
            # labels = labels.cuda()

            outputs = cnn(images)

            _,predicted = torch.max(outputs,1)          #return(data,index) input = outpus.data   dim= 1  
            total += labels.size(0)                     #labels.size(0) = BATCH
            correct += (predicted == labels).sum().item()
    time_end = time.time()
    print('Accuracy of the network on the 10000 test images: %d %% | time_cost:%.2fs' % (
    100 * correct / total,time_end-time_start))
    Acc.append(correct/total)
    print('Test Finished')


def Myimshow(img):
    #!!Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)!!
    img = img * 0.5 +0.5                                 #-1~1 -> 0-1
    npimg = img.numpy()
    #imshow(img)  note: img = (H,W,C)
    plt.imshow(np.transpose(npimg,(1,2,0)))              #reshape from (3,32,32) to (32,32,3)
    plt.show()


def plot_filter1(cnn):
    print("start plot the first filters")
    cnn.cpu()
    weight = cnn.conv1[0].weight.data.numpy()
    weight = weight - np.min(weight)                    # data to >0
    weight = weight/(np.max(weight))                    # data to (0~1)
    plt.figure()
    for idx,filt in enumerate(weight):                  #(channel,adx,adx)
        plt.subplot(4,8,idx+1)
        filt = filt.transpose(1,2,0)                    
        plt.imshow(filt)                                #(adx,adx,channel)
    plt.show()    

def plot_filter2(cnn):
    cnn.cpu()
    weight = cnn.conv6[0].weight.data.numpy()
    plt.figure()
    for idx,filt in enumerate(weight):
        Sum = 0
        for i in range(filt.shape[0]):
            Sum += filt[i,:,:]
        Sum = Sum - np.min(Sum)
        filt = Sum/(np.max(Sum))
        plt.subplot(11,12,idx+1)
        plt.imshow(filt,cmap='gray')
    plt.show() 

def pltline(loss_list,Acc):
    plt.title('<loss function> and <accuracy> change with steps (six layers)')
    index = [x+1 for x in range(len(loss_list))]
    plt.plot(index,loss_list,color = 'green',label = 'loss function')
    plt.plot(index,Acc,color= 'red',label = 'test accuracy')
    plt.legend()                                    #show map

    plt.xlabel('step<1000>')
    plt.ylabel('rate or value') 
    plt.show()

def printmodel(cnn):
    cnn.cpu()
    for data in testloader:
        images,labels = data
        dot = make_dot(cnn(images),params=dict(cnn.named_parameters()))
        dot.view()
        break

#保存参数到数据库
def save_to_db(epoch,x):
    conn = mysql.connector.connect(user='admin', password='123456',host='47.102.143.204',port='3306', database='test')
    cursor = conn.cursor()
    try:
        cursor.execute('insert into cnn(epoch,layer,std,mean)'
                    'values(%s, %s, %s, %s)',(epoch,x['type'],x['var'],x['mean']))
    except Exception as e:
        print(e)
    conn.commit()
    cursor.close()
    conn.close()
        
if __name__=='__main__':
    
    loss_list = []                                  #save loss every 1000 step in a list   
    Acc = []                                        #save accuracy every 1000 step in a list    
    cnn =CNN()
    # cnn.cuda()
    cnn.to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=LR)#update params based on the gradient
    loss_function = nn.CrossEntropyLoss()          #loss function 

    #train the net model
    # time_start_head = time.time()
    # train(optimizer,loss_function,cnn)
    # time_end_head = time.time()
    # print("total train time cost:%.2f"%(time_end_head-time_start_head))
    print('save model params to cnn.pkl')
    torch.save(cnn.state_dict(), 'cnn.pkl')
    # test the net model
    # time_start_head = time.time()
    # test(cnn)
    # time_end_head = time.time()
    # print("total test time cost:%.2f"%(time_end_head-time_start_head))



    # #output each layer's size
    # summary(cnn,(3,32,32))


    #print some of the images from <trainset>
    # dataiter = iter(trainloader)
    # images,labels = dataiter.next()                 #get the first batch data(image,label)
    # print(' '.join('%5s'%classes[labels[j]] for j in range(BATCH)))
    # Myimshow(torchvision.utils.make_grid(images))   #function:(B,C,H,W) ->B*(C,H,W)


    # # torch.save(cnn, 'cnn.pkl')
    # cnn2 = torch.load('./data/cnn.pkl')
    # cnn2.to(device)
    # # plot_filter1(cnn2)
    # plot_filter2(cnn2)
    # # test(cnn2)

    #print curve
    # pltline(loss_list,Acc)

    #print cnn model 
    # printmodel(cnn)