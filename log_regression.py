import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

#import data
df = pd.read_csv('.\supply_chain_train.csv')

#extract x and y
x = df.values[:, :-1]
y = df.values[:, -1]

#drop useless attributes
x = np.delete(x, [0, 1], axis=1)

#drop missing data
idx_delete = [i for i in range(x.shape[0]) if 'Unknown' in x[i, :]]
x = np.delete(x, idx_delete, axis=0)
y = np.delete(y, idx_delete, axis=0)

#encode categorical data
labelencoder_x = LabelEncoder()
x = np.array([labelencoder_x.fit_transform(x[:, i]) for i in range(x.shape[1])]).T

#normalization and standardization
x = (x - x.mean(axis=0)) / x.std(axis=0)

#logistic regression
class logistic_regression():
    #initial
    def __init__(self, x, y, lr, iter, i_th, lamda = 0):
        self.x = x
        self.y = y
        self.lr = lr
        self.sample_num = int(np.floor(x.shape[0]/5))
        self.iter = iter
        self.w = np.ones((x.shape[1], 1))
        self.i_th = i_th
        self.lamda = lamda

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    #BGD method(sample_rate = 1/5)
    def BGD_gradient(self):
        idx = np.random.randint(0, self.x.shape[0], self.sample_num)
        gradient = np.dot(np.transpose(self.x[idx, :]), (self.sigmoid(np.dot(self.x[idx, :], self.w))-self.y[idx].astype('float')))/self.sample_num + self.lamda*self.w.astype('float')
        return gradient

    #SGD method
    def SGD_gradient(self):
        idx = np.random.randint(0, self.x.shape[0])
        gradient = np.transpose(self.x[idx, :])*(self.sigmoid(np.dot(self.x[idx, :], self.w))-self.y[idx].astype('float')) + self.lamda*self.w.astype('float')
        return gradient

    def train(self, accuracy, precision, recall, loss):
        iter_num = self.iter
        tbar = tqdm(range(iter_num))
        num = iter_num
        for _ in tbar:
            tbar.set_description('Training ' + str(self.i_th) + '-th log_reg sub-model')
            self.w = self.w-self.lr*self.BGD_gradient()
            iter_num -= 1
            accuracy[num - iter_num - 1] = self.accuracy(test_x, test_y) + accuracy[num - iter_num - 1]
            precision[num - iter_num - 1] = self.precision(test_x, test_y) + precision[num - iter_num - 1]
            recall[num - iter_num - 1] = self.recall(test_x, test_y) + recall[num - iter_num - 1]
            loss[num - iter_num - 1] = self.loss(test_x, test_y) + loss[num - iter_num - 1]
            sleep(0.01)    

    def accuracy(self, x_input, y_input):
        y_prob = np.array(self.sigmoid(np.dot(x_input, self.w).astype('float')))
        y_pre = np.round(y_prob)
        return sum(y_pre == y_input)/y_input.shape[0]
    
    def precision(self, x_input, y_input):
        y_prob = np.array(self.sigmoid(np.dot(x_input, self.w).astype('float')))
        y_pre = np.round(y_prob)
        tp = sum(np.logical_and(y_pre == 1, y_input == 1))
        fp = sum(np.logical_and(y_pre == 1, y_input == 0))
        return tp/(tp+fp)
    
    def recall(self, x_input, y_input):
        y_prob = np.array(self.sigmoid(np.dot(x_input, self.w).astype('float')))
        y_pre = np.round(y_prob)
        tp = sum(np.logical_and(y_pre == 1, y_input == 1))
        fn = sum(np.logical_and(y_pre == 0, y_input == 1))
        return tp/(tp+fn)
    
    def loss(self, x_input, y_input):
        y_prob = np.array(self.sigmoid(np.dot(x_input, self.w).astype('float')))
        y_pre = np.round(y_prob)
        return abs(y_pre-y_input).sum()/y_input.shape[0]

class regression():
    def __init__(self, x, y, lr, iter, i_th, lamda = 0):
        self.x = x
        self.y = y
        self.lr = lr
        self.iter = iter
        self.sample_num = int(np.floor(x.shape[0]/20))
        self.w = np.ones((x.shape[1], 1))
        self.i_th = i_th
        self.lamda = lamda

    def BGD_gradient(self):
        idx = np.random.randint(0, self.x.shape[0], self.sample_num)
        gradient = -np.mean((self.y[idx, :].astype('float') - np.dot(self.x[idx, :], self.w)))*np.expand_dims(np.mean(self.x[idx, :], axis=0), axis=1) + self.lamda*self.w.astype('float')
        return gradient
    
    def train(self, accuracy, precision, recall, loss):
        iter_num = self.iter
        tbar = tqdm(range(iter_num))
        num = iter_num
        for _ in tbar:
            tbar.set_description('Training ' + str(self.i_th) + '-th regression sub-model')
            self.w = self.w-self.lr*self.BGD_gradient()
            iter_num -= 1
            accuracy[num - iter_num - 1] = self.accuracy(mapping(test_x), test_y) + accuracy[num - iter_num - 1]
            precision[num - iter_num - 1] = self.precision(mapping(test_x), test_y) + precision[num - iter_num - 1]
            recall[num - iter_num - 1] = self.recall(mapping(test_x), test_y) + recall[num - iter_num - 1]
            loss[num - iter_num - 1] = self.loss(test_x, test_y) + loss[num - iter_num - 1]
            sleep(0.01)  

    def accuracy(self, x_input, y_input):
        y_pre = np.array((np.dot(x_input, self.w).astype('float') >= 0.5))
        return sum(y_pre == y_input)/y_input.shape[0]
    
    def precision(self, x_input, y_input):
        y_pre = np.array((np.dot(x_input, self.w).astype('float') >= 0.5))
        tp = sum(np.logical_and(y_pre == 1, y_input == 1))
        fp = sum(np.logical_and(y_pre == 1, y_input == 0))
        return tp/(tp+fp)
    
    def recall(self, x_input, y_input):
        y_pre = np.array((np.dot(x_input, self.w).astype('float') >= 0.5))
        tp = sum(np.logical_and(y_pre == 1, y_input == 1))
        fn = sum(np.logical_and(y_pre == 0, y_input == 1))
        return tp/(tp+fn)      

    def loss(self, x_input, y_input):
        y_pre = np.array(np.dot(x_input, self.w).astype('float'))
        return abs(y_pre-y_input).sum()/y_input.shape[0]

class linear_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, x.shape[1], dtype = torch.float32), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, 1, dtype = torch.float32), requires_grad=True)

    def forward(self, x):
        return torch.matmul(self.w, torch.transpose(x, 0, 1)) + self.bias
    
    def accuracy(self, x_input, y_input):
        y_pre = np.transpose(np.array(((x_input.data.numpy().astype('float')) >= 0.5)))
        return sum(y_pre == y_input.data.numpy())/y_input.data.numpy().shape[0]

    def precision(self, x_input, y_input):
        y_pre = np.transpose(np.array(((x_input.data.numpy().astype('float')) >= 0.5)))
        tp = sum(np.logical_and(y_pre == 1, y_input.data.numpy() == 1))
        fp = sum(np.logical_and(y_pre == 1, y_input.data.numpy() == 0))
        return tp/(tp+fp)
    
    def recall(self, x_input, y_input):
        y_pre = np.transpose(np.array(((x_input.data.numpy().astype('float')) >= 0.5)))
        tp = sum(np.logical_and(y_pre == 1, y_input.data.numpy() == 1))
        fn = sum(np.logical_and(y_pre == 0, y_input.data.numpy() == 1))
        return tp/(tp+fn)      

    def loss(self, x_input, y_input):
        y_pre = np.transpose(np.array(((x_input.data.numpy().astype('float')) >= 0.5)))
        return abs(y_pre-y_input.data.numpy()).sum()/y_input.data.numpy().shape[0]

def mapping(input):
    #return np.concatenate((np.power(input, -1), np.power(input, 3)), axis=1)
    return np.power(input, 1)

if __name__ == '__main__':
    #train and test
    iter_num = 100
    #logistic regression
    w_standard_log_reg = np.zeros((x.shape[1], 1))
    accuracy_log_reg = np.zeros((iter_num, 1))
    precision_log_reg = np.zeros((iter_num, 1))
    recall_log_reg = np.zeros((iter_num, 1))
    loss_log_reg = np.zeros((iter_num, 1))
    #regression
    w_standard_reg = np.zeros((1, mapping(x).shape[1]))
    accuracy_reg = np.zeros((iter_num, 1))
    precision_reg = np.zeros((iter_num, 1))
    recall_reg = np.zeros((iter_num, 1))
    loss_reg = np.zeros((iter_num, 1))

    loop = 10
    #data balance method
    for num in range(loop):    
        #split train and test data
        idx_zero = np.where(y == 0)[0]
        idx_one = np.array([i for i  in range(x.shape[0]) if i not in idx_zero])
        np.random.shuffle(idx_one)
        idx_one = idx_one[:idx_zero.shape[0]]
        x_split_zero, x_split_one, y_split_zero, y_split_one = x[idx_zero], x[idx_one], y[idx_zero], y[idx_one]
        train_zero_x, test_zero_x, train_zero_y, test_zero_y= train_test_split(x_split_zero, y_split_zero, test_size=0.2)
        train_one_x, test_one_x, train_one_y, test_one_y= train_test_split(x_split_one, y_split_one, test_size=0.2)
        train_x, test_x, train_y, test_y = np.concatenate((train_zero_x, train_one_x), axis=0), np.concatenate((test_zero_x, test_one_x), axis=0), np.concatenate((train_zero_y, train_one_y), axis=0), np.concatenate((test_zero_y, test_one_y), axis=0)
        train_y ,test_y = np.expand_dims(train_y, axis=1), np.expand_dims(test_y, axis=1)

        #regression
        '''
        reg = regression(mapping(train_x), train_y, 1, iter_num, num, 0.01)
        reg.train(accuracy_reg, precision_reg, recall_reg, loss_reg)
        print('train accuracy of ' + str(num) + '-th reg sub-model: ', reg.accuracy(mapping(test_x), test_y)[0])
        print('train precision of ' + str(num) + '-th reg sub-model: ', reg.precision(mapping(test_x), test_y)[0])
        print('train recall of ' + str(num) + '-th reg sub-model: ', reg.recall(mapping(test_x), test_y)[0])
        w_standard_reg = w_standard_reg + reg.w
        '''

        model = linear_regression()
        mseloss = nn.MSELoss()
        opt = optim.SGD(model.parameters(), lr=0.005)
        dataset = TensorDataset(torch.from_numpy(mapping(train_x).astype('float32')), torch.from_numpy(train_y.astype('float32')))
        train_dl = DataLoader(dataset, batch_size=16, shuffle=True)
        tbar = tqdm(range(iter_num))
        i = iter_num
        for _ in tbar:
            tbar.set_description('Training ' + str(num) + '-th regression sub-model')
            for xdl, ydl in train_dl:
                pred = model(xdl)
                loss = mseloss(pred, ydl)
                opt.zero_grad()
                loss.backward()
                opt.step()
            i -= 1
            accuracy_reg[iter_num - i - 1] = model.accuracy(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))) + accuracy_reg[iter_num - i - 1]
            precision_reg[iter_num - i - 1] = model.precision(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))) + precision_reg[iter_num - i - 1]
            recall_reg[iter_num - i - 1] = model.recall(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))) + recall_reg[iter_num - i - 1]
            loss_reg[iter_num - i - 1] = loss.data.numpy() + loss_reg[iter_num - i - 1]
        print('train accuracy of ' + str(num) + '-th reg sub-model: ', model.accuracy(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
        print('train precision of ' + str(num) + '-th reg sub-model: ', model.precision(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
        print('train recall of ' + str(num) + '-th reg sub-model: ', model.recall(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
        w_standard_reg = w_standard_reg + model.w.data.numpy()

        #logistic regression
        logi = logistic_regression(train_x, train_y, 1, iter_num, num, 0.00001)
        logi.train(accuracy_log_reg, precision_log_reg, recall_log_reg, loss_log_reg)
        print('train accuracy of ' + str(num) + '-th log_reg sub-model: ', logi.accuracy(test_x, test_y)[0])
        print('train precision of ' + str(num) + '-th log_reg sub-model: ', logi.precision(test_x, test_y)[0])
        print('train recall of ' + str(num) + '-th log_reg sub-model: ', logi.recall(test_x, test_y)[0])
        w_standard_log_reg = w_standard_log_reg + logi.w

    w_standard_log_reg /= loop
    logi.w = w_standard_log_reg
    print('test accuracy of all models of log_reg: ', logi.accuracy(test_x, test_y)[0])
    print('test precision of all models of log_reg: ', logi.precision(test_x, test_y)[0])
    print('test recall of all models of log_reg: ', logi.recall(test_x, test_y)[0])
    plt.plot(accuracy_log_reg/loop, label='accuracy')
    plt.plot(precision_log_reg/loop, label='precision')
    plt.plot(recall_log_reg/loop, label='recall')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('evaluation indicators of logistic regression')
    plt.legend()
    plt.show()
    plt.plot(loss_log_reg/loop, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss of logistic regression')
    plt.legend()
    plt.show()

    model.eval()
    w_standard_reg /= loop
    model.w.data = torch.from_numpy(w_standard_reg.astype('float32'))  
    print('test accuracy of all models of reg: ', model.accuracy(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
    print('test precision of all models of reg: ', model.precision(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
    print('test recall of all models of reg: ', model.recall(model(torch.from_numpy(mapping(test_x).astype('float32'))), torch.from_numpy(test_y.astype('float32'))))
    plt.plot(accuracy_reg/loop, label='accuracy')
    plt.plot(precision_reg/loop, label='precision')
    plt.plot(recall_reg/loop, label='recall')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('evaluation indicators of  regression')
    plt.legend()
    plt.show()
    plt.plot(loss_reg/loop, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss of logistic regression')
    plt.legend()
    plt.show()