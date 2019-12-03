import numpy as np
import sys
sys.path.append("../utils")
import data_process
import evaluate
import random
import ipdb

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

class QuadraticCost(object):
    @staticmethod
    def fn(a, y): 
        """ 
        平方误差损失函数
        :param a: 预测值
        :param y: 真实值
        :return:
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y): 
        """ 
        损失函数对z求偏导
        :param z: x的线性函数
        :param a:
        :param y:
        :return:
        """
        return (a - y) * sigmoid_prime(z)

class LR(object):
    """
    :param r2: R2小于该值后停止迭代
    """
    def __init__(self, train_X, train_y, valid_X, valid_y, eta, max_iter, r2, cost=QuadraticCost):
        self.train_x = train_X
        self.train_y = train_y
        self.valid_x = valid_X
        self.valid_y = valid_y
        self.var_y = np.var(self.valid_y)
        self.eta = float(eta)
        self.max_iter = max_iter
        self.r2 = r2
        self.cost = cost

        # 正态分布随机初始化参数
        self.w = np.random.randn(1, self.train_x.shape[1])

    def calcR2(self):
        """
        在验证集上计算R2
        """
        y_hat = self.predict(self.valid_x)
        mse = np.sum((y_hat - self.valid_y) ** 2) / len(self.valid_y) 
        r2 = 1.0 - mse / self.var_y
#        print "r2={}".format(r2) 
        return r2

    def predict(self, x):
        z = np.dot(self.w, x.T).T
        return z, sigmoid(z)

    def update_mini_batch(self, x, y, eta):
        """
        平方误差作为损失函数，梯度下降法更新参数
        """
        batch = len(x)
        step = eta/batch

        z, y_hat = self.predict(x)
        y_diff = self.cost.delta(z, y_hat, y) 
        self.w -= step * np.dot(y_diff.T, x)

    def shuffle_data(self):
        """
        每轮训练前随机打乱样本顺序
        """
        ids = list(range(len(self.train_x)))
        random.shuffle(ids)
        self.train_x = self.train_x[ids]
        self.train_y = self.train_y[ids]

    def train(self, mini_batch=100):
        for itr in range(self.max_iter):
            print("iteration={}".format(itr))
            self.shuffle_data()
            n = len(self.train_x)
            for i in range(0, n, mini_batch):
                x = self.train_x[i : i+mini_batch]        
                y = self.train_y[i : i+mini_batch]        
                learn_rate = np.exp(-itr) * self.eta  # 学习率指数递减
                self.update_mini_batch(x, y, learn_rate)

            if self.calcR2() > self.r2: 
                break


def test(filepath):
    # data
    X, y = data_process.load_data(filepath, minmax=(0,1), bias_term=True)
    y = y.reshape((y.shape[0], 1))
    sample, dim = X.shape
    
    train_size = int(0.7 * sample)
    valid_size = int(0.2 * sample)
    
    train_X = X[:train_size]
    valid_X = X[train_size : train_size+valid_size]
    test_X = X[train_size+valid_size:]

    train_y = y[:train_size]
    valid_y = y[train_size : train_size+valid_size]
    test_y = y[train_size+valid_size:]
    
    # parameters
    eta = 0.01
    max_iter = 500
    r2 = 0.9
    batch_size = 50

    # model
    lr = LR(train_X, train_y, valid_X, valid_y, eta, max_iter, r2)
    lr.train(mini_batch=batch_size)

    errorRate = evaluate.rightNum(test_X, test_y, lr.w)/len(test_y)
    print("error rate = ", errorRate)

if __name__ == "__main__":
    filepath = "/criteo/experiment/Functional_Mechanism/Data/Data2_Logistic.dat"
    test(filepath)
