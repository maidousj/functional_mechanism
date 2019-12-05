import numpy as np
import pandas as pd
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import ipdb

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

def hypothesis(w, X):
    z = np.dot(X, w.T)
    return z, sigmoid(z)

def update(X, y, w, eta):
    #ipdb.set_trace()
    batch = len(y)
    step = eta/batch

    gradient = cost_prime(X, y, w)
    w -= step * gradient
    return w

def cost(X, y, w):
    m = len(y)
    z, h = hypothesis(w, X)
    loss = np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    loss = -1.0/m * loss
    return loss

def cost_prime(X, y, w):
    z, h = hypothesis(w, X)
#    error = np.sum((h-y)*sigmoid_prime(z))
    error = np.sum((h-y)*X)
    return error

def logistic_regression(X, y, eta, max_iter, mini_batch=None):
    m, dims = X.shape
    w = np.random.randn(1, dims)
#    w = np.array([[.0,.0]])

    for i in range(max_iter):
        if(mini_batch != None):
            n = len(X)
            for j in range(0, n, mini_batch):
                X_batch = X[j : j+mini_batch]
                y_batch = y[j : j+mini_batch]
                w = update(X_batch, y_batch, w, eta)
        else:
            w = update(X, y, w, eta)
        if (i+1)%100 == 0:
            print('w = ', w)
            print('loss = ', cost(X, y, w)) 
    return w
        
if __name__ == "__main__":
    #data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    df = pd.read_csv("data.csv", header=0)
    df.columns = ["grade1","grade2","label"]
    X = df[["grade1","grade2"]]
    X = min_max_scaler.fit_transform(X)
    y = df["label"].map(lambda x: float(x.rstrip(';')))    
    y = np.array(y)
    y = y.reshape(y.shape[0],1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)

    eta = 0.1
    max_iter = 1000
    w = logistic_regression(X_train, y_train, eta, max_iter, mini_batch=10)
    
    z, y_hat = hypothesis(w, X_test)
    y_hat[y_hat>=0.5] = 1
    y_hat[y_hat<0.5] = 0

    print("zeros in y ", np.sum(y_test==0)/len(y_test))
    print("acc rate = ",np.sum(y_hat==y_test)/len(y_test))
