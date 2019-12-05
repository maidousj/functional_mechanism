import lr2
import sys
sys.path.append("../utils")
import data_process
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    filepath = "/criteo/experiment/Functional_Mechanism/Data/Data2_Logistic.dat"
    #data
    X, y = data_process.load_data(filepath, minmax=(-1,1)) 
    y = y.reshape((y.shape[0], 1))
    sample, dim = X.shape

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

    #params
    eta = 0.1
    max_iter = 400
    batch_size = 50

    #train
    w = lr2.logistic_regression(X_train, y_train, eta, max_iter, mini_batch=batch_size)

    #test
    z, y_hat = lr2.hypothesis(w, X_test)
    y_hat[y_hat>=0.5] = 1
    y_hat[y_hat<0.5] = 0

    print("zeros in y ", np.sum(y_test==0)/len(y_test))
    print("acc rate = ",np.sum(y_hat==y_test)/len(y_test))  
