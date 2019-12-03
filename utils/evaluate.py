import numpy as np

def rightNum(testX, testy, w, b):
    pred = np.dot(testX,w) + b
    pred[pred>=0] = 1
    pred[pred<0] = 0
    return np.sum(pred==testy)
    
def rightNum(testX, testy, w):
    pred = np.dot(testX, w.T)
    pred[pred>=0] = 1
    pred[pred<0] = 0
    return np.sum(pred==testy)

def rightNum(testX, testy, w, v):
    pred = np.dot(w, testX.T).T + np.longlong(
           np.sum((np.dot(testX, v) ** 2 - np.dot(testX ** 2, v ** 2)),
                  axis=1).reshape(len(testX), 1)) / 2.0 
    pred[pred>=0] = 1
    pred[pred<0] = 0
    return np.sum(pred==testy)
