import numpy as np

def rightNum(testX, testy, w, b):
    pred = np.dot(testX,w) + b
    pred[pred>=0] = 1
    pred[pred<0] = 0
    return np.sum(pred==testy)
    
