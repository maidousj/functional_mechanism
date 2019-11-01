import numpy as np
import scipy.optimize as op
import ipdb
import data_process

def noised_cost_func(w, c1, c2):
    return np.dot(np.dot(w.T, c2),w) + np.dot(c1.T,w)

def fm_logistic(X, y, eps):
    rows, dims = X.shape
    sensitivity = (1./4)*dims*dims+dims

    r0 = (1./8)*np.dot(X.T, X)
    if eps!=0:
        c2 = r0 + np.random.laplace(0, sensitivity/eps, size=(dims,dims))
    else:
        c2 = r0
    c2 = 0.5*(c2.T+c2) # for symmetric
    if eps!=0:
        c2 = c2 + 5.*np.sqrt(2)*(sensitivity/eps)*np.eye(dims) #Regularization

    r1 = np.dot(X.T, (0.5-y))
    if eps!=0:
        c1 = r1 + np.random.laplace(0, sensitivity/eps, dims)
    else:
        c1 = r1

    # calculate eignvalue
    val, vec = np.linalg.eig(c2)

    flag = val<1e-8

    need_del = np.where(flag == True)
    
    val = np.delete(val, need_del, axis=0)   
#    val = np.delete(val, need_del, axis=1)   
    val = np.diag(val)

    vec = np.delete(vec, need_del, axis=1)

#    c1 = np.delete(c1, need_del, axis=0)
    c1 = c1.reshape(c1.shape[0],1) # reshape to (dims,1)
    
    c2 = val
    c1 = np.dot(vec.T, c1)

    w0 = np.random.rand(dims-np.size(need_del),1)
    #result = op.minimize(fun=noised_cost_func, x0=w0, args=(c1, c2), method='TNC')
    result = op.minimize(fun=noised_cost_func, x0=w0, args=(c1, c2), method='L-BFGS-B')
    if result.success:
        result = np.dot(vec, result.x) 
    else:
        result = None
        return
    w = result[:-1]
    b = result[-1]
    return w, b

if __name__ == "__main__":
    filepath = "/criteo/experiment/Functional_Mechanism/Data/Data2_Logistic.dat"
    X,y = data_process.load_data(filepath, minmax=(-1,1), bias_term=True)
    w,b = fm_logistic(X, y, 1.0)
    print(w,b)
