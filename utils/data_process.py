import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
import ipdb

def load_data(filepath, minmax=None, normalize=False, bias_term=False):
    lines = np.loadtxt(filepath)
    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape

    if minmax is not None:
        minmax = MinMaxScaler(feature_range=minmax, copy=False)
        minmax.fit_transform(features)

    if normalize:
        # make sure each entry's L2 norm is 1
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(features)

    if bias_term:
        X = np.hstack([features, np.ones(shape=(N, 1))])
    else:
        X = features

    return X, labels

def test(filepath):
    X,y = load_data(filepath, minmax=(-1,1), bias_term=True)
    norm = np.linalg.norm(X, axis=1)
    print("# of examples whose norm is already < 1 : ", \
        np.count_nonzero(norm < 1))
    print("Min values")
    print(X.min(axis=0))

    print("\nMax values")
    print(X.max(axis=0))
    print("l2 norm=", np.linalg.norm(X[:5, 1:], axis=1))

if __name__ == "__main__":
    filepath = "/criteo/experiment/Functional_Mechanism/Data/tmp.txt"
    test(filepath)
