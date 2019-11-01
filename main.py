from sklearn.model_selection import RepeatedKFold
import data_process
import fm_logistic
import evluate

def test(filepath):
    K = 5  # 5-folds cross-validation
    cv_rep = 10
    eps = 1.0
    
    X, y = data_process.load_data(filepath, minmax=(-1,1), bias_term=True)
    rkf = RepeatedKFold(n_splits=K, n_repeats=cv_rep)
    errSum = 0.
    for train_index, test_index in rkf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index, :-1], y[test_index]

        w, b = fm_logistic.fm_logistic(train_X, train_y, eps)
        errorRate = evluate.rightNum(test_X, test_y, w, b)/len(test_y)

        errSum += errorRate

    print(errSum/(K*cv_rep))

if __name__ == "__main__":
    filepath = "/criteo/experiment/Functional_Mechanism/Data/Data2_Logistic.dat"
    test(filepath)

