import numpy as np
import sklearn
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import random

class MultiGaussClassify:
    def __init__(self, k, d, diag):
        self.k = k
        self.d = d
        self.diag = diag
    def fit(self, X, y):
        X_inprog = []
        y_inprog = []
        self.mean = []
        self.cov = []
        self.prob = []
        for i in range(self.k):
            temp_X = []
            temp_y = []
            class_mean = []
            temp_cov = []
            for j in range(np.shape(X)[0]):
                if(y[j] == i):
                    temp_X.append(X[j])
                    temp_y.append(y[j])
            temp_shape = np.shape(temp_X)
            for a in range(temp_shape[1]):
                s=0
                for b in range(temp_shape[0]):
                    s = s + temp_X[b][a]
                class_mean.append(1/(temp_shape[0])*s)
            for a in range(temp_shape[1]):
                t_cov = []
                for b in range(temp_shape[0]):
                    t_cov.append(temp_X[b][a] - class_mean[a])
                temp_cov.append(t_cov)
            temp_cov_T = np.transpose(temp_cov)
            cov_dot = np.dot(temp_cov,temp_cov_T)
            cov_div = np.divide(cov_dot,(temp_shape[0]-1))
            if(np.linalg.det(cov_div)==0):
                cov_div = np.add(cov_div,0.00001*(np.identity(self.d)))
            X_inprog.append(temp_X)
            y_inprog.append(temp_y)
            self.mean.append(class_mean)
            if(self.diag == True):
                self.cov.append(np.diag(np.diag(cov_div)))
            else:
                self.cov.append(cov_div)
            self.prob.append(np.shape(temp_y)[0]/np.shape(X)[0])
        
    def predict(self, X):
        predicted = np.empty(np.shape(X)[0])
        for j in range(np.shape(X)[0]):
            predicted_prob = []
            for i in range(self.k):
                temp_X = []
                for i1 in range(np.shape(X)[1]):
                    temp_X.append(X[j][i1]-self.mean[i][i1])
                temp_X_T = np.transpose(temp_X)
                cov_inv = np.linalg.inv(self.cov[i])
                det = np.linalg.det(self.cov[i])
                det_abs = np.absolute(det)
                k1 = (self.d/2)*np.log((np.pi)*2)
                k2 = 0.5*np.log(det_abs)
                k3 = np.dot(temp_X_T,cov_inv)
                k4 = 0.5*(np.dot(k3,temp_X))
                k5 = np.log(self.prob[i])
                value = -k1 -k2 -k4 + k5
                predicted_prob.append(value)
            predicted[j] = np.argmax(predicted_prob)
        return predicted
    
def my_cross_val(method,X,y,k):
    
    fold_batch = float(len(y)/k)
    error_rate = np.zeros(k)
    index = list(range(len(X)))
    random.shuffle(index)
    X_mod = [0]*len(X)
    y_mod = [0]*len(X)
    for i in range(len(X)):
        X_mod[i] = X[index[i]]
        y_mod[i] = y[index[i]] 
    for i in range(k):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for j in range(len(y)):
            if(j>= int(i*fold_batch) and j<int((i+1)*fold_batch)):
                test_x.append(X_mod[j])
                test_y.append(y_mod[j])
            else:
                train_x.append(X_mod[j])
                train_y.append(y_mod[j])
        if(method == 'LogisticRegression'):
            model = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
        if(method == 'multigaussclassify'):
            if(np.shape(X)[1]==64):
                model = MultiGaussClassify(10, np.shape(X)[1], False)
            else:
                model = MultiGaussClassify(2, np.shape(X)[1], False)
        if(method == 'multigaussdiagclassify'):
            if(np.shape(X)[1]==64):
                model = MultiGaussClassify(10, np.shape(X)[1], True)
            else:
                model = MultiGaussClassify(2, np.shape(X)[1], True)
        model.fit(train_x,train_y)
        predict = model.predict(test_x)
        error_count = 0
        for a in range(len(test_y)):
            if(predict[a] != test_y[a]):
                error_count += 1
        error_rate[i] = error_count/len(test_y)
    mean = np.mean(error_rate, axis=0)
    sigma = np.std(error_rate)
    return(error_rate,mean,sigma)

def print_table_values(method,dataset,error_rate,mean,std,q):
    print(f'Error rates for {method} with {dataset}')
    filename = method+dataset+q+".txt"
    f = open(filename,'w')
    print(f'Error rates for {method} with {dataset}', file =f)
    for i in range(len(error_rate)):
        print(f'Fold {i+1}: {error_rate[i]}')
    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std}')
    for i in range(len(error_rate)):
        print(f'F{i+1}\t', end="", file =f)
    print('Mean\t', end="", file =f)
    print('SD', file =f)
    for i in error_rate:
        print(f'{i}\t', end="", file =f)
    print(f'{mean}\t', end="", file =f)
    print(f'{std}\t', file =f)
    f.close()
    return