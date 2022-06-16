import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
class Kmeans:
    
    def __init__(self, data, K=2, gamma=(0.99999, 0.999)):
        self.X = data
        self.K = K
        self.centroid = np.zeros((self.K, self.X.shape[1]), dtype=np.float64)
        for k in range(self.K):
            self.centroid[k] = self.X[k]
        self.Y = np.zeros(self.X.shape[0], dtype=np.uint16)
        self.clustering()
        self.m, self.g = np.zeros(self.centroid.shape, dtype=np.float64), np.zeros(self.centroid.shape, dtype=np.float64)
        self.gamma = gamma
        self.optical = None
        self.history = None
        
    def clustering(self):
        for i in range(self.X.shape[0]):
            d1 = np.sum(np.square(self.centroid[0] - self.X[i]))
            self.Y[i] = 0
            for c in range(1, self.centroid.shape[0]):
                d2 = np.sum(np.square(self.centroid[c] - self.X[i]))
                if d1 > d2:
                    d1 = d2
                    self.Y[i] = c
    
    def loss(self):
        loss_temp = 0
        for cc in range(self.centroid.shape[0]):
            X_temp = np.zeros((len(self.Y[self.Y==cc]), self.X.shape[1]), dtype=np.float64)
            idx = 0
            for i in range(self.Y.shape[0]):
                if self.Y[i] == cc:
                    X_temp[idx] = self.X[i]
                    idx += 1
            for cr in range(self.centroid.shape[1]):
                loss_temp += np.sum(np.square(self.centroid[cc,cr] - X_temp[:,cr]))
        return loss_temp
    
    def fit(self, iteration=100, lr=0.0001):
        if self.optical is None:
            self.optical = {'loss': self.loss(), 
                            'centroid': self.centroid,
                            'cluster': self.Y}
        if self.history is None:
            self.history = np.zeros(iteration, dtype=np.float64)
            previous_learning = 0
        elif type(self.history) is np.ndarray:
            previous_learning = self.history.shape[0]
            self.history = np.append(self.history, np.zeros(iteration, dtype=np.float64), axis=0)
            
        # learning
        iteration = iteration+previous_learning
        for step in range(previous_learning, iteration):
            for cc in range(self.centroid.shape[0]):
                X_temp = np.zeros((len(self.Y[self.Y==cc]), self.X.shape[1]), dtype=np.float64)
                idx = 0
                for i in range(self.Y.shape[0]):
                    if self.Y[i] == cc:
                        X_temp[idx] = self.X[i]
                        idx += 1
                for cr in range(self.centroid.shape[1]):
                    dif = np.sum(self.centroid[cc,cr] - X_temp[:,cr])*2
                    self.m[cc,cr] = self.gamma[0]*self.m[cc,cr] + (1-self.gamma[0])*dif
                    mh = self.m[cc,cr]/(1-(self.gamma[0]**(step+1)))
                    dif = np.square(dif)
                    self.g[cc,cr] = self.gamma[1]*self.g[cc,cr] + (1-self.gamma[1])*dif
                    gh = self.g[cc,cr]/(1-(self.gamma[1]**(step+1)))
                    self.centroid[cc,cr] -= lr*mh/np.sqrt(gh+0.00000001)
            self.clustering()
            self.history[step] = self.loss()
            if self.history[step] < self.optical['loss']:
                self.optical['loss'] = self.history[step]
                self.optical['centroid'] = self.centroid.copy()
                self.optical['cluster'] = self.Y.copy()
            print('iteration:\t{}/{}\t\tloss: {}'.format(step+1, iteration, self.history[step]))
            