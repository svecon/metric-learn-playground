import csv
import numpy as np
import scipy.linalg as la

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class KNNClassifierPerClass():
    def __init__(self, n_neighbors=1):
        self.k = n_neighbors
    
    def fit(self, X, y):
        self.le = LabelEncoder()
        self.le.fit(y)
        self.y = self.le.transform(y)
        
        self.classes = sorted(np.unique(self.y))
        
        X_c = {}
        Xn_c = {}
        self.knn_c = {}
        self.means_c = {}
        self.std_c = {}
        for c in self.classes:
            X_c[c] = X[ np.where(self.y == c) ]
            self.means_c[c] = X_c[c].mean(axis=0)
            self.std_c[c] = X_c[c].std(axis=0)
            
            self.std_c[c][self.std_c[c] == 0] = 1
            
            self.knn_c[c] = KNeighborsClassifier(n_neighbors=self.k)
            self.knn_c[c].fit((X - self.means_c[c]) / self.std_c[c], self.y)
    
    def predict(self, X, verbose=False):
        predictedYs = np.zeros( (X.shape[0], self.k*len(self.classes)) , dtype=np.int8)
        for i,c in enumerate(self.classes):
            distances, indices = self.knn_c[c].kneighbors( (X-self.means_c[c])/self.std_c[c] )
            predictedYs[:, i*self.k : (i+1)*self.k] = np.vectorize(lambda x: self.y[x])(indices)

#         predictedYMajority = np.zeros(X.shape[0], dtype=np.int8)
        predictedYMajority = []
        
        if verbose: print(predictedYs)
        
        for i in range(X.shape[0]):
#             predictedYMajority[i] = np.bincount(predictedYs[i]).argmax()
            predictedYMajority.append( np.bincount(predictedYs[i]).argmax() )
        
        return self.le.inverse_transform(predictedYMajority)