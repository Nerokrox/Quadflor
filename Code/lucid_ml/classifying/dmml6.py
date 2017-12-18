from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from classifying.br_kneighbor_classifier import BRKNeighborsClassifier
from classifying.rocchioclassifier import RocchioClassifier
import scipy.sparse as sp
import numpy as np

class DMML6Classifier(BaseEstimator):

    def __init__(self, n_neighbors_knn, use_lsh_forest_knn, auto_optimize_k_knn, roccio_k, weights):
        self.knnaw = 0.75
        self.knnbw = 0.75
        self.rocciow = 0.9
        if len(weights) == 3:
            self.knnaw = weights[0]
            self.knnbw = weights[1]
            self.rocciow = weights[2]
        self.knnA = BRKNeighborsClassifier(mode='b', n_neighbors=n_neighbors_knn, use_lsh_forest=use_lsh_forest_knn,
                                         algorithm='brute', metric='cosine', auto_optimize_k=auto_optimize_k_knn)
        self.knnB = BRKNeighborsClassifier(mode='b', n_neighbors=n_neighbors_knn+2, use_lsh_forest=use_lsh_forest_knn,
                                         algorithm='brute', metric='cosine', auto_optimize_k=auto_optimize_k_knn)
        self.roccioC = RocchioClassifier(metric = 'cosine', k = roccio_k)
    
    def fit(self, X, y):
        self.knnA.fit(X, y)
        self.knnB.fit(X, y)
        self.roccioC.fit(X, y)

    def predict(self, X):
        knnapred = self.knnA.predict(X)
        print(str(knnapred.shape[0]) + " " + str(knnapred.shape[1]))
        knnbpred = self.knnB.predict(X)
        print(str(knnbpred.shape[0]) + " " + str(knnbpred.shape[1]))
        roccioc = self.roccioC.predict(X)
        print(str(roccioc.shape[0]) + " " + str(roccioc.shape[1]))

        compred = self.knnaw * knnapred + self.knnbw * knnbpred + self.rocciow * roccioc
        (rows, cols) = compred.shape
        print(str(rows) + " " + str(cols))
        result = sp.csr_matrix((rows, cols))
        predicted_labels = sp.dok_matrix((rows, cols))
        for i in range(0, rows):
            for j in range(0, cols):
                if compred[i, j] >= 1.5:
                    predicted_labels[i, j] = 1
        predicted_labels = sp.csr_matrix(predicted_labels)
        result = sp.vstack(predicted_labels)
        print(str(result.shape[0]) + " " + str(result.shape[1]))
        return result