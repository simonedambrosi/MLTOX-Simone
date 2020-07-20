# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:01:15 2020

@author: Simone
"""
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""
Building 4 matrices:
    1) Hamming on categorical: 
        inputs: DataFrame, List of categorical features
    
    2) Euclidean on numerical:
        inputs: DataFrame, List of numerical features
    
    3) Hamming on pubchem2d:
        input: DataFrame containing a column of pubchem2d
    
    4) Tanimoto:
        input: DataFrame contaning a column of SMILES
"""

def hamming_matrix(X, cat_features):
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    a = np.array((X['pubchem2d'].iloc[0].replace('', ' ').strip().split(' '),
                  X['pubchem2d'].iloc[1].replace('', ' ').strip().split(' ')))
    
    for i in range(2,len(X.pubchem2d)):
        a = np.concatenate((a,[X.pubchem2d.iloc[i].replace('', ' ').strip().split(' ')]))

    pub_matrix = squareform(pdist(a, metric = 'hamming'))
    
    return pub_matrix

def tanimoto_matrix(X):
    return squareform(GetTanimotoDistMat([FingerprintMols.FingerprintMol(MolFromSmiles(X.smiles[i]))
                                                 for i in range(len(X.smiles))]))

'''
Compute distances 1 and 2.
'''

def basic_matrix(X, cat_features = [], num_features = [],
                   a_ham = 0.0016102620275609393):
    
    # Hamming and Euclidean
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features)
    
'''
Cross-Validation to find the best combination of alphas and hyperparameters of KNN
'''


def cv_params(X, y, categorical, non_categorical, sequence_pub = [], sequence_tan = [],
                     choice = [0,0], ks = range(1,6,2), leaf_size = range(30, 101, 10)):
    
    np.random.seed(123)
    
    print('START...')
    best_accuracy = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    print('Computing Basic Matrix: Hamming 1 and Euclidean 2...')
    basic_mat = basic_matrix(X, categorical, non_categorical)
    
    if choice == [1,0]:
        
        for ap in sequence_pub:
            print('\n Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(ap))
            
            dist_matr = ap * pubchem2d_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    for train_index, test_index in kf.split(dist_matr):
                
                        X_train = dist_matr.iloc[train_index, train_index]
                        X_test = dist_matr.iloc[test_index, train_index]
                        y_train = y[train_index]
                        y_test = y[test_index]
                        
                        neigh = KNeighborsClassifier(metric = 'precomputed',
                                                     n_neighbors=k, n_jobs=-2,
                                                     leaf_size=ls)
                        neigh.fit(X_train, y_train.ravel())
                        y_pred = neigh.predict(X_test)

                        accs.append(accuracy_score(y_test, y_pred))

                    avg_acc = np.mean(accs)
                    if (avg_acc > best_accuracy):
                        print("New best params found! alpha:{}, k:{}, leaf:{}, acc:{}".format(ap, k, ls,
                                                                                                  avg_acc))
                        best_alpha = ap
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls
      
    elif choice == [0,1]:
        
        for at in sequence_tan:
            print('\n Adding Tanimoto... alpha = {}'.format(at))
            dist_matr = at * tanimoto_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    for train_index, test_index in kf.split(dist_matr):
                
                        X_train = dist_matr.iloc[train_index, train_index]
                        X_test = dist_matr.iloc[test_index, train_index]
                        y_train = y[train_index]
                        y_test = y[test_index]
                        
                        neigh = KNeighborsClassifier(metric = 'precomputed',
                                                     n_neighbors=k, n_jobs=-2,
                                                     leaf_size=ls)
                        neigh.fit(X_train, y_train.ravel())
                        y_pred = neigh.predict(X_test)

                        accs.append(accuracy_score(y_test, y_pred))

                    avg_acc = np.mean(accs)
                    if (avg_acc > best_accuracy):
                        print("New best params found! alpha:{}, k:{}, leaf:{}, acc:{}".format(at, k, ls,
                                                                                                  avg_acc))
                        best_alpha = at
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls
        
    
    
    return best_accuracy, best_alpha, best_k, best_leaf





















