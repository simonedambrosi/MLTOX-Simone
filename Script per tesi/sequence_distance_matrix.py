# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:01:15 2020

@author: Simone
"""
import numpy as np
import pandas as pd
from math import sqrt
from time import ctime

from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

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
    return squareform(pdist(pd.DataFrame(pd.DataFrame(X['pubchem2d'].values).apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list()),  metric = 'hamming'))

def tanimoto_matrix(X):
    return squareform(GetTanimotoDistMat([FingerprintMols.FingerprintMol(MolFromSmiles(X.smiles[i]))
                                                 for i in range(len(X.smiles))]))

'''
Compute distances.
'''

def basic_matrix(X, cat_features = [], num_features = [],
                   a_ham = 0.0016102620275609393):
    
    # Hamming and Euclidean
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features)
    
def euc_pub_matrix(X, num_features = [], a_pub = 0):
    return euclidean_matrix(X, num_features) + a_pub * pubchem2d_matrix(X)

def euc_tan_matrix(X, num_features = [], a_tan = 0):
    return euclidean_matrix(X, num_features) + a_tan * tanimoto_matrix(X)

'''
Cross-Validation to find the best combination of alphas and hyperparameters of KNN
'''


def cv_params(X, y, categorical, non_categorical, sequence_pub = [], sequence_tan = [],
                     choice = [0,0], ks = range(1,6,2), leaf_size = range(30, 101, 10)):
    
    np.random.seed(123)
    print(ctime())
    print('START...')
    best_accuracy = 0
    # best_sterror = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0

    print('Computing Basic Matrix: Hamming 1 and Euclidean 2...')
    basic_mat = basic_matrix(X, categorical, non_categorical)

    if choice == [1,0]:
        
        for ap in sequence_pub:
            print('\n', ctime())
            print('Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(ap))
            dist_matr = ap * pubchem2d_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            print(ctime())
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    rmse = []
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
                        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
                        
                    avg_acc = np.mean(accs)
                    se_acc = sem(accs)
                    
                    avg_rmse = np.mean(rmse)
                    se_rmse = sem(rmse)
                    if (avg_acc > best_accuracy):
                        print('''New best params found! alpha:{}, k:{}, leaf:{},
                                                        acc:  {}, st.error:  {},
                                                        rmse: {}, st.error:  {}'''.format(ap, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = ap
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls
                        # best_sterror = sem(accs)
        
        #### TANIMOTO DA SISTEMARE #########
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
        
    
    print(ctime())
    return best_accuracy, best_alpha, best_k, best_leaf



def cv_params_new(X,y, categorical,non_categorical,
                  sequence_pub = [], sequence_tan = [], sequence_ham = [],
                  choice = [0,0,0],
                  ks = range(1,6,2), leaf_size = range(30, 101, 10),
                  a_ham = 0,  a_pub = 0, a_tan = 0):
    
    '''
    choice: primo elemento--> ottimizzare hamming
            secondo elemento --> ottimizzare pubchem2d
            terzo elemento --> ottimizzare tanimoto
    '''
    np.random.seed(123)
    print(ctime())
    print('START...')
    best_accuracy = 0
    # best_sterror = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    if choice == [1,0,0]:
        print('Computing Euclidean and Pubchem2d Matrix...')
        basic_mat = euc_pub_matrix(X, non_categorical, a_pub)
        
        for ah in sequence_ham:
            print('\n', ctime())
            print('Adding Hamming 1 (Categorical)... alpha = {}'.format(ah))
            dist_matr = ah * pubchem2d_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            print(ctime())
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    rmse = []
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
                        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
                        
                    avg_acc = np.mean(accs)
                    se_acc = sem(accs)
                    
                    avg_rmse = np.mean(rmse)
                    se_rmse = sem(rmse)
                    if (avg_acc > best_accuracy):
                        print('''New best params found! alpha:{}, k:{}, leaf:{},
                                                        acc:  {}, st.error:  {},
                                                        rmse: {}, st.error:  {}'''.format(ah, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = ah
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls


    elif choice ==  [0,1,0]:
        print('Computing Basic Matrix: Hamming 1 and Euclidean 2...')
        basic_mat = basic_matrix(X, categorical, non_categorical, a_ham)
        
        for ap in sequence_pub:
            print('\n', ctime())
            print('Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(ap))
            dist_matr = ap * pubchem2d_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            print(ctime())
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    rmse = []
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
                        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
                        
                    avg_acc = np.mean(accs)
                    se_acc = sem(accs)
                    
                    avg_rmse = np.mean(rmse)
                    se_rmse = sem(rmse)
                    if (avg_acc > best_accuracy):
                        print('''New best params found! alpha:{}, k:{}, leaf:{},
                                                        acc:  {}, st.error:  {},
                                                        rmse: {}, st.error:  {}'''.format(ap, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = ap
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls
        
    elif choice == [0,0,1]:
        print('Computing Euclidean and Tanimoto Matrix...')
        basic_mat = euc_tan_matrix(X, non_categorical, a_tan)

        for at in sequence_tan:
            print('\n', ctime())
            print('Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(at))
            dist_matr = at * pubchem2d_matrix(X)
            dist_matr += basic_mat
            dist_matr = pd.DataFrame(dist_matr)
            print(ctime())
            print('Start CV...')
            for k in ks:
                for ls in leaf_size:
                    
                    kf = KFold(n_splits=5, shuffle=True)
                    accs = []
                    rmse = []
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
                        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
                        
                    avg_acc = np.mean(accs)
                    se_acc = sem(accs)
                    
                    avg_rmse = np.mean(rmse)
                    se_rmse = sem(rmse)
                    if (avg_acc > best_accuracy):
                        print('''New best params found! alpha:{}, k:{}, leaf:{},
                                                        acc:  {}, st.error:  {},
                                                        rmse: {}, st.error:  {}'''.format(at, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = at
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls

    
    return best_accuracy, best_alpha, best_k, best_leaf







