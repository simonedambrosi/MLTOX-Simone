# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:35:44 2020

@author: Simone
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from math import sqrt
from time import ctime
from scipy.stats import sem
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    return squareform(pdist(X[X.columns[X.columns.str.contains('pub')]],  metric = 'hamming'))

def cv_knn(X, y, ks = range(1,10,2), leaf_size = range(10, 101, 10)):
    best_accuracy = 0
    for k in ks:
        for ls in leaf_size:
            accs = []
            rmse = []
            for i in range(1,5):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

                knn = KNeighborsClassifier(n_neighbors = k, leaf_size = ls, n_jobs = -2)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                accs.append(accuracy_score(y_test, y_pred))
                rmse.append(sqrt(mean_squared_error(y_test, y_pred)))

            avg_acc = np.mean(accs)
            se_acc = sem(accs)

            avg_rmse = np.mean(rmse)
            se_rmse = sem(rmse)
            if (avg_acc > best_accuracy):
                print('''New best params found! k:{}, leaf:{}, acc:  {}, st.error:  {},rmse: {}, st.error:  {}'''\
                      .format(k, ls, avg_acc, se_acc, avg_rmse, se_rmse))
                best_accuracy = avg_acc
                best_k = k
                best_leaf = ls

    return best_k, best_leaf


def knn_metrics(X, y, k, ls):
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    for i in range(1,5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
        knn = KNeighborsClassifier(n_neighbors = k, leaf_size = ls, n_jobs = -2)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
    
    avg_acc = np.mean(accs)
    se_acc = sem(accs)
    
    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)
    
    avg_sens = np.mean(sens)
    se_sens = sem(sens)
    
    avg_precs = np.mean(precs)
    se_precs = sem(precs)
    
    avg_specs = np.mean(specs)
    se_specs = sem(specs)
    
    print('''Accuracy: \t {}, se: {}
RMSE: \t \t {}, se: {}
Sensitivity: \t {}, se: {}
Precision: \t {}, se: {}
Specificity: \t {}, se: {}'''.format(avg_acc, se_acc, avg_rmse, se_rmse, avg_sens, se_sens,
                                     avg_precs, se_precs, avg_specs, se_specs))
    
    return

def rf_metrics(X, y):
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    for i in range(1,5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
        rfc = RandomForestClassifier(n_jobs = -2)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
    
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
    
    avg_acc = np.mean(accs)
    se_acc = sem(accs)
    
    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)
    
    avg_sens = np.mean(sens)
    se_sens = sem(sens)
    
    avg_precs = np.mean(precs)
    se_precs = sem(precs)
    
    avg_specs = np.mean(specs)
    se_specs = sem(specs)
    
    print('''Accuracy: \t {}, se: {}
RMSE: \t \t {}, se: {}
Sensitivity: \t {}, se: {}
Precision: \t {}, se: {}
Specificity: \t {}, se: {}'''.format(avg_acc, se_acc, avg_rmse, se_rmse, avg_sens, se_sens,
                                     avg_precs, se_precs, avg_specs, se_specs))
    
    return


def knn(X,y, a_pub, k, leaf_size):

    best_k = k
    best_leaf = leaf_size

    print('Basic Matrix...', ctime())
    dist_matr = euclidean_matrix(X, X.columns[~X.columns.str.contains('pub')])
    print('Adding pubchem2d', ctime())
    dist_matr += a_pub * pubchem2d_matrix(X)
    dist_matr = pd.DataFrame(dist_matr)
    print('End distance matrix...', ctime())
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    for train_index, test_index in kf.split(dist_matr):

        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]

        neigh = KNeighborsClassifier(metric = 'precomputed',
                                     n_neighbors=best_k, n_jobs=-1,
                                     leaf_size=best_leaf)
        neigh.fit(X_train, y_train.ravel())
        y_pred = neigh.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))

        del X_train, X_test, y_train, y_test

    avg_acc = np.mean(accs)
    se_acc = sem(accs)

    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)

    avg_sens = np.mean(sens)
    se_sens = sem(sens)

    avg_precs = np.mean(precs)
    se_precs = sem(precs)

    avg_specs = np.mean(specs)
    se_specs = sem(specs)

    print('''Accuracy: \t {}, se: {}
    RMSE: \t\t {}, se: {}
    Sensitivity: \t {}, se: {}
    Precision: \t {}, se: {}
    Specificity: \t {}, se: {}'''.format(avg_acc, se_acc, avg_rmse, se_rmse, avg_sens, se_sens,
                                     avg_precs, se_precs, avg_specs, se_specs))
    return

def cv_params_knn(X, y, sequence_pub,
                  ks= range(1,10,2), leaf_size = range(10, 101, 10)):
    print(ctime())
    print('START...')
    best_accuracy = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    print('Computing Euclidean Matrix...')
    basic_mat = euclidean_matrix(X, X.columns[~X.columns.str.contains('pub')])
    print('Computing Pubchem...')
    pub_mat = pubchem2d_matrix(X)
    
    for ap in sequence_pub:
        print('Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(ap))
        dist_matr = ap * pub_mat
        dist_matr += basic_mat
        dist_matr = pd.DataFrame(dist_matr)
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
    print(ctime())
    return best_accuracy, best_alpha, best_k, best_leaf
