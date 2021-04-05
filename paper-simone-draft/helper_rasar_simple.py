# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:01:07 2021

@author: Simone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import ctime
from math import sqrt
from collections import Counter

from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score, mean_squared_error, f1_score

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import sem


def right_neighbor(neighbors, X_train, X_train_i):
    # IDX Neighbors
    idx_neigh_0 = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train_i.iloc[x].name)
    idx_neigh_1 = pd.DataFrame(neighbors[1])[1].apply(lambda x: X_train_i.iloc[x].name)
    
    idx_neigh = idx_neigh_0.copy()
    
    # dove l'indice del primo vicino risulta essere uguale a se stesso lo sostituisco con il secondo vicino
    idx_neigh[X_train.index == idx_neigh_0] = idx_neigh_1[X_train.index == idx_neigh_0].values
    
    # Distance from the Nearest Neighbor that is NOT itself
    dist_0 = pd.DataFrame(neighbors[0])[0]
    dist_1 = pd.DataFrame(neighbors[0])[1]
    
    distance = dist_0.copy()
    distance[X_train.index == idx_neigh_0] = dist_1[X_train.index == idx_neigh_0].values
    
    return idx_neigh, distance


def df_train_simple_rasar(X_train, y_train):
    
    X_train0 = X_train[y_train == 0].copy()
    X_train1 = X_train[y_train == 1].copy()
    
    knn0 = KNeighborsClassifier(n_neighbors = 2)
    knn0.fit(X_train0, y_train[y_train == 0])

    neigh_train0 = knn0.kneighbors(X_train, return_distance = True)
    idx_neigh0, dist0 = right_neighbor(neigh_train0, X_train, X_train0)

    knn1 = KNeighborsClassifier(n_neighbors = 2)
    knn1.fit(X_train1, y_train[y_train == 1])

    neigh_train1 = knn1.kneighbors(X_train, return_distance = True)
    idx_neigh1, dist1 = right_neighbor(neigh_train1, X_train, X_train1)

    df_rasar_train = pd.DataFrame({'dist_neigh0': dist0, 'dist_neigh1': dist1, 'label_train': y_train})
    
    return df_rasar_train


def df_test_simple_rasar(X_train, X_test, y_train, y_test):
    
    X_train0 = X_train[y_train == 0].copy()
    X_train1 = X_train[y_train == 1].copy()
    
    knn0 = KNeighborsClassifier(n_neighbors = 1)
    knn0.fit(X_train0, y_train[y_train == 0])
    neigh_test0 = knn0.kneighbors(X_test, return_distance = True)

    knn1 = KNeighborsClassifier(n_neighbors = 1)
    knn1.fit(X_train1, y_train[y_train == 1])
    neigh_test1 = knn1.kneighbors(X_test, return_distance = True)

    df_rasar_test = pd.DataFrame({'dist_neigh0': neigh_test0[0].ravel(), 'dist_neigh1': neigh_test1[0].ravel(),
                                 'label_test': y_test})
    
    return df_rasar_test

def cv_simple_rasar(X, y, hyper_params = dict()):
    
    accs = []
    sens = []
    specs = []
    f1s = []

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    print(ctime())
    for train_index, test_index in kf.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        df_rasar_train = df_train_simple_rasar(X_train, y_train)

        df_rasar_test = df_test_simple_rasar(X_train, X_test, y_train, y_test)
        
        lrc = LogisticRegression(n_jobs = -1)
        
        for k,v in hyper_params.items():
            setattr(lrc, k, v)
        
        lrc.fit(df_rasar_train[['dist_neigh0','dist_neigh1']], y_train)
        y_pred = lrc.predict(df_rasar_test[['dist_neigh0','dist_neigh1']])

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        f1s.append(f1_score(y_test, y_pred))

    print(ctime())
    print('Accuracy:   ', np.mean(accs),  'se:', sem(accs))
    print('Sensitivity:', np.mean(sens),  'se:', sem(sens))
    print('Specificity:', np.mean(specs), 'se:', sem(specs))
    print('F1 score:   ', np.mean(f1s),   'se:', sem(f1s))
        
    return

############## USE ONLY TRAINING SET AS X AND y ##################

def cv_params_simple_rasar(X, y, hyper_params_tune):
    
    params_comb = list(ParameterSampler(hyper_params_tune, n_iter = 100, random_state = 52))

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)

    test_acc = dict()
    test_sens = dict()
    test_spec = dict()
    test_f1 = dict()


    for i in range(0,len(params_comb)):
        test_acc['mod' + str(i)] = list()
        test_sens['mod' + str(i)] = list()
        test_spec['mod' + str(i)] = list()
        test_f1['mod' + str(i)] = list()

    print(ctime())

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)

    for train_index, test_index in kf.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        df_rasar_train = df_train_simple_rasar(X_train, y_train)

        df_rasar_test = df_test_simple_rasar(X_train, X_test, y_train, y_test)

        for i in range(0, len(params_comb)):     
            lrc = LogisticRegression(n_jobs = -1)

            for k,v in params_comb[i].items():
                setattr(lrc, k, v)

            lrc.fit(df_rasar_train[['dist_neigh0','dist_neigh1']], y_train)
            y_pred = lrc.predict(df_rasar_test[['dist_neigh0','dist_neigh1']])

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            test_acc['mod' + str(i)].append(accuracy_score(y_test, y_pred))
            test_sens['mod' + str(i)].append(recall_score(y_test, y_pred))
            test_spec['mod' + str(i)].append(tn/(tn+fp))
            test_f1['mod' + str(i)].append(f1_score(y_test, y_pred))

    print(ctime())

    tab_lr_rasar = pd.DataFrame(columns = ['test_acc', 'test_sens', 'test_spec', 'test_f1'])

    tab_lr_rasar.loc[:,'test_acc'] = pd.DataFrame(test_acc).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_sens'] = pd.DataFrame(test_sens).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_spec'] = pd.DataFrame(test_spec).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_f1'] = pd.DataFrame(test_f1).mean(axis = 0)

    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0,100)])
    tab_lr_rasar = pd.concat([params_df, tab_lr_rasar], axis = 1)
    
    return tab_lr_rasar




