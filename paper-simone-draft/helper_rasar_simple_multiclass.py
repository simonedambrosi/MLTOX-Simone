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


def unsuper_simple_rasar_multiclass(X_train, X_test, y_train, y_test):
    
    df_simple_train = pd.DataFrame()
    df_simple_test = pd.DataFrame()
    
    for i in range(1,6):
        # in order to train K-NN
        X_train_i = X_train[y_train == i].copy()
        
        ##########################
        ######## DF RASAR -- TRAIN
        ##########################
        
        knn_train = KNeighborsClassifier(n_jobs = -2, leaf_size = 30, n_neighbors = 2)
        knn_train.fit(X_train_i, y_train[y_train == i])
        
        neigh_i = knn_train.kneighbors(X_train, return_distance = True)
        idx_neigh_i, dist_neigh_i = right_neighbor(neigh_i, X_train, X_train_i)
        
        df_simple_train.loc[:, 'LC50_MOR_' + str(i)] = dist_neigh_i        
        
        ##########################
        ######## DF RASAR -- TEST
        ##########################
        
        knn_test = KNeighborsClassifier(n_jobs = -2, leaf_size = 30, n_neighbors = 1)
        knn_test.fit(X_train_i, y_train[y_train == i])
        
        neigh_i = knn_test.kneighbors(X_test, return_distance = True)
        
        df_simple_test.loc[:, 'LC50_MOR_' + str(i)] = neigh_i[0].ravel()
        
    return df_simple_train, df_simple_test

def cv_simple_rasar_multiclass(X, y, hyper_params = dict()):
    
    accs = []

    sens_micro = []
    sens_macro = []
    sens_weight = []

    precs_micro = []
    precs_macro = []
    precs_weight = []

    f1s_micro = []
    f1s_macro = []
    f1s_weight = []

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    print(ctime())
    for train_index, test_index in kf.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        df_rasar_train, df_rasar_test = unsuper_simple_rasar_multiclass(X_train, X_test, y_train, y_test)
        
        lrc = LogisticRegression(n_jobs = -1)
#         lrc = RandomForestClassifier(n_jobs = -1)
        
        for k,v in hyper_params.items():
            setattr(lrc, k, v)
        
        lrc.fit(df_rasar_train, y_train)
        y_pred = lrc.predict(df_rasar_test)

        accs.append(accuracy_score(y_test, y_pred))
    
        sens_micro.append(recall_score(y_test, y_pred, average = 'micro'))
        sens_macro.append(recall_score(y_test, y_pred, average = 'macro'))
        sens_weight.append(recall_score(y_test, y_pred, average = 'weighted'))

        precs_micro.append(precision_score(y_test, y_pred, average = 'micro'))
        precs_macro.append(precision_score(y_test, y_pred, average = 'macro'))
        precs_weight.append(precision_score(y_test, y_pred, average = 'weighted'))

        f1s_micro.append(f1_score(y_test, y_pred, average = 'micro'))
        f1s_macro.append(f1_score(y_test, y_pred, average = 'macro'))
        f1s_weight.append(f1_score(y_test, y_pred, average = 'weighted'))

    print(ctime())
    print('Accuracy:            ', np.mean(accs), 'se:', sem(accs), end = '\n\n')

    print('Micro Recall:        ', np.mean(sens_micro), 'se:', sem(sens_micro))
    print('Macro Recall:        ', np.mean(sens_macro), 'se:', sem(sens_macro))
    print('Weighted Recall:     ', np.mean(sens_weight), 'se:', sem(sens_weight), end = '\n\n')

    print('Micro Precision:     ', np.mean(precs_micro), 'se:', sem(precs_micro))
    print('Macro Precision:     ', np.mean(precs_macro), 'se:', sem(precs_macro))
    print('Weighted Precision:  ', np.mean(precs_weight), 'se:', sem(precs_weight), end = '\n\n')

    print('Micro F1:            ', np.mean(f1s_micro), 'se: ', sem(f1s_micro))
    print('Macro F1:            ', np.mean(f1s_macro), 'se: ', sem(f1s_macro))
    print('Weighted F1:         ', np.mean(f1s_weight), 'se: ', sem(f1s_weight))
        
    return

##################################################################
############## USE ONLY TRAINING SET AS X AND y ##################
##################################################################

def cv_params_simple_rasar_multiclass(X, y, hyper_params_tune):
    
    params_comb = list(ParameterSampler(hyper_params_tune, n_iter = 100, random_state = 52))

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)

    test_acc = dict()
    test_sens = dict()
    test_prec = dict()
    test_f1 = dict()


    for i in range(0,len(params_comb)):
        test_acc['mod' + str(i)] = list()
        test_sens['mod' + str(i)] = list()
        test_prec['mod' + str(i)] = list()
        test_f1['mod' + str(i)] = list()

    print(ctime())

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)

    for train_index, test_index in kf.split(X):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        df_rasar_train, df_rasar_test = unsuper_simple_rasar_multiclass(X_train, X_test, y_train, y_test)

        for i in range(0, len(params_comb)):
#             lrc = RandomForestClassifier(n_jobs = -1)
            lrc = LogisticRegression(n_jobs = -1)

            for k,v in params_comb[i].items():
                setattr(lrc, k, v)

            lrc.fit(df_rasar_train, y_train)
            y_pred = lrc.predict(df_rasar_test)

            test_acc['mod' + str(i)].append(accuracy_score(y_test, y_pred))
            test_sens['mod' + str(i)].append(recall_score(y_test, y_pred, average = 'macro'))
            test_prec['mod' + str(i)].append(precision_score(y_test, y_pred, average = 'macro'))
            test_f1['mod' + str(i)].append(f1_score(y_test, y_pred, average = 'macro'))

    print(ctime())

    tab_lr_rasar = pd.DataFrame(columns = ['test_acc', 'test_sens', 'test_prec', 'test_f1'])

    tab_lr_rasar.loc[:,'test_acc'] = pd.DataFrame(test_acc).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_sens'] = pd.DataFrame(test_sens).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_prec'] = pd.DataFrame(test_prec).mean(axis = 0)
    tab_lr_rasar.loc[:,'test_f1'] = pd.DataFrame(test_f1).mean(axis = 0)

    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0,100)])
    tab_lr_rasar = pd.concat([params_df, tab_lr_rasar], axis = 1)
    
    return tab_lr_rasar




