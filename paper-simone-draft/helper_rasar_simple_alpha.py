# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:13:21 2020

@author: Simone
"""

import pandas as pd
import numpy as np
from time import ctime
from math import sqrt

from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score, f1_score

from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem



def load_data_rasar(DATA_PATH, encoding, seed = 42):
    
    db = pd.read_csv(DATA_PATH).drop(columns = ['Unnamed: 0', 'test_cas','smiles'])
    
    numerical = ['atom_number', 'bonds_number','Mol', 'MorganDensity', 'LogP',
            'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'oh_count', 'MeltingPoint', 'WaterSolubility']
    
    # Categoriche + obs_duration_mean (già numeri)
    categorical = ['conc1_type', 'exposure_type', 'control_type', 'media_type',
               'application_freq_unit', 'species', 'class', 'tax_order', 'family', 'genus']

    
    # MinMax trasform for numerical variables
    for nc in numerical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
    
    
    # Ordinal Encoding for categorical variables
    encoder = OrdinalEncoder(dtype = int)
    encoder.fit(db[categorical])
    db[categorical] = encoder.transform(db[categorical])+1
    
    
    # Apro i pubchem
    db = pd.concat([db, pd.DataFrame(pd.DataFrame(db['pubchem2d'].values).\
                                     apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '), axis = 1)[0].to_list(),
                   columns = ['pub'+ str(i) for i in range(1,882)])], axis = 1)
    
    
    db.drop(columns = ['fish'], inplace = True)
    
    # Encoding for target variable: binary and multiclass
    if encoding == 'binary':
        db['conc1_mean'] = np.where(db['conc1_mean'].values > 1, 0, 1)

    elif encoding == 'multiclass':
        t = db['conc1_mean'].copy()
        db['conc1_mean'] = multiclass_encoding(t)
    
    X = db.drop(columns = 'conc1_mean')
    y = db['conc1_mean'].values

    # splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    # ricongiungo train con test
    X_try = X_train.append(X_test)

    # tengo traccia della lunghezza del train set
    len_X_train = len(X_train)

    return X_try, X_train, X_test, y_train, y_test, len_X_train



def hamming_matrix(X, cat_features):
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    return squareform(pdist(X[X.columns[X.columns.str.contains('pub')]],  metric = 'hamming'))

def euc_ham_pub_matrix(X, num_features, cat_features, a_ham, a_pub):
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features) + a_pub * pubchem2d_matrix(X)

def ham_pub_matrix(X, cat_features = [], a_ham = 0, a_pub = 0):
    return a_ham * hamming_matrix(X, cat_features) + a_pub * pubchem2d_matrix(X)

############################################## Simple RASAR ##############################################



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


def unsuper_simple_rasar(train_distance_matrix, test_distance_matrix, X_train, X_test, y_train, y_test):
    ######## starting DATAFRAME ##########
    
    X_train0 = X_train[y_train == 0].copy()
    X_train1 = X_train[y_train == 1].copy()
    
    # in order to train 1-NN
    dist_matr_train_0 = train_distance_matrix.iloc[y_train == 0, y_train == 0]
    dist_matr_train_1 = train_distance_matrix.iloc[y_train == 1, y_train == 1]
    
    # To find neighbors for train experiments --> df_rasar_train
    dist_matr_train_train_0 = train_distance_matrix.iloc[:,y_train == 0]
    dist_matr_train_train_1 = train_distance_matrix.iloc[:,y_train == 1]
    
    # To find neighbors for test experiments --> df_rasar_test
    dist_matr_test_train_0 = test_distance_matrix.iloc[:, y_train == 0]
    dist_matr_test_train_1 = test_distance_matrix.iloc[:, y_train == 1]
    
    ####### DF train RASAR ###############
    
    # finding the nearest 0s experiments for training experiments that is not itself
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 2)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_train_train_0, return_distance = True)
    _, dist0 = right_neighbor(neigh0, X_train, X_train0)
    
    # finding the nearest 1s experiments for training experiments that is not itself
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors=2)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_train_train_1, return_distance = True)
    _, dist1 = right_neighbor(neigh1, X_train, X_train1)
    
    df_rasar_train = pd.DataFrame({'dist_neigh0': dist0, 'dist_neigh1': dist1})
    
    ####### DF test RASAR ################
    
    # finding the nearest 0s experiments to test data
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_test_train_0, return_distance = True)
#     idx_neigh_0 = pd.DataFrame(neigh0[1])[0].apply(lambda x: X_train.iloc[y_train==0].iloc[x].name)

    # finding the nearest 1s experiments to test data
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_test_train_1, return_distance = True)
#     idx_neigh_1 = pd.DataFrame(neigh1[1])[0].apply(lambda x: X_train.iloc[y_train==1].iloc[x].name)
    
    df_rasar_test = pd.DataFrame({'dist_neigh0': neigh0[0].ravel(), 'dist_neigh1': neigh1[0].ravel()})
    
    return df_rasar_train, df_rasar_test



def cv_simple_rasar_alpha(X, y, hyper_params = dict()):
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP', 'MeltingPoint', 'WaterSolubility']
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 3456)
    
    
    print('Computing distance matrix', ctime())
    ham_pub_matr = ham_pub_matrix(X, categorical, a_ham = 0.07498942093324558 , a_pub = 0.5623413251903491)
    
    euc_matrix = euclidean_matrix(X, non_categorical)
    euc_matrix = pd.DataFrame(euc_matrix)

    accs = []
    sens = []
    specs = []
    f1s = []
    
    i = 0
    for train_index, test_index in kf.split(euc_matrix):
        
        i+=1
        max_euc = euc_matrix.iloc[train_index, train_index].values.max()
        
        print('Epoch {}: '.format(i), ctime())
        dist_matr = pd.DataFrame(ham_pub_matr + euc_matrix.divide(max_euc).values)
    
        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        del dist_matr
    
        rasar_train, rasar_test = unsuper_simple_rasar(X_train, X_test,
                                                       X.iloc[train_index], X.iloc[test_index],
                                                       y_train, y_test)
        
        lrc = LogisticRegression(n_jobs = -1)
        
        for k,v in hyper_params.items():
            setattr(lrc, k, v)

        lrc.fit(rasar_train[['dist_neigh0','dist_neigh1']], y_train)
        y_pred = lrc.predict(rasar_test[['dist_neigh0','dist_neigh1']])
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        f1s.append(f1_score(y_test, y_pred))
        
    print('...END Simple RASAR', ctime())
    
    print('Accuracy:   ', np.mean(accs),  'se:', sem(accs))
    print('Sensitivity:', np.mean(sens),  'se:', sem(sens))
    print('Specificity:', np.mean(specs), 'se:', sem(specs))
    print('F1 score:   ', np.mean(f1s),   'se:', sem(f1s))
    
    return




############## USE ONLY TRAINING SET AS X AND y ##################

def cv_params_simple_rasar_alpha(X, y, hyper_params_tune):
    
    params_comb = list(ParameterSampler(hyper_params_tune, n_iter = 150, random_state = 52))

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
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP', 'MeltingPoint', 'WaterSolubility']
    
    ham_pub_matr = ham_pub_matrix(X, categorical, a_ham = 0.07498942093324558 , a_pub = 0.5623413251903491)
    
    euc_matrix = euclidean_matrix(X, non_categorical)
    euc_matrix = pd.DataFrame(euc_matrix)
    

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    n_epoch = 0
    for train_index, test_index in kf.split(X):

        n_epoch+=1
        print('Epoch {}: '.format(n_epoch), ctime())
        
        max_euc = euc_matrix.iloc[train_index, train_index].values.max()
        
        
        dist_matr = pd.DataFrame(ham_pub_matr + euc_matrix.divide(max_euc).values)
    
        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        del dist_matr
    
        rasar_train, rasar_test = unsuper_simple_rasar(X_train, X_test,
                                                       X.iloc[train_index], X.iloc[test_index],
                                                       y_train, y_test)

        for i in range(0, len(params_comb)):     
            lrc = LogisticRegression(n_jobs = -1)

            for k,v in params_comb[i].items():
                setattr(lrc, k, v)

            lrc.fit(rasar_train[['dist_neigh0','dist_neigh1']], y_train)
            y_pred = lrc.predict(rasar_test[['dist_neigh0','dist_neigh1']])

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

    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0,len(params_comb))])
    tab_lr_rasar = pd.concat([params_df, tab_lr_rasar], axis = 1)
    
    return tab_lr_rasar


    
    
    
    
