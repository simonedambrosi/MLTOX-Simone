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
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score, mean_squared_error


from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem




def load_data_rasar(DATA_PATH, encoding, seed = 42):
    
    db = pd.read_csv(DATA_PATH).drop(columns = ['Unnamed: 0', 'test_cas']).drop_duplicates()
    
    # bonds_number and Mol have been minmax transformed yet
    non_categorical = ['obs_duration_mean', 'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'MorganDensity', 'LogP',
                       'oh_count']
    
    # MinMax trasform for numerical variables
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
    
    
    # Ordinal Encoding for categorical variables
    encoder = OrdinalEncoder(dtype = int)
    
    encoder.fit(db[['conc1_type', 'exposure_type', 'control_type', 'media_type', 'application_freq_unit',
                    'class', 'tax_order', 'family', 'genus', 'species']])

    db[['conc1_type', 'exposure_type', 'control_type', 'media_type', 'application_freq_unit',
                    'class', 'tax_order', 'family', 'genus', 'species']] = encoder.transform(
        db[['conc1_type', 'exposure_type', 'control_type', 'media_type', 'application_freq_unit',
                    'class', 'tax_order', 'family', 'genus', 'species']])+1
    
    # Encoding for target variable: binary and multiclass
    if encoding == 'binary':
        db['conc1_mean'] = np.where(db['conc1_mean'].values > 1, 1, 0)

    elif encoding == 'multiclass':
        t = db['conc1_mean'].copy()
        db['conc1_mean'] = multiclass_encoding(t)
    
    
    X = db.drop(columns = 'conc1_mean')
    y = db['conc1_mean'].values

    return X, y


def hamming_matrix(X, cat_features):
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    return squareform(pdist(pd.DataFrame(pd.DataFrame(X['pubchem2d'].values).apply(
        lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list()),  metric = 'hamming'))

def euc_ham_pub_matrix(X, num_features, cat_features, a_ham, a_pub):
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features) + a_pub * pubchem2d_matrix(X)

################################# 1-NN with LogisticRegression ~ 1-NN ###########################################

def nn1_rasar():
    X_try, X_train, X_test, y_train, y_test, len_X_train = load_data_rasar('data/lc_db_processed.csv',
                                                                       encoding = 'binary', seed = 42)
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
               'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    print('Computing distance matrix...', ctime())
    dist_matr = euc_ham_pub_matrix(X_try, non_categorical, categorical, 
                               a_ham = 0.009473684210526315, a_pub = 0.007105263157894737)
    
    dist_matr = pd.DataFrame(dist_matr)
    
    # CV da fare??
    dist_matr_train = dist_matr.iloc[:len_X_train,:len_X_train]
    dist_matr_test = dist_matr.iloc[len_X_train:,:len_X_train]
    
    print('Computing 1-NN...', ctime())
    neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2, leaf_size=40)
    neigh.fit(dist_matr_train, y_train)
    
    nn1 = pd.concat([pd.DataFrame(X_test.index, columns = ['idx_test']),
                     pd.DataFrame(neigh.kneighbors(dist_matr_test, return_distance = False),
                              columns = ['idx_neigh_train'])],
                    axis = 1)
    nn1 = pd.concat([nn1, pd.DataFrame(y_train[nn1.idx_neigh_train], columns = ['label_train'])], axis = 1)
    print(ctime())
    return nn1

##################################### 5-NN with LogisticRegression #####################################

def nn5_rasar(X, y):
        
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
               'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, 
                                         a_ham = 0.014399033208816327, a_pub = 0.001920141938638803)    
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        train_5nn, test_5nn = df_5nn_rasar(dist_matr_train, dist_matr_test, y_train)
        
        lrc = LogisticRegression(random_state=0, fit_intercept = False, solver = 'saga', penalty = 'elasticnet',
                        l1_ratio = 1)
        lrc.fit(train_5nn, y_train)
        y_pred = lrc.predict(test_5nn)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        
    print('...END Simple RASAR', ctime())
    
    avg_accs = np.mean(accs)
    se_accs = sem(accs)
    
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
Specificity: \t {}, se: {}'''.format(avg_accs, se_accs, avg_rmse, se_rmse, avg_sens, se_sens,
                                     avg_precs, se_precs, avg_specs, se_specs))
    return 

def df_5nn_rasar(train_matrix, test_matrix, y_train):
    neigh5 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=5, n_jobs=-2, leaf_size=60)
    neigh5.fit(train_matrix, y_train)
    
    ######## DF Train #########
    tmp_train = pd.DataFrame(neigh5.kneighbors(train_matrix, return_distance = False),
                             columns = ['neigh1', 'neigh2', 'neigh3', 'neigh4', 'neigh5'])
    
    train_5nn = pd.DataFrame({'label_neigh1': y_train[tmp_train.neigh1], 'label_neigh2': y_train[tmp_train.neigh1],
                       'label_neigh3': y_train[tmp_train.neigh3], 'label_neigh4': y_train[tmp_train.neigh4],
                       'label_neigh5': y_train[tmp_train.neigh5]}).T.apply(pd.value_counts).T.fillna(0)

    train_5nn.columns = ['label0_count', 'label1_count']
    
    ######## DF Test ##########
    tmp_test = pd.DataFrame(neigh5.kneighbors(test_matrix, return_distance = False),
                       columns = ['neigh1', 'neigh2', 'neigh3', 'neigh4', 'neigh5'])

    test_5nn = pd.DataFrame({'label_neigh1': y_train[tmp_test.neigh1], 'label_neigh2': y_train[tmp_test.neigh1],
                       'label_neigh3': y_train[tmp_test.neigh3], 'label_neigh4': y_train[tmp_test.neigh4],
                       'label_neigh5': y_train[tmp_test.neigh5]}).T.apply(pd.value_counts).T.fillna(0)

    test_5nn.columns = ['label0_count', 'label1_count']
    
    return train_5nn, test_5nn



############################################## Simple RASAR ##############################################



def right_neighbor(neighbors, X_train, y_train, y_check):
    # IDX Neighbors
    idx_neigh_0 = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train.iloc[y_train==y_check].iloc[x].name)
    idx_neigh_1 = pd.DataFrame(neighbors[1])[1].apply(lambda x: X_train.iloc[y_train==y_check].iloc[x].name)
    
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
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, leaf_size = 40,
                                n_neighbors = 2)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_train_train_0, return_distance = True)
    idx_neigh0, dist0 = right_neighbor(neigh0, X_train, y_train, 0)
    
    # finding the nearest 1s experiments for training experiments that is not itself
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, leaf_size = 80,
                                n_neighbors=2)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_train_train_1, return_distance = True)
    idx_neigh1, dist1 = right_neighbor(neigh1, X_train, y_train, 1)
    
    df_rasar_train = pd.DataFrame({'idx_train': X_train.index.values,
                      'label_train': y_train,
                      'idx_neigh0': idx_neigh0.values,
                      'dist_neigh0': dist0,       
                      'idx_neigh1': idx_neigh1.values,
                      'dist_neigh1': dist1})
    
    ####### DF test RASAR ################
    
    # finding the nearest 0s experiments to test data
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2, leaf_size=40)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_test_train_0, return_distance = True)
    idx_neigh_0 = pd.DataFrame(neigh0[1])[0].apply(lambda x: X_train.iloc[y_train==0].iloc[x].name)

    # finding the nearest 1s experiments to test data
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2, leaf_size=40)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_test_train_1, return_distance = True)
    idx_neigh_1 = pd.DataFrame(neigh1[1])[0].apply(lambda x: X_train.iloc[y_train==1].iloc[x].name)
    
    df_rasar_test = pd.DataFrame({'idx_test': X_test.index.values,
                  'label_test': y_test,
                  'idx_neigh0': idx_neigh_0.values,
                  'dist_neigh0': neigh0[0].ravel(),
                  'idx_neigh1': idx_neigh_1.values,
                  'dist_neigh1': neigh1[0].ravel()})
    
    return df_rasar_train, df_rasar_test


def cv_simple_rasar(X, y):
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
           'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, 
                               a_ham = 0.009473684210526315, a_pub = 0.007105263157894737)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    sens = []
    precs = []
    specs = []
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        rasar_train, rasar_test = unsuper_simple_rasar(dist_matr_train, dist_matr_test, X.iloc[train_index], X.iloc[test_index],
                                             y_train, y_test)
        
        lrc = LogisticRegression(random_state=0, fit_intercept = False, solver = 'saga', penalty = 'elasticnet',
                        l1_ratio = 1)
        lrc.fit(rasar_train[['dist_neigh0','dist_neigh1']], y_train)
        y_pred = lrc.predict(rasar_test[['dist_neigh0','dist_neigh1']])
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        
    print('...END Simple RASAR', ctime())
    
    avg_accs = np.mean(accs)
    se_accs = sem(accs)
    
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
Specificity: \t {}, se: {}'''.format(avg_accs, se_accs, avg_rmse, se_rmse, avg_sens, se_sens,
                                     avg_precs, se_precs, avg_specs, se_specs))
    
    
    return avg_accs, se_accs, avg_rmse, se_rmse

