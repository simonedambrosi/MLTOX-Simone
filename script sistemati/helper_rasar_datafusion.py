# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:40:59 2020

@author: Simone D'Ambrosi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import ctime
from math import sqrt
from collections import Counter

from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score, mean_squared_error

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import sem

import h2o
from h2o.estimators import H2ORandomForestEstimator



def load_datafusion_datasets(DATA_MORTALITY_PATH, DATA_OTHER_ENDPOINT_PATH, encoding = 'binary', seed = 42):
    
    db_datafusion = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(columns = 'Unnamed: 0')
    db_mortality = pd.read_csv(DATA_MORTALITY_PATH).drop(columns = 'Unnamed: 0')   
    
    # bonds_number and Mol have been minmax transformed yet
    non_categorical = ['obs_duration_mean', 'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'MorganDensity', 'LogP',
                       'oh_count']
    
    # MinMax trasform for numerical variables -- MORTALITY
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db_mortality[[nc]])
        db_mortality[[nc]] = minmax.transform(db_mortality[[nc]])
    
    # OTHER ENDPOINT/EFFECT
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db_datafusion[[nc]])
        db_datafusion[[nc]] = minmax.transform(db_datafusion[[nc]])
    
    categorical = ['conc1_type', 'exposure_type', 'control_type', 'media_type', 'application_freq_unit',
                'class', 'tax_order', 'family', 'genus', 'species']
    
    db_mortality, db_datafusion = encoding_categorical(categorical, db_mortality, db_datafusion)
    
    if encoding == 'binary':
        db_mortality['conc1_mean'] = np.where(db_mortality['conc1_mean'].values > 1, 1, 0)
        
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean']
            db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean'] = np.where(conc > np.median(conc), 1, 0)
            
    elif encoding == 'multiclass':
        h2o.init()
        t = db_mortality['conc1_mean'].copy()
        db_mortality['conc1_mean'] = multiclass_encoding(t)
        
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean'].copy()
            db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean'] = multiclass_encoding(conc.values, 
                                                                                              conc.quantile([.2,.4,.6,.8]).values)
    
    return db_mortality, db_datafusion

def encoding_categorical(cat_cols, database, database_df):
    categories = []
    for column in database[cat_cols]:
        cat_db = np.array(sorted(database[column].unique()))
        cat_datafusion = np.array(sorted(database_df[column].unique()))
        for i in cat_datafusion:
            if i not in cat_db:
                cat_db = np.append(cat_db, i)

        categories.append(cat_db)
    
    encoder = OrdinalEncoder(categories = categories, dtype = int)

    encoder.fit(database[cat_cols])
    database[cat_cols] = encoder.transform(database[cat_cols])+1

    encoder.fit(database_df[cat_cols])
    database_df[cat_cols] = encoder.transform(database_df[cat_cols])+1
    
    return database, database_df


def find_similar_exp(exp_mortality, db_datafusion, compare_features):
    out = db_datafusion.conc1_mean[(db_datafusion[compare_features] == exp_mortality[compare_features]).all(axis = 1)].values
    try:
        return -1 if Counter(out).most_common(1)[0][0] == 0 else 1
    except:
        return 0


def hamming_matrix_datafusion(X_mor, X_fus, cat_features):
    return cdist(X_mor[cat_features], X_fus[cat_features], metric = "hamming")

def euclidean_matrix_datafusion(X_mor, X_fus, num_features):
    return cdist(X_mor[num_features], X_fus[num_features], metric = "euclidean")

def pubchem2d_matrix_datafusion(X_mor, X_fus):
    df_pub_mor = pd.DataFrame(pd.DataFrame(X_mor['pubchem2d'].values).apply(
        lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list())
    
    df_pub_fus = pd.DataFrame(pd.DataFrame(X_fus['pubchem2d'].values).apply(
        lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list())
    
    return cdist(df_pub_mor, df_pub_fus, metric = 'hamming')

def euc_ham_pub_matrix_datafusion(X_mor, X_fus, num_features, cat_features, a_ham, a_pub):
    return a_ham * hamming_matrix_datafusion(X_mor, X_fus, cat_features) +\
                   euclidean_matrix_datafusion(X_mor, X_fus, num_features) +\
           a_pub * pubchem2d_matrix_datafusion(X_mor, X_fus)


def hamming_matrix(X, cat_features):
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    return squareform(pdist(pd.DataFrame(pd.DataFrame(X['pubchem2d'].values).apply(
        lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list()),  metric = 'hamming'))

def euc_ham_pub_matrix(X, num_features, cat_features, a_ham, a_pub):
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features) + a_pub * pubchem2d_matrix(X)



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


# unsupervised step of simple rasar
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



def unsuper_datafusion_rasar(db_mortality_train, db_mortality_test, db_datafusion,
                             alpha_1 = 0.009473684210526315, alpha_3 = 0.007105263157894737):
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
               'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
               'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    comparing = ['test_cas', 'obs_duration_mean', 'conc1_type', 'exposure_type', 'control_type', 'media_type',
             'application_freq_unit', 'class', 'tax_order', 'family', 'genus', 'species']
    
    db_datafusion_rasar_train = pd.DataFrame()
    db_datafusion_rasar_test = pd.DataFrame()
    
    for endpoint in db_datafusion.endpoint.unique():
        
        db_endpoint = db_datafusion[db_datafusion.endpoint == endpoint]
        
        for effect in db_endpoint.effect.unique():
            
            db_end_eff_0 = db_endpoint[(db_endpoint.effect == effect) & (db_endpoint.conc1_mean == 0)]
            db_end_eff_1 = db_endpoint[(db_endpoint.effect == effect) & (db_endpoint.conc1_mean == 1)]
            
            # in order to train 1-NN
            train_matrix_0 = euc_ham_pub_matrix(db_end_eff_0, non_categorical, categorical,
                                                a_ham = alpha_1,
                                                a_pub = alpha_3)
            train_matrix_1 = euc_ham_pub_matrix(db_end_eff_1, non_categorical, categorical,
                                                a_ham = alpha_1,
                                                a_pub = alpha_3)
            
            knn0 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 1)
            knn0.fit(pd.DataFrame(train_matrix_0), np.repeat(0, train_matrix_0.shape[0]))
            
            knn1 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 1)
            knn1.fit(pd.DataFrame(train_matrix_1), np.repeat(1, train_matrix_1.shape[0]))
            
            ####################
            ########### DF-TRAIN
            ####################
            
            train_test_matrix_0 = euc_ham_pub_matrix_datafusion(db_mortality_train, db_end_eff_0,
                                                                non_categorical, categorical,
                                                                a_ham = alpha_1,
                                                                a_pub = alpha_3)
            
            neigh0 = knn0.kneighbors(train_test_matrix_0, return_distance = True)
            
            
            train_test_matrix_1 = euc_ham_pub_matrix_datafusion(db_mortality_train, db_end_eff_1,
                                                                non_categorical, categorical,
                                                                a_ham = alpha_1,
                                                                a_pub = alpha_3)
            
            neigh1 = knn1.kneighbors(train_test_matrix_1, return_distance = True)
            
            db_datafusion_rasar_train[endpoint + '_' + effect + '_0'] = neigh0[0].ravel()
            db_datafusion_rasar_train[endpoint + '_' + effect + '_1'] = neigh1[0].ravel()
            
            # FINDING LABELS
            db_datafusion_rasar_train[endpoint+ '_' + effect + '_label'] = db_mortality_train.apply(lambda x: find_similar_exp(
                x, db_endpoint[db_endpoint.effect == effect], comparing), axis = 1).reset_index(drop = True)
            
            
            ###################
            ########### DF-TEST
            ###################
            
            test_test_matrix_0 = euc_ham_pub_matrix_datafusion(db_mortality_test, db_end_eff_0,
                                                               non_categorical, categorical,
                                                               a_ham = alpha_1,
                                                               a_pub = alpha_3)
            
            neigh0 = knn0.kneighbors(test_test_matrix_0, return_distance = True)
            
            test_test_matrix_1 = euc_ham_pub_matrix_datafusion(db_mortality_test, db_end_eff_1,
                                                               non_categorical, categorical,
                                                               a_ham = alpha_1,
                                                               a_pub = alpha_3)
            
            neigh1 = knn1.kneighbors(test_test_matrix_1, return_distance = True)
            
            db_datafusion_rasar_test[endpoint + '_' + effect + '_0'] = neigh0[0].ravel()
            db_datafusion_rasar_test[endpoint + '_' + effect + '_1'] = neigh1[0].ravel()
            
            # FINDING LABELS
            db_datafusion_rasar_test[endpoint+ '_' + effect + '_label'] = db_mortality_test.apply(lambda x: find_similar_exp(
                x, db_endpoint[db_endpoint.effect == effect], comparing), axis = 1).reset_index(drop = True)
            
            
    return db_datafusion_rasar_train, db_datafusion_rasar_test


###############################################################################################################
################################################### BINARY ####################################################
############################### Using sklearn.ensemble.RandomForestClassifier #################################
###############################################################################################################


def cv_datafusion_rasar(X, y, db_datafusion, params = {}):
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
        
        print('Train and test...', end = '')
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar(dist_matr_train, dist_matr_test, X.iloc[train_index],
                                                                     X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar(X.iloc[train_index], X.iloc[test_index], db_datafusion)
                
        train_rf = pd.concat([simple_rasar_train[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_test], axis = 1)
        
        print('done... model...', end ='')
        
        if params == {}:
            rfc = RandomForestClassifier(n_estimators = 300, n_jobs = -2)
        else:
            rfc = RandomForestClassifier(n_jobs = -1)
            for k,v in params.items():
                setattr(rfc, k, v)
        
        rfc.fit(train_rf, y_train)
        y_pred = rfc.predict(test_rf)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        sens.append(recall_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        
        print('done')
        
    print('...END DataFusion RASAR', ctime())
    
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


def cv_depth_datafusion_rasar(X, y, db_datafusion, max_depth_list = []):
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
           'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    output_train_acc = dict()
    output_train_rmse = dict()
    output_test_acc = dict()
    output_test_rmse = dict()
    
    for md in max_depth_list:
        output_train_acc[md] = list()
        output_train_rmse[md] = list()
        output_test_acc[md] = list()
        output_test_rmse[md] = list()
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, 
                               a_ham = 0.009473684210526315, a_pub = 0.007105263157894737)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_index, test_index in kf.split(distance_matrix):
        
        
        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar(dist_matr_train, dist_matr_test, X.iloc[train_index],
                                                                     X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar(X.iloc[train_index], X.iloc[test_index],
                                                                                 db_datafusion)
                
        train_rf = pd.concat([simple_rasar_train[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_test], axis = 1)
        
        print('Train and test done... models...', end = '')
        
        for md in max_depth_list:
            rfc = RandomForestClassifier(n_estimators = 300, max_depth = md, n_jobs = -1)
            rfc.fit(train_rf, y_train)
            y_train_pred = rfc.predict(train_rf)
            y_test_pred = rfc.predict(test_rf)
            
            output_train_acc[md].append(accuracy_score(y_train, y_train_pred))
            output_train_rmse[md].append(sqrt(mean_squared_error(y_train, y_train_pred)))
            
            output_test_acc[md].append(accuracy_score(y_test, y_test_pred))
            output_test_rmse[md].append(sqrt(mean_squared_error(y_test, y_test_pred)))
        
        print('done')
        
    print('...END CV', ctime())
    
    return output_train_acc, output_train_rmse, output_test_acc, output_test_rmse


def cv_random_datafusion_rasar(X, y, db_datafusion, params_dict = dict(), n_models = 100, seed = 42):
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
           'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    params_comb = list(ParameterSampler(params_dict, n_iter = n_models, random_state = seed))
    
    output_train_acc = dict()
    output_train_rmse = dict()
    output_test_acc = dict()
    output_test_rmse = dict()

    for i in range(0,len(params_comb)):
        output_train_acc['mod' + str(i)] = list()
        output_train_rmse['mod' + str(i)] = list()
        output_test_acc['mod' + str(i)] = list()
        output_test_rmse['mod' + str(i)] = list()
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, 
                               a_ham = 0.009473684210526315, a_pub = 0.007105263157894737)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    
    for train_index, test_index in kf.split(distance_matrix):
        
        
        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar(dist_matr_train, dist_matr_test, X.iloc[train_index],
                                                                     X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar(X.iloc[train_index], X.iloc[test_index], db_datafusion)
                
        train_rf = pd.concat([simple_rasar_train[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test[['dist_neigh0', 'dist_neigh1']], datafusion_rasar_test], axis = 1)
        
        print('Train and test done... models...', end = '')
        
        for i in range(0, len(params_comb)):
            
            rfc = RandomForestClassifier(n_jobs = -1)
        
            for k,v in params_comb[i].items():
                setattr(rfc, k, v)
            
            rfc.fit(train_rf, y_train)
            y_train_pred = rfc.predict(train_rf)
            y_test_pred = rfc.predict(test_rf)

            output_train_acc['mod' + str(i)].append(accuracy_score(y_train, y_train_pred))
            output_train_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_train, y_train_pred)))

            output_test_acc['mod' + str(i)].append(accuracy_score(y_test, y_test_pred))
            output_test_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_test, y_test_pred)))

        print('done')
        
    print('...END CV', ctime())
    
    return params_comb, output_train_acc, output_train_rmse, output_test_acc, output_test_rmse

