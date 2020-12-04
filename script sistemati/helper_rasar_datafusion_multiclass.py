# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 21:51:14 2020

@author: Simone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import ctime
from math import sqrt
import math
from collections import Counter
import itertools

from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, precision_score, accuracy_score, mean_squared_error

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import sem

import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

###############################################################################################
####################################  LOADING AND ENCODING ####################################
###############################################################################################

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
#         h2o.init()
#         h2o.no_progress() # disable progress bar during model fitting
        t = db_mortality['conc1_mean'].copy()
        db_mortality['conc1_mean'] = multiclass_encoding(t)
        
        for ef in db_datafusion.effect.unique():
            conc = db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean'].copy()
            db_datafusion.loc[db_datafusion.effect == ef, 'conc1_mean'] = multiclass_encoding(conc.values, 
                                                                                              conc.quantile([.2,.4,.6,.8]).values)
    
    return db_mortality, db_datafusion

###############################################################################################
#########################################  DISTANCES  ######################################### 
###############################################################################################

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

###########################################################################################################
##########################################  CV Data Fusion RASAR ##########################################
###########################################################################################################

def find_similar_exp_multiclass(exp_mortality, db_datafusion, compare_features):
    out = db_datafusion.conc1_mean[(db_datafusion[compare_features] == exp_mortality[compare_features]).all(axis = 1)].values
    try:
        return Counter(out).most_common(1)[0][0]
    except:
        return 'Unknown'


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




def unsuper_simple_rasar_multiclass(train_distance_matrix, test_distance_matrix, X_train, X_test, y_train, y_test):
    
    df_simple_train = pd.DataFrame()
    df_simple_test = pd.DataFrame()
    
    for i in range(1,6):
        # in order to train K-NN
        matr_train_i = train_distance_matrix.loc[y_train == i, y_train == i]
        
        # to find neighbors to train data
        matr_train_train_i = train_distance_matrix.loc[:, y_train == i]
        
        # to find neighbors to test data
        matr_test_train_i = test_distance_matrix.loc[:, y_train == i]
        
        ##########################
        ######## DF RASAR -- TRAIN
        ##########################
        
        knn_train = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, leaf_size = 30, n_neighbors = 2)
        knn_train.fit(matr_train_i, y_train[y_train == i])
        
        neigh_i = knn_train.kneighbors(matr_train_train_i, return_distance = True)
        idx_neigh_i, dist_neigh_i = right_neighbor(neigh_i, X_train, y_train, i)
        
        df_simple_train.loc[:, 'LC50_MOR_' + str(i)] = dist_neigh_i        
        
        ##########################
        ######## DF RASAR -- TEST
        ##########################
        
        knn_test = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, leaf_size = 30, n_neighbors = 1)
        knn_test.fit(matr_train_i, y_train[y_train == i])
        
        neigh_i = knn_test.kneighbors(matr_test_train_i, return_distance = True)
        
        df_simple_test.loc[:, 'LC50_MOR_' + str(i)] = neigh_i[0].ravel()
        
    return df_simple_train, df_simple_test




def unsuper_datafusion_rasar_multiclass(db_mortality_train, db_mortality_test, db_datafusion,
                                        alpha_1 = 2.7825594022071245, alpha_3 = 1000):
    
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
            
            for i in range(1,6):
                
                db_end_eff_i = db_endpoint[(db_endpoint.effect == effect) & (db_endpoint.conc1_mean == i)]
                
                # in order to train 1-NN
                matr_trainfus_i = euc_ham_pub_matrix(db_end_eff_i, non_categorical, categorical, a_ham = alpha_1, a_pub = alpha_3)
                
                knn = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 1)
                knn.fit(pd.DataFrame(matr_trainfus_i), np.repeat(i, matr_trainfus_i.shape[0]))
                
                ##########################
                ######## DF RASAR -- TRAIN
                ##########################
                
                matr_trainmor_trainfus_i = euc_ham_pub_matrix_datafusion(db_mortality_train, db_end_eff_i, non_categorical, categorical, 
                                                                         a_ham = alpha_1, a_pub = alpha_3)
            
                neigh_train_i = knn.kneighbors(matr_trainmor_trainfus_i, return_distance = True)
                                
                db_datafusion_rasar_train[endpoint + '_' + effect + '_'+ str(i)] = neigh_train_i[0].ravel()
                
                ##########################
                ######## DF RASAR -- TEST
                ##########################
                
                matr_testmor_trainfus_i = euc_ham_pub_matrix_datafusion(db_mortality_test, db_end_eff_i, non_categorical, categorical, 
                                                                        a_ham = alpha_1, a_pub = alpha_3)
            
                neigh_test_i = knn.kneighbors(matr_testmor_trainfus_i, return_distance = True)
                
                db_datafusion_rasar_test[endpoint + '_' + effect + '_' + str(i)] = neigh_test_i[0].ravel()
                
                
            db_datafusion_rasar_train[endpoint+ '_' + effect + '_label'] = db_mortality_train.apply(
                lambda x: find_similar_exp_multiclass(x, db_endpoint[db_endpoint.effect == effect], comparing),
                axis = 1).reset_index(drop = True)
                
            db_datafusion_rasar_test[endpoint+ '_' + effect + '_label'] = db_mortality_test.apply(
                lambda x: find_similar_exp_multiclass(x, db_endpoint[db_endpoint.effect == effect], comparing),
                axis = 1).reset_index(drop = True)
    
    return db_datafusion_rasar_train, db_datafusion_rasar_test



def cv_datafusion_rasar_multiclass(X, y, db_datafusion, final_model = False):
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
           'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, a_ham = 2.7825594022071245, a_pub = 1000)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    wrecs = []
    wprecs = []
    for train_index, test_index in kf.split(distance_matrix):
        
        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        print('Train and test...')
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar_multiclass(dist_matr_train, dist_matr_test, X.iloc[train_index], 
                                                                                X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar_multiclass(X.iloc[train_index], X.iloc[test_index], 
                                                                                            db_datafusion, 
                                                                                            alpha_1 = 2.7825594022071245, alpha_3 = 1000)
        
        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis = 1)
        
        train_rf.loc[:, 'target'] = y_train
        test_rf.loc[:, 'target'] = y_test
        
        train_rf_h2o = h2o.H2OFrame(train_rf)
        test_rf_h2o = h2o.H2OFrame(test_rf)
        
        for col in train_rf.columns:
            if 'label' in col:
                train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                test_rf_h2o[col] = test_rf_h2o[col].asfactor()
        
        train_rf_h2o['target'] = train_rf_h2o['target'].asfactor()
        test_rf_h2o['target'] = test_rf_h2o['target'].asfactor()
        
        
        print('done...', ctime(), 'model...')
        
        if not final_model:
            rfc = H2ORandomForestEstimator(categorical_encoding = 'one_hot_explicit', seed = 123)
        else:
            # parameters are cross-validated...
            rfc = H2ORandomForestEstimator(ntrees = 500, sample_rate = 0.6, nbins_cats = 4096, nbins = 64, min_rows = 1,
                                           max_depth = 17, min_split_improvement = 0.0001, histogram_type = 'QuantilesGlobal',
                                           col_sample_rate_per_tree = 0.83, categorical_encoding = 'one_hot_explicit', seed = 123)
                
        rfc.train(y = 'target', training_frame = train_rf_h2o)
        y_pred = rfc.predict(test_rf_h2o).as_data_frame()['predict']
        
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        wrecs.append(recall_score(y_test, y_pred, average = 'weighted'))
        wprecs.append(precision_score(y_test, y_pred, average = 'weighted'))
        
        print('done', ctime())
        
    print('...END DataFusion RASAR', ctime())
    
    avg_accs = np.mean(accs)
    se_accs = sem(accs)
    
    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)
    
    avg_wrecs = np.mean(wrecs)
    se_wrecs = sem(wrecs)
    
    avg_wprecs = np.mean(wprecs)
    se_wprecs = sem(wprecs)
    
    print('''Accuracy: \t {}, se: {}
RMSE: \t\t {}, se: {}
W. Recall: \t {}, se: {}
W. Precision: \t {}, se: {}'''.format(avg_accs, se_accs, avg_rmse, se_rmse, avg_wrecs, se_wrecs,
                                     avg_wprecs, se_wprecs))
    
    return


def cv_depth_datafusion_rasar_multiclass(X, y, db_datafusion, max_depth_list = list()):
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
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, a_ham = 2.7825594022071245, a_pub = 1000)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    wrecs = []
    wprecs = []
    for train_index, test_index in kf.split(distance_matrix):
        
        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        print('Train and test...', ctime(), end = '')
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar_multiclass(dist_matr_train, dist_matr_test, X.iloc[train_index], 
                                                                                X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar_multiclass(X.iloc[train_index], X.iloc[test_index], 
                                                                                            db_datafusion, 
                                                                                            alpha_1 = 2.7825594022071245, alpha_3 = 1000)
        
        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis = 1)
        
        train_rf.loc[:, 'target'] = y_train
        test_rf.loc[:, 'target'] = y_test
        
        train_rf_h2o = h2o.H2OFrame(train_rf)
        test_rf_h2o = h2o.H2OFrame(test_rf)
        
        for col in train_rf.columns:
            if 'label' in col:
                train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                test_rf_h2o[col] = test_rf_h2o[col].asfactor()
        
        train_rf_h2o['target'] = train_rf_h2o['target'].asfactor()
        test_rf_h2o['target'] = test_rf_h2o['target'].asfactor()
        
        print('done... models...', ctime(), end = '')
    
        drf_grid = H2ORandomForestEstimator(max_runtime_secs = 3600, seed = 123, stopping_rounds = 5,
                                    stopping_metric = "RMSE", stopping_tolerance = 1e-4,
                                    categorical_encoding = 'onehotexplicit')
        
        hyper_params = {'max_depth': max_depth_list}
        
        # grid search
        grid = H2OGridSearch(drf_grid, hyper_params, search_criteria = {'strategy': "Cartesian"})
        grid.train(y = 'target' , training_frame = train_rf_h2o)
        
        sorted_grid = grid.get_grid(sort_by = 'rmse', decreasing = True)
        
        for i in range(0, len(max_depth_list)):
            model = h2o.get_model(sorted_grid.sorted_metric_table()['model_ids'][i])
            y_train_pred = model.predict(train_rf_h2o).as_data_frame()['predict'].round()
            y_test_pred = model.predict(test_rf_h2o).as_data_frame()['predict'].round()
            
            output_train_acc[model.actual_params['max_depth']].append(accuracy_score(y_train, y_train_pred))
            output_train_rmse[model.actual_params['max_depth']].append(sqrt(mean_squared_error(y_train, y_train_pred)))
    
            output_test_acc[model.actual_params['max_depth']].append(accuracy_score(y_test, y_test_pred))
            output_test_rmse[model.actual_params['max_depth']].append(sqrt(mean_squared_error(y_test, y_test_pred)))
        
        h2o.remove_all()
    
    print('...END CV', ctime())
    
    tab_depth_rasar = pd.DataFrame(columns = ['train_acc', 'test_acc', 'train_rmse', 'test_rmse'])

    tab_depth_rasar.loc[:,'train_acc'] = pd.DataFrame(output_train_acc).mean(axis = 0)
    tab_depth_rasar.loc[:,'test_acc'] = pd.DataFrame(output_test_acc).mean(axis = 0)
    tab_depth_rasar.loc[:,'train_rmse'] = pd.DataFrame(output_train_rmse).mean(axis = 0)
    tab_depth_rasar.loc[:,'test_rmse'] = pd.DataFrame(output_test_rmse).mean(axis = 0)
    
    tab_depth_rasar = tab_depth_rasar.reset_index().rename(columns = {'index': 'max_depth'})
    
    return tab_depth_rasar


def cv_random_datafusion_rasar_multiclass(X, y, db_datafusion, params_dict = dict(), n_models = 100, seed = 42):
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
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, a_ham = 2.7825594022071245, a_pub = 1000)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    wrecs = []
    wprecs = []
    for train_index, test_index in kf.split(distance_matrix):
        
        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        print('Train and test...', ctime(), end = '')
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar_multiclass(dist_matr_train, dist_matr_test, X.iloc[train_index], 
                                                                                X.iloc[test_index], y_train, y_test)
        
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar_multiclass(X.iloc[train_index], X.iloc[test_index], 
                                                                                            db_datafusion, 
                                                                                            alpha_1 = 2.7825594022071245, alpha_3 = 1000)
        
        train_rf = pd.concat([simple_rasar_train, datafusion_rasar_train], axis = 1)
        test_rf = pd.concat([simple_rasar_test, datafusion_rasar_test], axis = 1)
        
        train_rf.loc[:, 'target'] = y_train
        test_rf.loc[:, 'target'] = y_test
        
        train_rf_h2o = h2o.H2OFrame(train_rf)
        test_rf_h2o = h2o.H2OFrame(test_rf)
        
        for col in train_rf.columns:
            if 'label' in col:
                train_rf_h2o[col] = train_rf_h2o[col].asfactor()
                test_rf_h2o[col] = test_rf_h2o[col].asfactor()
        
        train_rf_h2o['target'] = train_rf_h2o['target'].asfactor()
        test_rf_h2o['target'] = test_rf_h2o['target'].asfactor()
        
        print('done... models...', ctime(), end = '')
    
        for i in range(0, len(params_comb)):
            
            drf = H2ORandomForestEstimator(max_runtime_secs = 3600, seed = 123,
                                    stopping_metric = "RMSE", stopping_tolerance = 1e-4, stopping_rounds = 5,
                                    categorical_encoding = 'onehotexplicit')
            
            for k,v in params_comb[i].items():
                setattr(drf, k, v)
            
            drf.train(y = 'target', training_frame = train_rf_h2o)
            
            y_train_pred = drf.predict(train_rf_h2o).as_data_frame()['predict'].round()
            y_test_pred = drf.predict(test_rf_h2o).as_data_frame()['predict'].round()
            
            output_train_acc['mod' + str(i)].append(accuracy_score(y_train, y_train_pred))
            output_train_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_train, y_train_pred)))
    
            output_test_acc['mod' + str(i)].append(accuracy_score(y_test, y_test_pred))
            output_test_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_test, y_test_pred)))
        
        h2o.remove_all()
            
        print('done')
            
    print('...END CV', ctime())
    
    tab_rasar_final = pd.DataFrame(columns = ['train_acc', 'test_acc', 'train_rmse', 'test_rmse'])

    tab_rasar_final.loc[:,'train_acc'] = pd.DataFrame(output_train_acc).mean(axis = 0)
    tab_rasar_final.loc[:,'test_acc'] = pd.DataFrame(output_test_acc).mean(axis = 0)
    tab_rasar_final.loc[:,'train_rmse'] = pd.DataFrame(output_train_rmse).mean(axis = 0)
    tab_rasar_final.loc[:,'test_rmse'] = pd.DataFrame(output_test_rmse).mean(axis = 0)
    
    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0, len(params_comb))])
    tab_rasar_final = pd.concat([tab_rasar_final, params_df], axis = 1)
    
    return tab_rasar_final


##########################################################################################################
#################################  SIMPLE RASAR -- MULTICLASS  ###########################################
##########################################################################################################

def cv_simple_rasar_multiclass(X, y, final_model = False):
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
           'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                   'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    print('Computing distance matrix...', ctime())
    distance_matrix = euc_ham_pub_matrix(X, non_categorical, categorical, 
                               a_ham = 2.7825594022071245, a_pub = 1000)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    wrecs = []
    wprecs = []
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        rasar_train, rasar_test = unsuper_simple_rasar_multiclass(dist_matr_train, dist_matr_test,
                                                                  X.iloc[train_index], X.iloc[test_index], 
                                                                  y_train, y_test)
        
        
        
        if not final_model:
            lrc = LogisticRegression(multi_class = 'multinomial', max_iter = 10000)
        else:
            lrc = LogisticRegression(multi_class = 'multinomial', max_iter = 20000, solver = 'saga', tol = 1e-5,
                                     penalty = 'elasticnet', fit_intercept = False, l1_ratio = 0.6, n_jobs=-1)
            
        lrc.fit(rasar_train, y_train)
        y_pred = lrc.predict(rasar_test)
        
        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        wrecs.append(recall_score(y_test, y_pred, average = 'weighted'))
        wprecs.append(precision_score(y_test, y_pred, average = 'weighted'))
        
    print('...END Simple RASAR -- Multiclass', ctime())
    
    avg_accs = np.mean(accs)
    se_accs = sem(accs)
    
    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)
    
    avg_wrecs = np.mean(wrecs)
    se_wrecs = sem(wrecs)
    
    avg_wprecs = np.mean(wprecs)
    se_wprecs = sem(wprecs)
    
    print('''Accuracy: \t {}, se: {}
RMSE: \t\t {}, se: {}
W. Recall: \t {}, se: {}
W. Precision: \t {}, se: {}'''.format(avg_accs, se_accs, avg_rmse, se_rmse, avg_wrecs, se_wrecs,
                                     avg_wprecs, se_wprecs))
    
    return


def cv_random_simple_rasar_multiclass(X, y, params_dict, n_models = 132, seed = 42):
    
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
                               a_ham = 2.7825594022071245, a_pub = 1000)
    distance_matrix = pd.DataFrame(distance_matrix)
    
    print('Start CV...', ctime())
    kf = KFold(n_splits=5, shuffle=True)
    accs = []
    rmse = []
    for train_index, test_index in kf.split(distance_matrix):

        dist_matr_train = distance_matrix.iloc[train_index,train_index]
        dist_matr_test = distance_matrix.iloc[test_index,train_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        print('Train and test...', ctime(), end = '')
        
        rasar_train, rasar_test = unsuper_simple_rasar_multiclass(dist_matr_train, dist_matr_test,
                                                                  X.iloc[train_index], X.iloc[test_index], 
                                                                  y_train, y_test)
        
        print('done... models...', ctime(), end = '')
        
        for i in range(0, len(params_comb)):
            
            lrc = LogisticRegression(n_jobs = -1, penalty = 'elasticnet', solver = 'saga', max_iter = 10000)
        
            for k,v in params_comb[i].items():
                setattr(lrc, k, v)
            
            lrc.fit(rasar_train, y_train)
            y_train_pred = lrc.predict(rasar_train)
            y_test_pred = lrc.predict(rasar_test)

            output_train_acc['mod' + str(i)].append(accuracy_score(y_train, y_train_pred))
            output_train_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_train, y_train_pred)))

            output_test_acc['mod' + str(i)].append(accuracy_score(y_test, y_test_pred))
            output_test_rmse['mod' + str(i)].append(sqrt(mean_squared_error(y_test, y_test_pred)))
        
        print('done')
        
    print('...END Simple RASAR -- Multiclass', ctime())
    
    tab_rasar_final = pd.DataFrame(columns = ['train_acc', 'test_acc', 'train_rmse', 'test_rmse'])

    tab_rasar_final.loc[:,'train_acc'] = pd.DataFrame(output_train_acc).mean(axis = 0)
    tab_rasar_final.loc[:,'test_acc'] = pd.DataFrame(output_test_acc).mean(axis = 0)
    tab_rasar_final.loc[:,'train_rmse'] = pd.DataFrame(output_train_rmse).mean(axis = 0)
    tab_rasar_final.loc[:,'test_rmse'] = pd.DataFrame(output_test_rmse).mean(axis = 0)
    
    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0, len(params_comb))])
    tab_rasar_final = pd.concat([tab_rasar_final, params_df], axis = 1)
    
    return tab_rasar_final










