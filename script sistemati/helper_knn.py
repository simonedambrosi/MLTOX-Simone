# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:15:04 2020

@author: Simone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import ctime
from math import sqrt
from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix

from scipy.spatial.distance import pdist, squareform
from scipy.stats import sem

from rdkit.DataStructs.cDataStructs import CreateFromBitString
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat


def load_data_knn(DATA_PATH, encoding, seed = 42):
    
    db = pd.read_csv(DATA_PATH).drop(columns = ['Unnamed: 0', 'test_cas','smiles'])
    
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
    return squareform(pdist(pd.DataFrame(pd.DataFrame(X['pubchem2d'].values).apply(
        lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list()),  metric = 'hamming'))

def tanimoto_matrix(X):
    return squareform(GetTanimotoDistMat(X['pubchem2d'].apply(CreateFromBitString).to_list()))

def basic_matrix(X, cat_features, num_features, a_ham):
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features)

def euc_pub_matrix(X, num_features = [], a_pub = 0):
    return euclidean_matrix(X, num_features) + a_pub * pubchem2d_matrix(X)



def create_table(train_mat, test_mat, y_train, y_test, encoding = 'binary'):
    
    if encoding == 'binary':
        accs_train = []
        accs_test = []

        rmse_train = []
        rmse_test = []

        auc_train = []
        auc_test = []

        rec_train = []
        rec_test = []

        prec_train = []
        prec_test = []

        for k in [i for i in range(1,13,1)] + [i for i in range(13,52,2)]:
            print(k, end =' ')
            neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors = k, leaf_size = 60)
            neigh.fit(train_mat, y_train.ravel())
            y_pred_train = neigh.predict(train_mat)
            y_pred = neigh.predict(test_mat)

            accs_train.append(accuracy_score(y_train, y_pred_train))
            accs_test.append(accuracy_score(y_test, y_pred))

            rmse_train.append(sqrt(mean_squared_error(y_train, y_pred_train)))
            rmse_test.append(sqrt(mean_squared_error(y_test, y_pred)))

            auc_train.append(roc_auc_score(y_train, y_pred_train))
            auc_test.append(roc_auc_score(y_test, y_pred))

            rec_train.append(recall_score(y_train,y_pred_train))
            rec_test.append(recall_score(y_test, y_pred))

            prec_train.append(precision_score(y_train, y_pred_train))
            prec_test.append(precision_score(y_test, y_pred))

        table = pd.DataFrame({'acc_train': accs_train, 'acc_test': accs_test,
                            'rmse_train': rmse_train, 'rmse_test': rmse_test,
                            'auc_train': auc_train, 'auc_test': auc_test,
                            'recall_train': rec_train, 'recall_test':rec_test,
                            'precision_train': prec_train, 'precision_test': prec_test},
                          index = ['k = ' + str(i) for i in range(1,13,1)] + ['k = ' + str(i) for i in range(13,52,2)])
        
    elif encoding == 'multiclass':
        accs_train = []
        accs_test = []

        rmse_train = []
        rmse_test = []

        for k in [i for i in range(1,13,1)] + [i for i in range(13,52,2)]:
            print(k, end =' ')
            neigh = KNeighborsClassifier(metric = 'precomputed', n_neighbors = k, leaf_size = 60)
            neigh.fit(train_mat, y_train.ravel())
            y_pred_train = neigh.predict(train_mat)
            y_pred = neigh.predict(test_mat)

            accs_train.append(accuracy_score(y_train, y_pred_train))
            accs_test.append(accuracy_score(y_test, y_pred))

            rmse_train.append(sqrt(mean_squared_error(y_train, y_pred_train)))
            rmse_test.append(sqrt(mean_squared_error(y_test, y_pred)))
        
        table = pd.DataFrame({'acc_train': accs_train, 'acc_test': accs_test,
                            'rmse_train': rmse_train, 'rmse_test': rmse_test},
                          index = ['k = ' + str(i) for i in range(1,13,1)] + ['k = ' + str(i) for i in range(13,52,2)])
            
            
    return table

def display_table(table, main_title, encoding = 'binary'):
    k = [i for i in range(1,13,1)] + [i for i in range(13,52,2)]
    
    if encoding == 'binary':
        plt.figure(figsize = [20,15])
        plt.subplot(3,2,1)
        plt.suptitle(main_title)

        plt.plot(k, table.acc_train, '-o')
        plt.plot(k, table.acc_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('Accuracy')

        plt.subplot(3,2,2)
        plt.plot(k,table.rmse_train, '-o')
        plt.plot(k,table.rmse_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('RMSE')
        
        plt.subplot(3,2,3)
        plt.plot(k, table.auc_train, '-o')
        plt.plot(k, table.auc_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('AUC')

        plt.subplot(3,2,4)
        plt.plot(k, table.recall_train, '-o')
        plt.plot(k, table.recall_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('Recall')

        plt.subplot(3,2,5)
        plt.plot(k, table.precision_train, '-o')
        plt.plot(k, table.precision_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('Precision')
        
    elif encoding == 'multiclass':
        plt.figure(figsize = [14,5])
        plt.subplot(1,2,1)
        plt.suptitle(main_title)

        plt.plot(k, table.acc_train, '-o')
        plt.plot(k, table.acc_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('Accuracy')

        plt.subplot(1,2,2)
        plt.plot(k,table.rmse_train, '-o')
        plt.plot(k,table.rmse_test, '-o')
        plt.grid()
        plt.xlabel('k')
        plt.title('RMSE')
    
    return plt.show()



def cv_params_new(X,y, categorical, non_categorical,
                  sequence_pub = [], sequence_tan = [], sequence_ham = [],
                  choice = [0,0],
                  ks = range(1,10,2), leaf_size = range(10, 101, 10),
                  a_ham = 0,  a_pub = 0, a_tan = 0):
    
    '''
    choice: [0,0] optimize Hamming 1 with Euclidean
            [1,0] optimize Hamming 1 with Euclidean and Pubchem2d
            [0,1] optimize Pubchem2d with Euclidean and Hamming 1
    '''
    np.random.seed(123)
    print(ctime())
    print('START...')
    best_accuracy = 0
    best_alpha = 0
    best_k = 0
    best_leaf = 0
    
    if choice == [1,0]:
        print('Computing Euclidean and Pubchem2d Matrix...')
        basic_mat = euc_pub_matrix(X, non_categorical, a_pub)
        
        for ah in sequence_ham:
            print('Adding Hamming 1 (Categorical)... alpha = {}'.format(ah))
            dist_matr = ah * pubchem2d_matrix(X)
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
                                                        rmse: {}, st.error:  {}'''.format(ah, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = ah
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls


    elif choice ==  [0,1]:
        print('Computing Basic Matrix: Hamming 1 and Euclidean 2...')
        basic_mat = basic_matrix(X, categorical, non_categorical, a_ham)
        
        for ap in sequence_pub:
            print('Adding Hamming 3 (Pubchem2d)... alpha = {}'.format(ap))
            dist_matr = ap * pubchem2d_matrix(X)
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
        
    elif choice == [0,0]:
        print('Computing Euclidean ...')
        basic_mat = euclidean_matrix(X, non_categorical)

        for ah in sequence_ham:
            print('Adding Hamming 1 (Categorical)... alpha = {}'.format(ah))
            dist_matr = ah * hamming_matrix(X, categorical)
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
                                                        rmse: {}, st.error:  {}'''.format(ah, k, ls,
                                                                                        avg_acc, se_acc,
                                                                                        avg_rmse, se_rmse))
                        best_alpha = ah
                        best_k = k
                        best_accuracy = avg_acc
                        best_leaf = ls
                        
    print(ctime())
    return best_accuracy, best_alpha, best_k, best_leaf



def cv_binary_knn(X, y, a_ham, a_pub, k, leaf_size):
    
    # Best combination
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    best_k = k
    best_leaf = leaf_size

    print('Basic Matrix...', ctime())
    dist_matr = basic_matrix(X, categorical, non_categorical, a_ham)
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
                                     n_neighbors=best_k, n_jobs=-2,
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


def cv_multiclass_knn(X, y, a_ham, a_pub, k, leaf_size):
    
    # Best combination
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP']
    
    best_k = k
    best_leaf = leaf_size

    print('Basic Matrix...', ctime())
    dist_matr = basic_matrix(X, categorical, non_categorical, a_ham)
    print('Adding pubchem2d', ctime())
    dist_matr += a_pub * pubchem2d_matrix(X)
    dist_matr = pd.DataFrame(dist_matr)
    print('End distance matrix...', ctime())
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    accs = []
    rmse = []
    wrecs = []
    wprecs = []
    for train_index, test_index in kf.split(dist_matr):

        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = y[train_index]
        y_test = y[test_index]

        neigh = KNeighborsClassifier(metric = 'precomputed',
                                     n_neighbors=best_k, n_jobs=-2,
                                     leaf_size=best_leaf)
        neigh.fit(X_train, y_train.ravel())
        y_pred = neigh.predict(X_test)


        accs.append(accuracy_score(y_test, y_pred))
        rmse.append(sqrt(mean_squared_error(y_test, y_pred)))
        wrecs.append(recall_score(y_test, y_pred, average = 'weighted'))
        wprecs.append(precision_score(y_test, y_pred, average = 'weighted'))
        
        del X_train, X_test, y_train, y_test

    avg_acc = np.mean(accs)
    se_acc = sem(accs)

    avg_rmse = np.mean(rmse)
    se_rmse = sem(rmse)

    avg_wrecs = np.mean(wrecs)
    se_wrecs = sem(wrecs)
    
    avg_wprecs = np.mean(wprecs)
    se_wprecs = sem(wprecs)
    
    print('''Accuracy: \t {}, se: {}
RMSE: \t\t {}, se: {}
W. Recall: \t {}, se:{}
W. Precision: \t {}, se: {}'''.format(avg_acc, se_acc, avg_rmse, se_rmse, avg_wrecs, se_wrecs, avg_wprecs, se_wprecs))
    return






