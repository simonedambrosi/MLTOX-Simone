# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:26:46 2020

@author: Simone
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score, mean_squared_error

def import_data_encoded(encoding = 'binary'):
    
    db = pd.read_csv('dataset/db_modelli_smiles_pubchem.csv').drop(columns = ['Unnamed: 0', 'test_cas'])
    
    # Ordinal Encoding
    encoder = OrdinalEncoder(dtype = int)

    encoder.fit(db[['species', 'conc1_type', 'exposure_type', 'class', 'tax_order', 'family', 'genus']])

    db[['species', 'conc1_type', 'exposure_type', 'class', 'tax_order', 'family', 'genus']] = encoder.transform(
        db[['species', 'conc1_type', 'exposure_type', 'class', 'tax_order', 'family', 'genus']])+1
    
    
    if encoding == 'binary':
        db['conc1_mean'] = np.where(db['conc1_mean'].values > 1, 1, 0)

    elif encoding == 'multiclass':
        t = db['conc1_mean'].copy()
        db['conc1_mean'] = multiclass_encoding(t)
    
    X = db.drop(columns = 'conc1_mean')
    y = db['conc1_mean'].values
    
    # splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # ricongiungo train con test
    X_try = X_train.append(X_test)

    # tengo traccia della lunghezza del train set
    len_X_train = len(X_train)

    return X_try, X_train, X_test, y_train, y_test, len_X_train


'''
Encoding for multiclass
'''
def multiclass_encoding(var):
    for i in range(0,len(var)):
        if var[i] <= 10**-1:
            var[i] = 5
        
        elif 10**-1 < var[i] <= 10**0:
            var[i] = 4
        
        elif 10**0 < var[i] <= 10**1:
            var[i] = 3
            
        elif 10**1 < var[i] <= 10**2:
            var[i] = 2
            
        else:
            var[i] = 1
    return pd.to_numeric(var, downcast = 'integer')

'''
Function to create tables of skills of KNN for both binary and multiclass problem
'''

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
    
'''
Brute display of table
'''
def display_table(table, main_title, encoding = 'binary', k):
#     k = [i for i in range(1,13,1)] + [i for i in range(13,52,2)]
    
    if encoding == 'binary':
        plt.figure(figsize = [20,15])
        plt.subplot(3,2,1)
        plt.suptitle(main_title)

        plt.plot(k, table.acc_train)
        plt.plot(k, table.acc_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('Accuracy')

        plt.subplot(3,2,2)
        plt.plot(k,table.rmse_train)
        plt.plot(k,table.rmse_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('RMSE')
        
        plt.subplot(3,2,3)
        plt.plot(k, table.auc_train)
        plt.plot(k, table.auc_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('AUC')

        plt.subplot(3,2,4)
        plt.plot(k, table.recall_train)
        plt.plot(k, table.recall_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('Recall')

        plt.subplot(3,2,5)
        plt.plot(k, table.precision_train)
        plt.plot(k, table.precision_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('Precision')
        
    elif encoding == 'multiclass':
        plt.figure(figsize = [14,5])
        plt.subplot(1,2,1)
        plt.suptitle(main_title)

        plt.plot(k, table.acc_train)
        plt.plot(k, table.acc_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('Accuracy')

        plt.subplot(1,2,2)
        plt.plot(k,table.rmse_train)
        plt.plot(k,table.rmse_test)
        plt.grid()
        plt.xlabel('k')
        plt.title('RMSE')
    
    return plt.show()





