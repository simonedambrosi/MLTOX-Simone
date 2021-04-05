# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 13:40:02 2021

@author: Simone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import ctime
from math import sqrt
from general_helper import multiclass_encoding

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler 
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score, mean_squared_error, confusion_matrix


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
