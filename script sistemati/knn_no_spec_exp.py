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


