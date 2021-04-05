# -*- coding: utf-8 -*-
"""
Created on Mon Apr 5 12:57:14 2021

@author: Simone
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from time import ctime
from general_helper import multiclass_encoding

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix, f1_score

import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator


def load_data_rf(DATA_PATH, encoding, seed = 42):
    
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
    
    
    # Apro i pubchem
    db = pd.concat([db, pd.DataFrame(pd.DataFrame(db['pubchem2d'].values).\
                                     apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '), axis = 1)[0].to_list(),
                   columns = ['pub'+ str(i) for i in range(1,882)])], axis = 1)
    
    
    db.drop(columns = ['fish', 'pubchem2d'], inplace = True)
    
    # Encoding for target variable: binary and multiclass
    if encoding == 'binary':
        db['conc1_mean'] = np.where(db['conc1_mean'].values > 1, 0, 1)

    elif encoding == 'multiclass':
        t = db['conc1_mean'].copy()
        db['conc1_mean'] = multiclass_encoding(t)
    
    X = db.drop(columns = 'conc1_mean')
    y = db['conc1_mean'].values

    # splitting
    train, test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    # ricongiungo train con test
    db_output = train.append(test)
    
    train.loc[:, 'conc1_mean'] = y_train
    test.loc[:, 'conc1_mean'] = y_test
    db_output.loc[:, 'conc1_mean'] = np.concatenate((y_train, y_test))
    
    # tengo traccia della lunghezza del train set
    len_train = len(train)

    return db_output, train, test, len_train   



def cv_DistributedRandomForest_H2O(db, hyper_params = dict()):
    
    print(ctime())
    target = 'conc1_mean'
    predictors = db.columns.to_list()
    predictors.remove(target)
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    
    accs = []
    f1s = []
    sens = []
    specs = []
    
    n_epochs = 0
    for train_index, test_index in kf.split(db):
        
        n_epochs += 1
        print('Epoch {}: '.format(n_epochs), ctime())
        train_h2o = h2o.H2OFrame(db.iloc[train_index])
        test_h2o = h2o.H2OFrame(db.iloc[test_index])
        
        for i in db.columns:
            if db[i].dtypes == 'object':
                train_h2o[i] = train_h2o[i].asfactor()
                test_h2o[i] = test_h2o[i].asfactor()

        train_h2o['conc1_mean'] = train_h2o['conc1_mean'].asfactor()
        test_h2o['conc1_mean'] = test_h2o['conc1_mean'].asfactor()
        
        drf = H2ORandomForestEstimator(seed = 123, categorical_encoding = 'onehotexplicit', binomial_double_trees = True)
    
        for k,v in hyper_params.items():
            setattr(drf, k, v)
        
        drf.train(x = predictors, y = target, training_frame = train_h2o)
        
        y_pred = drf.predict(test_h2o).as_data_frame()['predict']
        y_test = test_h2o['conc1_mean'].as_data_frame()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
    
    print(ctime())
    print('Accuracy:   ', np.mean(accs),  'se:', sem(accs))
    print('Sensitivity:', np.mean(sens),  'se:', sem(sens))
    print('Specificity:', np.mean(specs), 'se:', sem(specs))
    print('F1 score:   ', np.mean(f1s),   'se:', sem(f1s))
    
    return

