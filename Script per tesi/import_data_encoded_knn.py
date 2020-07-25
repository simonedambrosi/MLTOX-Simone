# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 00:26:46 2020

@author: Simone
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


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