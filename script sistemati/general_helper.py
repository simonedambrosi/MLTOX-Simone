# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:06:54 2020

@author: Simone
"""

import pandas as pd
import numpy as np


def null_output_counts(dataframe):
    
    # Find columns that start with the interesting feature
    features_interested = list(dataframe.columns)
    
    df_nan  = pd.DataFrame (index = features_interested, columns = ['null_values_inc_NC_NR%', '#outputs'])
    
    #Count total NaN + NR + NC
    for i in features_interested:
        df_nan.loc[i, 'null_values_inc_NC_NR%'] = (sum(dataframe[i].isnull()) + len(dataframe[dataframe[i] == "NR"]) + len(dataframe[dataframe[i] =="NC"]))/len(dataframe)*100
        df_nan.loc[i, '#outputs'] = len(dataframe[i].unique())
    return df_nan

def checkTree(dataset):
    '''
    Vogliamo vedere se nel dataset delle specie di animali su cui sono stati effettuati i test è presente un albero 
    filogenetico, ossia se le specie di pesci condividono il genere allora condividono anche la famiglia, 
    se condividono la famiglia allora condividono anche l'ordine tassonomico così via fino alla classe.
    '''
    lst_fam = list()
    lst_to = list() 
    lst_c = list()  # le verifiche incrociate
    
    for r1 in dataset.values:
        for r2 in dataset.values:
            if r1[3] == r2[3]:
                lst_fam.append(r1[2] == r2[2])
            if r1[2] == r2[2]:
                lst_to.append(r1[1] == r2[1])
            if r1[1] == r2[1]:
                lst_c.append(r1[0] == r2[0])
    
    lst_fam = pd.Series(lst_fam)
    lst_to = pd.Series(lst_to)
    lst_c = pd.Series(lst_c)
    
    if (sum(lst_fam == False) == 0) & (sum(lst_to == False) == 0) & (sum(lst_c == False) == 0):
        print('Tutto ok, è un albero')
    else:
        print('Ops... non è un albero')
        
def df_to_newick(dataframe):
    idx = np.random.randint(0,len(dataframe),200)

    dataframe = dataframe.iloc[idx]
    new_tree = '('
    for c in dataframe['class'].unique():
        df = dataframe[dataframe['class'] == c]

        new_tree += '('

        for to in df['tax_order'].unique():
            tmp0 = df[df['tax_order'] == to]

            new_tree += '('

            for fam in tmp0['family'].unique():
                sp = tmp0['gen_spec'][tmp0['family'] == fam].values.tolist()

                if len(sp) != 1:
                    new_tree += '(' + ','.join(sp) + ')' + fam +','
                else:
                    new_tree += '(' + ''.join(sp) + ')' + fam + ','
            new_tree = new_tree[:-1]
            new_tree += ')' + to + ','

        new_tree = new_tree[:-1]
        new_tree += ')' + c + ';'
        new_tree

    new_tree = new_tree[:-1]
    new_tree += ');'
    
    return new_tree

def multiclass_encoding(var, threshold = [10**-1, 10**0, 10**1, 10**2]):
    for i in range(0,len(var)):
        if var[i] <= threshold[0]:
            var[i] = 5
        
        elif threshold[0] < var[i] <= threshold[1]:
            var[i] = 4
        
        elif threshold[1] < var[i] <= threshold[2]:
            var[i] = 3
            
        elif threshold[2] < var[i] <= threshold[3]:
            var[i] = 2
            
        else:
            var[i] = 1
    return pd.to_numeric(var, downcast = 'integer')