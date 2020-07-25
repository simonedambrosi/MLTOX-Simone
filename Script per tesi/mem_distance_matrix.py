# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:52:30 2020

@author: Simone
"""
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat


def train_test_distances(X, len_X_train, cat_features, num_features, alphas = [1,1,1,1], choice = [0,0]):
    print('Start...')
    if choice == [0,0]:
        ham1, euc2 = mem_distance_matrices(X, cat_features, num_features, choice)
        
        dist_matr = combine_distances(ham1, euc2, 0, 0, alphas, choice)
    
    elif choice == [1,0]:
        ham1, euc2, pub3  = mem_distance_matrices(X, cat_features, num_features, choice)
        
        dist_matr = combine_distances(ham1, euc2, pub3, 0, alphas, choice)
        
    elif choice == [0,1]:
        ham1, euc2, tan4  = mem_distance_matrices(X, cat_features, num_features, choice)
        
        dist_matr = combine_distances(ham1, euc2, 0, tan4, alphas, choice)
        
    else:
         ham1, euc2, pub3, tan4 = mem_distance_matrices(X, cat_features, num_features, choice)
        
         dist_matr = combine_distances(ham1, euc2, pub3, tan4, alphas, choice)
        
    dist_matr_train = dist_matr[:len_X_train,:len_X_train]
    dist_matr_test = dist_matr[len_X_train:,:len_X_train]
    print('...FINISH')
    return dist_matr_train, dist_matr_test
    

'''
How to combine computed distances
'''

def combine_distances(ham1, euc2, pub3, tan4, alphas = [1,1,1,1], choice = [0,0]):
    print('Combining...')
    if choice == [0,0]:
        print('''Parametri: Hamming 1 --> {}, Euclidean 2 --> {}'''.format(alphas[0],alphas[1]))
        return alphas[0] * ham1 + alphas[1] * euc2
    
    elif choice == [1,0]:
        print('''Parametri: Hamming 1 --> {}, Euclidean 2 --> {}, Pubchem2d 3 --> {}'''.format(alphas[0],
                                                                                             alphas[1],
                                                                                             alphas[2]))
        return alphas[0] * ham1 + alphas[1] * euc2 + alphas[2] * pub3
    
    elif choice == [0,1]:
        print('''Parametri: Hamming 1 --> {}, Euclidean 2 --> {}, Tanimoto 4 --> {}'''.format(alphas[0],
                                                                                             alphas[1],
                                                                                             alphas[3]))
        return alphas[0] * ham1 + alphas[1] * euc2 + alphas[3] * tan4
    
    else:
        print('''Parametri: Hamming 1 --> {}, Euclidean 2 --> {},
            Pubchem2d 3 --> {}, Tanimoto 4 --> {}'''.format(alphas[0], alphas[1], alphas[2], alphas[3]))
        return alphas[0] * ham1 + alphas[1] * euc2 + alphas[2] * pub3 + alphas[3] * tan4




'''
Compute distances 1 and 2. In addition, it's possible to choose either Pubchem2d, Tanimoto or both.
'''

def mem_distance_matrices(X, cat_features = [], num_features = [], choice = [0,0]):
    '''
    Position 0 di choice --> Pubchem2d
    Position 1 di choice --> Tanimoto
    '''
    
    # Hamming and Euclidean
    if choice == [0,0]:
        print('You choose Hamming 1 and Euclidean 2...')
        return hamming_matrix(X, cat_features), euclidean_matrix(X, num_features)
    
    # Hamming, Euclidean and Hamming/pubchem2d
    elif choice == [1,0]:
        print('You choose Hamming 1, Euclidean 2 and Hamming on pubchem2d 3...')
        return hamming_matrix(X, cat_features), euclidean_matrix(X, num_features), pubchem2d_matrix(X)
    
    # Hamming, Euclidean and Tanimoto
    elif choice == [0,1]:
        print('You choose Hamming 1, Euclidean 2, Tanimoto 4...')
        return hamming_matrix(X, cat_features), euclidean_matrix(X, num_features), tanimoto_matrix(X)
    
    # ALL
    else:
        print('You choose Hamming 1, Euclidean 2, Hamming on pubchem2d 3 and Tanimoto 4...')
        return hamming_matrix(X, cat_features), euclidean_matrix(X, num_features), pubchem2d_matrix(X), tanimoto_matrix(X)


"""
Building 4 matrices:
    1) Hamming on categorical: 
        inputs: DataFrame, List of categorical features
    
    2) Euclidean on numerical:
        inputs: DataFrame, List of numerical features
    
    3) Hamming on pubchem2d:
        input: DataFrame containing a column of pubchem2d
    
    4) Tanimoto:
        input: DataFrame contaning a column of SMILES
"""

def hamming_matrix(X, cat_features):
    print('Start Hamming su categorical...')
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    print('Start Euclidean...')
    return squareform(pdist(X[num_features], metric = "euclidean"))

# 8 minutes more or less
def pubchem2d_matrix(X):
    print('Start Hamming su Pubchem2d...')
    return squareform(pdist(pd.DataFrame(pd.DataFrame(X['pubchem2d'].values).apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '),axis = 1)[0].to_list()),  metric = 'hamming'))

# more than 20 minutes
def tanimoto_matrix(X):
    print('Start Tanimoto...')
    return squareform(GetTanimotoDistMat([FingerprintMols.FingerprintMol(MolFromSmiles(X.smiles[i]))
                                                 for i in range(len(X.smiles))]))

'''
Fast Matrices
'''

def basic_matrix(X, cat_features = [], num_features = [], a_ham = 0.0016102620275609393):   
    # Hamming and Euclidean
    return a_ham * hamming_matrix(X, cat_features) + euclidean_matrix(X, num_features)


def fast_dist_mat(X, len_X_train, cat_features, num_features, alphas = [0,1,0,0], choice = [0,0]):
    dist_matr = basic_matrix(X, cat_features, num_features, alphas[0]) 
    if choice == [1,0]:
        dist_matr += alphas[2] * pubchem2d_matrix(X)
    elif choice == [0,1]:
        dist_matr += alphas[3] * tanimoto_matrix(X)
    
    return dist_matr[:len_X_train,:len_X_train], dist_matr[len_X_train:,:len_X_train]
    
