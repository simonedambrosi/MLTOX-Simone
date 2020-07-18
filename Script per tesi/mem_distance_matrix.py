# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:52:30 2020

@author: Simone
"""
import numpy as np

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
        return alphas[0] * ham1 + alphas[1] * euc2
    
    elif choice == [1,0]:
        return alphas[0] * ham1 + alphas[1] * euc2 + alphas[2] * pub3
    
    elif choice == [0,1]:
        return alphas[0] * ham1 + alphas[1] * euc2 + alphas[3] * tan4
    
    else:
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

def pubchem2d_matrix(X):
    print('Start Hamming su Pubchem2d...')
    a = np.array((X.pubchem2d[0].replace('', ' ').strip().split(' '),
                  X.pubchem2d[1].replace('', ' ').strip().split(' ')))
    
    for i in range(2,len(X.pubchem2d)):
        a = np.concatenate((a,[X.pubchem2d[i].replace('', ' ').strip().split(' ')]))

    pub_matrix = squareform(pdist(a, metric = 'hamming'))
    
    return pub_matrix

def tanimoto_matrix(X):
    print('Start Tanimoto...')
    return squareform(GetTanimotoDistMat([FingerprintMols.FingerprintMol(MolFromSmiles(X.smiles[i]))
                                                 for i in range(len(X.smiles))]))



