# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:08:08 2020

@author: Simone
"""

from helper_preprocessing import *
from smiles_proc import *

def load_process_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES, DATA_PATH_CHEMICALS):

    tests, species, results = load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES,)

    filtered_res = prefilter(species, tests, results)

    best_db = select_impute_features(filtered_res)

    final_db = repeated_experiments(best_db)

    final_db.loc[:, 'test_cas'] = final_db.test_cas.apply(to_cas)

    # find smiles e pubchem evitati
    chem_feat = process_chemicals(DATA_PATH_CHEMICALS)
    chem_feat = process_smiles_features(chem_feat)

    final_db = final_db.merge(chem_feat, left_on = 'test_cas', right_on = 'test_cas')
    
    return final_db