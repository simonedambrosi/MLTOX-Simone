# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:08:08 2020

@author: Simone
"""

from helper_preprocessing_copy1 import *
from smiles_proc import *

def load_process_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES, DATA_PATH_CHEMICALS):

    tests, species, results = load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES,)

    filtered_res = prefilter(species, tests, results)
    
    ct = pd.crosstab(filtered_res.effect, filtered_res.endpoint)
    new_ct = ct.loc[ct.sum(axis = 1) > 200, ct.sum(axis = 0) > 200]

    dfr = pd.DataFrame()

    for j in new_ct.columns:
        for i in new_ct.index:
            if (new_ct.loc[i,j] >100) & (i != 'MOR'):
                pp = filtered_res[filtered_res.endpoint == j]
                pp = pp[pp.effect == i]

                best_db = select_impute_features(pp)
                try:
                    if repeated_experiments(best_db).shape[0] > 100:
                        final_db = repeated_experiments(best_db)
                        final_db.loc[:,'endpoint'] = pd.Series(np.repeat(j,final_db.shape[0]))
                        final_db.loc[:,'effect'] = pd.Series(np.repeat(i,final_db.shape[0]))

                        dfr = pd.concat([dfr, final_db])
                except:
                    print()
    
    dfr.loc[:, 'test_cas'] = dfr.test_cas.apply(to_cas)

    # find smiles e pubchem evitati
    chem_feat = process_chemicals(DATA_PATH_CHEMICALS)
    chem_feat = process_smiles_features(chem_feat)

    dfr = dfr.merge(chem_feat, left_on = 'test_cas', right_on = 'test_cas')
    
    return dfr

