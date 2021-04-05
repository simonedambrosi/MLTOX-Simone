# -*- coding: utf-8 -*-
"""
@author: Simone
"""

from helper_preprocessing_datafusion import *
from smiles_proc import *

def load_process_data_datafusion(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES):

    tests, species, results = load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES,)

    filtered_res = prefilter_dfr(species, tests, results)
    
    dfr = crosstab_rep_exp(filtered_res)

    dfr.loc[:, 'test_cas'] = dfr.test_cas.apply(to_cas)
        
    chemicals = pd.read_csv('dati utili/chemicals.csv').drop(columns = 'Unnamed: 0')

    lst_cas = chemicals.test_cas.unique().tolist()
    lst_cas_dfr = dfr.test_cas.unique().tolist()

    cas_to_extract = list()
    for i in lst_cas_dfr:
        if i not in lst_cas:
            cas_to_extract.append(i)

    cas_to_extract = pd.DataFrame(cas_to_extract, columns = ['test_cas'])

    extracted_chem = extract_chemical_data(cas_to_extract)
    extracted_chem = process_smiles_features(extracted_chem)

    chemicals_dfr = pd.concat([chemicals, extracted_chem], axis = 0).reset_index(drop = True)

    chemicals_dfr.to_csv('chemicals_datafusion.csv')

    final_db = dfr.merge(chemicals_dfr, left_on = 'test_cas', right_on = 'test_cas')

    final_db.to_csv('datafusion_db_processed.csv')
    
    return final_db

def crosstab_rep_exp(dataframe):
    ct = pd.crosstab(dataframe.effect, dataframe.endpoint)
    
    new_ct = ct.loc[ct.sum(axis = 1) > 200, ct.sum(axis = 0) > 200]

    dfr = pd.DataFrame()

    for j in new_ct.columns:
        for i in new_ct.index:
            if (new_ct.loc[i,j] >100) & (i != 'MOR'):
                pp = dataframe[dataframe.endpoint == j]
                pp = pp[pp.effect == i]

                best_db = select_impute_features(pp)
                try:
                    if repeated_experiments(best_db).shape[0] > 100:
                        rep_exp_db = repeated_experiments(best_db)
                        rep_exp_db.loc[:,'endpoint'] = pd.Series(np.repeat(j,rep_exp_db.shape[0]))
                        rep_exp_db.loc[:,'effect'] = pd.Series(np.repeat(i,rep_exp_db.shape[0]))

                        dfr = pd.concat([dfr, rep_exp_db])
                except:
                    continue
    return dfr