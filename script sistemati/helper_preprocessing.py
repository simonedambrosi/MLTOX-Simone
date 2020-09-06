# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 10:25:45 2020

@author: Simone
"""

import pandas as pd
import numpy as np
from smiles_proc import *
from sklearn.preprocessing import MinMaxScaler

def load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES):
    tests = pd.read_csv(DATA_PATH_TESTS, sep = '\|', engine = 'python')
    print('tests loaded')
    species = pd.read_csv(DATA_PATH_SPECIES, sep = '\|', engine = 'python')
    print('species loaded')
    results = pd.read_csv(DATA_PATH_RESULTS, sep = '\|', engine = 'python')
    print('results loaded')

    return tests, species, results

def prefilter(species, tests, results):
    # Filtro sugli endpoint 
    resc = results[(results.endpoint.str.contains('EC50')) | (results.endpoint.str.contains('LC50'))]
    
    # Filtro sull'effetto -- Vedere anche gli altri effetti (occhio Growth perchè sono tutti Embrions)
    resc_mor = resc[resc.effect.str.contains('MOR')]

    # Filtro sulle specie
    species = species[~species.ecotox_group.isnull()]
    species_fish = species[species.ecotox_group.str.contains("Fish")]

    # No Embrions
    test_no_EM = tests[tests.organism_lifestage != "EM"]

    # Unisco le informazioni sui pesci con gli esperimenti
    test_fish_only = test_no_EM.merge(species_fish, on="species_number")

    # Unisco le informazioni sui risultati degli esperimenti con gli esperimenti
    results_prefilter = resc_mor.merge(test_fish_only, on = "test_id")
    
    return results_prefilter

def select_impute_features(prefiltered_results):
    
    keep_columns = ['obs_duration_mean', 'obs_duration_unit',
       'endpoint', 'effect', 'measurement', 'conc1_type', 'conc1_mean',
       'conc1_unit', 'test_cas', 'test_location', 'exposure_type',
       'control_type', 'media_type', 'application_freq_unit', 'class',
       'tax_order', 'family', 'genus', 'species']
    
    db = prefiltered_results.copy()
    db = db[keep_columns]
    
    db = impute_conc(db)
    
    db = impute_test_feat(db)
    
    db = impute_duration(db)
    
    db = impute_species(db)
    
    return db
    
def impute_conc(results_prefiltered):
    
    db = results_prefiltered.copy()
    
    to_drop_conc_mean = db[db.conc1_mean == 'NR'].index
    db_filtered_mean = db.drop(index = to_drop_conc_mean).copy()

    db_filtered_mean.loc[:,'conc1_mean'] = db_filtered_mean.conc1_mean\
                                                        .apply(lambda x: x.replace("*", "") if "*" in x else x).copy()

    to_drop_invalid_conc = db_filtered_mean[db_filtered_mean.conc1_mean == '>100000'].index
    db_filtered_mean.drop(index = to_drop_invalid_conc, inplace = True)

    db_filtered_mean.loc[:,'conc1_mean'] = db_filtered_mean.conc1_mean.astype(float).copy()

    to_drop_useless = db_filtered_mean[db_filtered_mean.conc1_mean == 0].index
    db_filtered_mean.drop(index = to_drop_useless, inplace = True)

    db_filtered_mean.loc[:,'conc1_unit'] = db_filtered_mean.conc1_unit\
                                                        .apply(lambda x: x.replace("AI ", "") if 'AI' in x else x)
    
    db_filtered_mean.loc[(db_filtered_mean.conc1_unit == 'ppb') | (db_filtered_mean.conc1_unit == 'ug/L'), 'conc1_mean'] = \
                db_filtered_mean.conc1_mean[(db_filtered_mean.conc1_unit == 'ppb') | (db_filtered_mean.conc1_unit == 'ug/L')]/1000
    
    db_filtered_mean.loc[db_filtered_mean.conc1_unit == 'ng/L', 'conc1_mean'] = db_filtered_mean.conc1_mean[db_filtered_mean.conc1_unit == 'ng/L']*(10**(-6))

    to_drop_unit = db_filtered_mean.loc[(db_filtered_mean.conc1_unit == 'uM') | (db_filtered_mean.conc1_unit == 'ul/L')].index
    db_filtered_mean.drop(index = to_drop_unit, columns = ['conc1_unit'], inplace = True)

    to_drop_type = db_filtered_mean.loc[(db_filtered_mean.conc1_type == 'NC') |(db_filtered_mean.conc1_type == 'NR')].index
    db_filtered_mean.drop(index = to_drop_type, inplace = True)
    
    return db_filtered_mean

def impute_test_feat(results_prefiltered):
    
    db = results_prefiltered.copy()
    
    db.loc[:,'exposure_type'] = db.exposure_type.apply(lambda x: x.replace("/", ""))
    db.loc[:,'exposure_type'] = db.exposure_type.apply(lambda x: x.replace("AQUA - NR", "AQUA") if "AQUA" in x else x)
    db.loc[:, 'exposure_type'] = db.exposure_type.apply(lambda x: 'AQUA' if 'NR' in x else x)
    
    db.drop(columns = ['test_location'], inplace = True)
    
    db.loc[:, 'control_type'] = db.control_type.apply(lambda x: x.replace("/", ""))
    db.loc[:, 'control_type'] = db.control_type.apply(lambda x: "Unknown" if "NR" in x else x)
    
    db.loc[:,'media_type'] = db.media_type.apply(lambda x: x.replace("/", ""))
    to_drop_media = db[db.media_type.isin(['NR', 'CUL', 'NONE', 'NC'])].index
    db.drop(index = to_drop_media, inplace = True)
    
    db.loc[:,'application_freq_unit'] = db.application_freq_unit.apply(lambda x: "X" if ('NR' in x) | ('NC' in x) else x) 
    
    return db

def impute_duration(results_prefiltered):
    
    db = results_prefiltered.copy()
    
    good_obs_unit = ["h", "d", "mi", "wk", "mo"] 
    db_filtered_unit = db[db.obs_duration_unit.isin(good_obs_unit)].copy()
    
    to_drop_obs_mean = db_filtered_unit[db_filtered_unit.obs_duration_mean == 'NR'].index
    db_filtered_unit.drop(index = to_drop_obs_mean, inplace = True)
    db_filtered_unit.obs_duration_mean = db_filtered_unit.obs_duration_mean.astype(float)
    
    
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'd', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'd']\
                                                                    .apply(lambda x: x*24)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'mi', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'mi']\
                                                                    .apply(lambda x: x/60)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'wk', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'wk']\
                                                                    .apply(lambda x: x*7*24)
    db_filtered_unit.loc[db_filtered_unit.obs_duration_unit == 'mo', 'obs_duration_mean'] = \
                                        db_filtered_unit.obs_duration_mean[db_filtered_unit.obs_duration_unit == 'mo']\
                                                                    .apply(lambda x: x*30*24)
    
    db_filtered_unit.drop(columns = ['obs_duration_unit'], inplace = True)
    
    db_processed_duration = db_filtered_unit[db_filtered_unit.obs_duration_mean.isin([24,48,72,96])].copy()
    
    return db_processed_duration

def impute_species(results_prefiltered):
    
    db = results_prefiltered.copy()
    # Dropping missing values relative to species (same values are missing for genus)
    to_drop_spec = db[db.species.isnull()].index
    db.drop(index = to_drop_spec, inplace = True)
    
    # Dropping missing values relative to family
    to_drop_fam = db[db.family.isnull()].index
    db.drop(index = to_drop_fam, inplace = True)
    
    return db

def process_smiles_features(chemical_features):
    
    db = chemical_features.copy()
    
    db.bonds_number = db.bonds_number.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["bonds_number"]])
    db[["bonds_number"]] = minmax.transform(db[["bonds_number"]])
    
    db.atom_number = db.atom_number.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["atom_number"]])
    db[["atom_number"]] = minmax.transform(db[["atom_number"]])
    
    db.Mol = db.Mol.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["Mol"]])
    db[["Mol"]] = minmax.transform(db[["Mol"]])
    
    return db

def process_chemicals(DATA_PATH_CHEMICAL_FEATURES):
    
    chem_feat = pd.read_csv(DATA_PATH_CHEMICAL_FEATURES).drop(columns = ['Unnamed: 0'])
    
    chem_feat = adding_smiles_features(chem_feat)
    to_drop_nofeat = chem_feat[chem_feat['bonds_number'] == 'NaN'].index
    chem_feat.drop(index = to_drop_nofeat, inplace = True)
    
    return chem_feat

def repeated_experiments(imputed_db):
    db = imputed_db.copy()
    db['fish'] = db['class'] + ' ' + db['tax_order'] + ' ' + db['family'] + ' ' + db['genus'] + ' ' + db['species']
    
    db_species = db[['class', 'tax_order', 'family', 'genus', "species", 'fish']]
    db_species = db_species.groupby("fish").first()
    
    final_db = db.groupby(by = ['test_cas', 'obs_duration_mean', 'conc1_type', 'fish', 'exposure_type',
                     'control_type', 'media_type', 'application_freq_unit']).agg('median').reset_index()
    final_db = final_db.merge(db_species, on='fish')

    return final_db



