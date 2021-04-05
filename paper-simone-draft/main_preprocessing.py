# -*- coding: utf-8 -*-

from helper_preprocessing import *
from smiles_proc import *


def load_process_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES):
    tests, species, results = load_raw_data(path_tests, path_results, path_species)

    prefiltered_data = prefilter(species, tests, results)
    
    del tests, species, results
    
    db = select_impute_features(prefiltered_data)
    db = repeated_experiments(db)
    db['test_cas'] = db.test_cas.apply(to_cas)
    
#    db.to_csv('repeated_experiments.csv')
    
    chemicals = extract_chemical_data(db)
    
    print('Cleaning Chemical data...', ctime())
    chemicals = process_smiles_features(chemicals)
    
    print('The extracted chemicals are:', chemicals.shape[0])
    
#    chemicals.to_csv('chemicals.csv')
    
    print('Merging chemical descriptors with experiments')
    final_db = db.merge(chemicals, right_on = 'test_cas', left_on = 'test_cas')
    
    print('Dimension of final dataset:', final_db.shape[0], 'experiments and', final_db.shape[1], 'features')
    print('Final features:', [i for i in final_db.columns])
    
#   final_db.to_csv('lc_db_processed.csv')

    return final_db
