import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, f1_score

from collections import Counter

from scipy.stats import sem

from time import ctime

from helper_rasar_simple import right_neighbor, df_train_simple_rasar, df_test_simple_rasar


def load_data_rasar_datafusion(path_mortality, path_datafusion):
    db = pd.read_csv(path_mortality).drop(columns = 'Unnamed: 0')
    db_df = pd.read_csv(path_datafusion).drop(columns = 'Unnamed: 0')

    # Pubchem
    db = pd.concat([db,
                    pd.DataFrame(pd.DataFrame(db['pubchem2d'].values).\
                                 apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '), 
                                                                            axis = 1)[0].to_list(),
                       columns = ['pub'+ str(i) for i in range(1,882)])],
                   axis = 1)

    db_df = pd.concat([db_df,
                    pd.DataFrame(pd.DataFrame(db_df['pubchem2d'].values).\
                                 apply(lambda x: x.str.replace('', ' ').str.strip().str.split(' '), 
                                                                            axis = 1)[0].to_list(),
                       columns = ['pub'+ str(i) for i in range(1,882)])],
                   axis = 1)

    # Numerical
    numerical = ['atom_number', 'bonds_number','Mol', 'MorganDensity', 'LogP', 'alone_atom_number', 'doubleBond',
                 'tripleBond', 'ring_number', 'oh_count', 'MeltingPoint', 'WaterSolubility']

    for nc in numerical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])

    for nc in numerical:
        minmax = MinMaxScaler()
        minmax.fit(db_df[[nc]])
        db_df[[nc]] = minmax.transform(db_df[[nc]])

    # Categorical
    categorical = ['obs_duration_mean', 'conc1_type', 'exposure_type', 'control_type', 'media_type',
                   'application_freq_unit', 'species', 'class', 'tax_order', 'family', 'genus']

    db, db_df = onehot_categorical(categorical, db, db_df)

    # encoding target
    db['target'] = np.where(db['conc1_mean'] > 1, 0, 1)
    db_df['target'] = np.where(db_df['conc1_mean'] > 1, 0, 1)

    # Cleaning

    db_mortality = db.drop(columns = ['test_cas', 'conc1_mean', 'fish', 'smiles','pubchem2d'])
    db_datafusion = db_df.drop(columns = ['test_cas', 'conc1_mean', 'smiles','pubchem2d'])
    
    return db_mortality, db_datafusion

def onehot_categorical(categorical_features, database, database_datafusion):
    categories = []
    for column in database[categorical_features]:
        cat_db = np.array(sorted(database[column].unique()))
        cat_datafusion = np.array(sorted(database_datafusion[column].unique()))
        for i in cat_datafusion:
            if i not in cat_db:
                cat_db = np.append(cat_db, i)

        categories.append(cat_db)
        
    encoder = OneHotEncoder(categories = categories, dtype = int, sparse = False)
    encoder.fit(database[categorical_features])
    
    final_db = pd.concat([database.drop(columns = categorical_features),
                     pd.DataFrame(encoder.transform(database[categorical_features]),
                                  columns = encoder.get_feature_names(categorical_features))], axis = 1)
    
    encoder.fit(database_datafusion[categorical_features])
    
    final_db_datafusion = pd.concat([database_datafusion.drop(columns = categorical_features),
                     pd.DataFrame(encoder.transform(database_datafusion[categorical_features]),
                                  columns = encoder.get_feature_names(categorical_features))], axis = 1)
    
    return final_db, final_db_datafusion

def find_similar_exp(exp_mortality, db_endpoint_effect, compare_features):
    '''Given an LC50-Mortality experiment, this function checks if there is another experiment with 
    other endpoint and effect done in the same condition. If it exists in the other database and the similar
    experiment has negative/positive (non toxic/toxic) outcome, the function will return respectively -1 or 1. 
    If the similar experiment does not exist, the function will return 0.'''
    out = db_endpoint_effect.target[
        (db_endpoint_effect[compare_features] == exp_mortality[compare_features]).all(axis = 1)].values
    try:
        return -1 if Counter(out).most_common(1)[0][0] == 0 else 1
    except:
        return 0 

def create_label_rasar(path_mortality, path_datafusion):
    
    db = pd.read_csv(path_mortality).drop(columns = 'Unnamed: 0')
    df = pd.read_csv(path_datafusion).drop(columns = 'Unnamed: 0')
    
    df['fish'] = df['class'] + ' ' + df['tax_order'] + ' ' + df['family'] + ' ' + df['genus'] + ' ' + df['species']
    
    db['target'] = np.where(db['conc1_mean'] > 1, 0, 1)
    df['target'] = np.where(df['conc1_mean'] > 1, 0, 1)
    
    comparing = ['test_cas', 'obs_duration_mean', 'conc1_type', 'exposure_type', 'control_type', 'media_type',
             'application_freq_unit', 'fish']

    grouped_datafusion = df.groupby(by=['endpoint', 'effect'])

    db_datafusion_rasar_label = pd.DataFrame()

    for g in grouped_datafusion.groups:
        name = g[0] + '_' + g[1] + '_' + 'label'

        group = grouped_datafusion.get_group(g).drop(columns = ['endpoint', 'effect'])
    
        db_datafusion_rasar_label[name] = db.apply(
            lambda x: find_similar_exp(x, group, comparing), axis = 1).reset_index(drop = True)
        
    return db_datafusion_rasar_label    
    

def df_datafusion_rasar(db_datafusion, db):

    grouped_datafusion = db_datafusion.groupby(by=['endpoint', 'effect', 'target'])

    db_datafusion_rasar = pd.DataFrame()

    for group in grouped_datafusion.groups:

        name = group[0] + '_' + group[1] + '_' + str(group[2])

        train_X = grouped_datafusion.get_group(group).drop(columns = ['endpoint', 'effect', 'target'])
        test_X = db.copy()

        train_y = grouped_datafusion.get_group(group)['target'].values

        knn = KNeighborsClassifier(n_jobs = -1, n_neighbors = 1)
        knn.fit(train_X, train_y)

        neigh = knn.kneighbors(test_X, return_distance = True)

        db_datafusion_rasar[name] = neigh[0].ravel()
        
    return db_datafusion_rasar


def rasar_train_test(db_train, db_test, y_train, y_test, db_datafusion, db_label):
    
    db_simple_rasar_train = df_train_simple_rasar(db_train, y_train)
    
    db_simple_rasar_test = df_test_simple_rasar(db_train, db_test, y_train, y_test)
    
    db_datafusion_rasar_train = df_datafusion_rasar(db_datafusion, db_train)
    
    db_datafusion_rasar_test = df_datafusion_rasar(db_datafusion, db_test)
    
    X_train_rasar = pd.concat([db_simple_rasar_train[['dist_neigh0', 'dist_neigh1']], db_datafusion_rasar_train,
                               db_label.iloc[db_train.index].reset_index(drop = True)], axis = 1)
    X_test_rasar = pd.concat([db_simple_rasar_test[['dist_neigh0', 'dist_neigh1']], db_datafusion_rasar_test,
                             db_label.iloc[db_test.index].reset_index(drop = True)], axis = 1)
    
    return X_train_rasar, X_test_rasar


def cv_datafusion_rasar(db_mortality, db_datafusion, db_label):

    accs = []
    sens = []
    specs = []
    f1s = []

    kf = KFold(n_splits=5, shuffle=True, random_state = 5645)
    print(ctime())
    for train_index, test_index in kf.split(db_mortality):

        db_mortality_train = db_mortality.drop(columns = 'target').iloc[train_index]
        db_mortality_test = db_mortality.drop(columns = 'target').iloc[test_index]

        y_train = db_mortality['target'].iloc[train_index].copy().values
        y_test = db_mortality['target'].iloc[test_index].copy().values

        X_train, X_test = rasar_train_test(db_mortality_train, db_mortality_test, y_train, y_test, db_datafusion, db_label)
     
        clf = RandomForestClassifier(n_jobs = -1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        f1s.append(f1_score(y_test, y_pred))
        print(ctime())
    
    print('Accuracy:   ', np.mean(accs),  'se:', sem(accs))
    print('Sensitivity:', np.mean(sens),  'se:', sem(sens))
    print('Specificity:', np.mean(specs), 'se:', sem(specs))
    print('F1 score:   ', np.mean(f1s),   'se:', sem(f1s))
    
    return
























