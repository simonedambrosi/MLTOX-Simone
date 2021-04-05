import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, ParameterSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, f1_score

from collections import Counter

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import sem

from time import ctime

def load_data_rasar_datafusion(DATA_MORTALITY_PATH, DATA_OTHER_ENDPOINT_PATH, encoding = 'binary', seed = 42):
    
    db_df = pd.read_csv(DATA_OTHER_ENDPOINT_PATH).drop(columns = 'Unnamed: 0')
    db = pd.read_csv(DATA_MORTALITY_PATH).drop(columns = 'Unnamed: 0')   
    
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
    
    
    
    # bonds_number and Mol have been minmax transformed yet
    non_categorical = ['obs_duration_mean', 'alone_atom_number', 'doubleBond', 'tripleBond', 'ring_number', 'MorganDensity', 'LogP',
                       'oh_count', 'WaterSolubility', 'MeltingPoint']
    
    # MinMax trasform for numerical variables -- MORTALITY
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db[[nc]])
        db[[nc]] = minmax.transform(db[[nc]])
    
    # OTHER ENDPOINT/EFFECT
    for nc in non_categorical:
        minmax = MinMaxScaler()
        minmax.fit(db_df[[nc]])
        db_df[[nc]] = minmax.transform(db_df[[nc]])
    
    categorical = ['conc1_type', 'exposure_type', 'control_type', 'media_type', 'application_freq_unit',
                'class', 'tax_order', 'family', 'genus', 'species']
    
    db, db_df = encoding_categorical(categorical, db, db_df)
    
    if encoding == 'binary':
        db['conc1_mean'] = np.where(db['conc1_mean'].values > 1, 0, 1)
        
        for ef in db_df.effect.unique():
            conc = db_df.loc[db_df.effect == ef, 'conc1_mean']
            db_df.loc[db_df.effect == ef, 'conc1_mean'] = np.where(conc > np.median(conc), 0, 1)
            
    elif encoding == 'multiclass':
        h2o.init()
        t = db['conc1_mean'].copy()
        db['conc1_mean'] = multiclass_encoding(t)
        
        for ef in db_df.effect.unique():
            conc = db_df.loc[db_df.effect == ef, 'conc1_mean'].copy()
            db_df.loc[db_df.effect == ef, 'conc1_mean'] = multiclass_encoding(conc.values, 
                                                                                              conc.quantile([.2,.4,.6,.8]).values)
    
    return db.drop(columns = ['test_cas','fish','smiles', 'pubchem2d']), db_df.drop(columns = ['test_cas','smiles', 'pubchem2d'])

def encoding_categorical(cat_cols, database, database_df):
    categories = []
    for column in database[cat_cols]:
        cat_db = np.array(sorted(database[column].unique()))
        cat_datafusion = np.array(sorted(database_df[column].unique()))
        for i in cat_datafusion:
            if i not in cat_db:
                cat_db = np.append(cat_db, i)

        categories.append(cat_db)
    
    encoder = OrdinalEncoder(categories = categories, dtype = int)

    encoder.fit(database[cat_cols])
    database[cat_cols] = encoder.transform(database[cat_cols])+1

    encoder.fit(database_df[cat_cols])
    database_df[cat_cols] = encoder.transform(database_df[cat_cols])+1
    
    return database, database_df


def hamming_matrix(X, cat_features):
    return squareform(pdist(X[cat_features], metric = "hamming"))

def euclidean_matrix(X, num_features):
    return squareform(pdist(X[num_features], metric = "euclidean"))

def pubchem2d_matrix(X):
    return squareform(pdist(X[X.columns[X.columns.str.contains('pub')]],  metric = 'hamming'))

def ham_pub_matrix(X, cat_features = [], a_ham = 0, a_pub = 0):
    return a_ham * hamming_matrix(X, cat_features) + a_pub * pubchem2d_matrix(X)


###########################################################################################################



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

###################################################################################################################
################################## SIMPLE RASAR ###################################################################
###################################################################################################################
def right_neighbor(neighbors, X_train, X_train_i):
    # IDX Neighbors
    idx_neigh_0 = pd.DataFrame(neighbors[1])[0].apply(lambda x: X_train_i.iloc[x].name)
    idx_neigh_1 = pd.DataFrame(neighbors[1])[1].apply(lambda x: X_train_i.iloc[x].name)
    
    idx_neigh = idx_neigh_0.copy()
    
    # dove l'indice del primo vicino risulta essere uguale a se stesso lo sostituisco con il secondo vicino
    idx_neigh[X_train.index == idx_neigh_0] = idx_neigh_1[X_train.index == idx_neigh_0].values
    
    # Distance from the Nearest Neighbor that is NOT itself
    dist_0 = pd.DataFrame(neighbors[0])[0]
    dist_1 = pd.DataFrame(neighbors[0])[1]
    
    distance = dist_0.copy()
    distance[X_train.index == idx_neigh_0] = dist_1[X_train.index == idx_neigh_0].values
    
    return idx_neigh, distance

def unsuper_simple_rasar(train_distance_matrix, test_distance_matrix, X_train, X_test, y_train, y_test):
    ######## starting DATAFRAME ##########
    
    X_train0 = X_train[y_train == 0].copy()
    X_train1 = X_train[y_train == 1].copy()
    
    # in order to train 1-NN
    dist_matr_train_0 = train_distance_matrix.iloc[y_train == 0, y_train == 0]
    dist_matr_train_1 = train_distance_matrix.iloc[y_train == 1, y_train == 1]
    
    # To find neighbors for train experiments --> df_rasar_train
    dist_matr_train_train_0 = train_distance_matrix.iloc[:,y_train == 0]
    dist_matr_train_train_1 = train_distance_matrix.iloc[:,y_train == 1]
    
    # To find neighbors for test experiments --> df_rasar_test
    dist_matr_test_train_0 = test_distance_matrix.iloc[:, y_train == 0]
    dist_matr_test_train_1 = test_distance_matrix.iloc[:, y_train == 1]
    
    ####### DF train RASAR ###############
    
    # finding the nearest 0s experiments for training experiments that is not itself
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 2)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_train_train_0, return_distance = True)
    _, dist0 = right_neighbor(neigh0, X_train, X_train0)
    
    # finding the nearest 1s experiments for training experiments that is not itself
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors=2)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_train_train_1, return_distance = True)
    _, dist1 = right_neighbor(neigh1, X_train, X_train1)
    
    df_rasar_train = pd.DataFrame({'dist_neigh0': dist0, 'dist_neigh1': dist1})
    
    ####### DF test RASAR ################
    
    # finding the nearest 0s experiments to test data
    knn0 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2)
    knn0.fit(dist_matr_train_0, y_train[y_train == 0])
    neigh0 = knn0.kneighbors(dist_matr_test_train_0, return_distance = True)
#     idx_neigh_0 = pd.DataFrame(neigh0[1])[0].apply(lambda x: X_train.iloc[y_train==0].iloc[x].name)

    # finding the nearest 1s experiments to test data
    knn1 = KNeighborsClassifier(metric = 'precomputed', n_neighbors=1, n_jobs=-2)
    knn1.fit(dist_matr_train_1, y_train[y_train == 1])
    neigh1 = knn1.kneighbors(dist_matr_test_train_1, return_distance = True)
#     idx_neigh_1 = pd.DataFrame(neigh1[1])[0].apply(lambda x: X_train.iloc[y_train==1].iloc[x].name)
    
    df_rasar_test = pd.DataFrame({'dist_neigh0': neigh0[0].ravel(), 'dist_neigh1': neigh1[0].ravel()})
    
    return df_rasar_train, df_rasar_test

###################################################################################################################
###################################################################################################################
###################################################################################################################


def euclidean_matrix_datafusion(X, Y, num_features):
    return cdist(X[num_features], Y[num_features], metric = "euclidean")
    
def hamming_matrix_datafusion(X, Y, cat_features):
    return cdist(X[cat_features], Y[cat_features], metric = "hamming")

def pubchem2d_matrix_datafusion(X, Y):
    return cdist(X[X.columns[X.columns.str.contains('pub')]], Y[Y.columns[Y.columns.str.contains('pub')]],
                 metric = 'hamming')

def ham_pub_matrix_datafusion(X, Y, cat_features, a_ham, a_pub):
    return a_ham * hamming_matrix_datafusion(X, Y, cat_features) + a_pub * pubchem2d_matrix_datafusion(X, Y) 


def unsuper_datafusion_rasar(db_mortality_train, db_mortality_test, db_datafusion, max_value):
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP', 'MeltingPoint',
                       'WaterSolubility']
    
    df_rasar_train = pd.DataFrame()
    df_rasar_test = pd.DataFrame()
    
    grouped_datafusion = db_datafusion.groupby(by=['endpoint', 'effect', 'conc1_mean'])

    for group in grouped_datafusion.groups:

        name = group[0] + '_' + group[1] + '_' + str(group[2])
        knn = KNeighborsClassifier(metric = 'precomputed', n_jobs = -2, n_neighbors = 1)
        
        
        train_db = grouped_datafusion.get_group(group)
        
        train_matrix = euclidean_matrix(train_db, non_categorical)/max_value + ham_pub_matrix(
            train_db, categorical, a_ham = 0.07498942093324558, a_pub = 0.5623413251903491)        
        
        test_train_matrix = euclidean_matrix_datafusion(db_mortality_train, train_db, non_categorical)/max_value + ham_pub_matrix_datafusion(db_mortality_train, train_db, categorical, a_ham = 0.07498942093324558, a_pub = 0.5623413251903491)
        
        test_test_matrix = euclidean_matrix_datafusion(db_mortality_test, train_db, non_categorical)/max_value + ham_pub_matrix_datafusion(db_mortality_test, train_db, categorical, a_ham = 0.07498942093324558, a_pub = 0.5623413251903491)

        
        knn.fit(train_matrix, train_db['conc1_mean'])
        
        neigh = knn.kneighbors(test_train_matrix, return_distance = True)
        df_rasar_train[name] = neigh[0].ravel()
        
        neigh = knn.kneighbors(test_test_matrix, return_distance = True)
        df_rasar_test[name] = neigh[0].ravel()
        
        ####### FIND LABELS
        
        
    return df_rasar_train, df_rasar_test


def cv_datafusion_rasar_alpha(db_mortality, db_datafusion, db_label, params = {}):

    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP', 'MeltingPoint', 'WaterSolubility']
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 3456)
    
    print('Computing distance matrix', ctime())
    ham_pub_matr = ham_pub_matrix(db_mortality, categorical,
                                  a_ham = 0.07498942093324558, a_pub = 0.5623413251903491)
    
    euc_matrix = euclidean_matrix(db_mortality, non_categorical)
    euc_matrix = pd.DataFrame(euc_matrix)

    accs = []
    sens = []
    specs = []
    f1s = []
    
    i = 0
    for train_index, test_index in kf.split(db_mortality):
        
        i+=1
        max_euc = euc_matrix.iloc[train_index, train_index].values.max()
        
        print('Epoch {}: '.format(i), ctime())
        dist_matr = pd.DataFrame(ham_pub_matr + euc_matrix.divide(max_euc).values)
    
        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = db_mortality['conc1_mean'].iloc[train_index].copy().values
        y_test = db_mortality['conc1_mean'].iloc[test_index].copy().values
        
        del dist_matr
        
        # qua ho le matrici di distanza (train x train) e (test x train)
        # e applico il simple rasar
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar(X_train, X_test,
                                                                     db_mortality.iloc[train_index],
                                                                     db_mortality.iloc[test_index],
                                                                     y_train, y_test)
        del X_train, X_test
        # creo il dataset con i dati del datafusion
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar(db_mortality.iloc[train_index],
                                                                                 db_mortality.iloc[test_index],
                                                                                 db_datafusion, max_euc)
        
        train_rf = pd.concat([simple_rasar_train[['dist_neigh0', 'dist_neigh1']],
                              datafusion_rasar_train, 
                              db_label.iloc[train_index].reset_index(drop=True)], axis = 1)
        
        test_rf = pd.concat([simple_rasar_test[['dist_neigh0', 'dist_neigh1']],
                             datafusion_rasar_test, 
                             db_label.iloc[test_index].reset_index(drop=True)], axis = 1)
        
        if params == {}:
            clf = RandomForestClassifier(n_estimators = 300, n_jobs = -2)
        else:
            clf = RandomForestClassifier(n_jobs = -1)
            for k,v in params.items():
                setattr(clf, k, v)
                
        clf.fit(train_rf, y_train)

        y_pred = clf.predict(test_rf)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accs.append(accuracy_score(y_test, y_pred))
        sens.append(recall_score(y_test, y_pred))
        specs.append(tn/(tn+fp))
        f1s.append(f1_score(y_test, y_pred))
        
    
    print('Accuracy:   ', np.mean(accs),  'se:', sem(accs))
    print('Sensitivity:', np.mean(sens),  'se:', sem(sens))
    print('Specificity:', np.mean(specs), 'se:', sem(specs))
    print('F1 score:   ', np.mean(f1s),   'se:', sem(f1s))
    
    return


#######################################################################################################
##################### TUNING Hyper-Parameters #########################################################
##################### Use only with 66% of dataset ####################################################
#######################################################################################################


def cv_params_datafusion_rasar_alpha(db_mortality, db_datafusion, db_label, hyper_params_tune = {}):
    
    params_comb = list(ParameterSampler(hyper_params_tune, n_iter = 150, random_state = 52))

    test_acc = dict()
    test_sens = dict()
    test_spec = dict()
    test_f1 = dict()


    for i in range(0,len(params_comb)):
        test_acc['mod' + str(i)] = list()
        test_sens['mod' + str(i)] = list()
        test_spec['mod' + str(i)] = list()
        test_f1['mod' + str(i)] = list()

    print(ctime())
    
    categorical = ['class', 'tax_order', 'family', 'genus', "species", 'control_type', 'media_type',
                   'application_freq_unit',"exposure_type", "conc1_type", 'obs_duration_mean']

    non_categorical = ['ring_number', 'tripleBond', 'doubleBond', 'alone_atom_number', 'oh_count',
                       'atom_number', 'bonds_number', 'Mol', 'MorganDensity', 'LogP', 'MeltingPoint', 'WaterSolubility']
    
    kf = KFold(n_splits=5, shuffle=True, random_state = 3456)
    
    print('Computing distance matrix', ctime())
    ham_pub_matr = ham_pub_matrix(db_mortality, categorical,
                                  a_ham = 0.07498942093324558, a_pub = 0.5623413251903491)
    
    euc_matrix = euclidean_matrix(db_mortality, non_categorical)
    euc_matrix = pd.DataFrame(euc_matrix)

    accs = []
    sens = []
    specs = []
    f1s = []
    
    n_epoch = 0
    for train_index, test_index in kf.split(db_mortality):
        n_epoch+=1
        max_euc = euc_matrix.iloc[train_index, train_index].values.max()
        
        print('Epoch {}: '.format(n_epoch), ctime())
        dist_matr = pd.DataFrame(ham_pub_matr + euc_matrix.divide(max_euc).values)
    
        X_train = dist_matr.iloc[train_index, train_index]
        X_test = dist_matr.iloc[test_index, train_index]
        y_train = db_mortality['conc1_mean'].iloc[train_index].copy().values
        y_test = db_mortality['conc1_mean'].iloc[test_index].copy().values
        
        del dist_matr
        
        simple_rasar_train, simple_rasar_test = unsuper_simple_rasar(X_train, X_test,
                                                                     db_mortality.iloc[train_index],
                                                                     db_mortality.iloc[test_index],
                                                                     y_train, y_test)
        del X_train, X_test
        # creo il dataset con i dati del datafusion
        datafusion_rasar_train, datafusion_rasar_test = unsuper_datafusion_rasar(db_mortality.iloc[train_index],
                                                                                 db_mortality.iloc[test_index],
                                                                                 db_datafusion, max_euc)
        
        train_rf = pd.concat([simple_rasar_train[['dist_neigh0', 'dist_neigh1']],
                              datafusion_rasar_train, 
                              db_label.iloc[train_index].reset_index(drop=True)], axis = 1)
        
        test_rf = pd.concat([simple_rasar_test[['dist_neigh0', 'dist_neigh1']],
                             datafusion_rasar_test, 
                             db_label.iloc[test_index].reset_index(drop=True)], axis = 1)
        
        for i in range(0, len(params_comb)):
            clf = RandomForestClassifier(n_jobs = -1)
            for k,v in params_comb[i].items():
                setattr(clf, k, v)
            clf.fit(train_rf, y_train)

            y_pred = clf.predict(test_rf)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            test_acc['mod' + str(i)].append(accuracy_score(y_test, y_pred))
            test_sens['mod' + str(i)].append(recall_score(y_test, y_pred))
            test_spec['mod' + str(i)].append(tn/(tn+fp))
            test_f1['mod' + str(i)].append(f1_score(y_test, y_pred))
    
    print(ctime())
            
    tab_rf_rasar = pd.DataFrame(columns = ['test_acc', 'test_sens', 'test_spec', 'test_f1'])

    tab_rf_rasar.loc[:,'test_acc'] = pd.DataFrame(test_acc).mean(axis = 0)
    tab_rf_rasar.loc[:,'test_sens'] = pd.DataFrame(test_sens).mean(axis = 0)
    tab_rf_rasar.loc[:,'test_spec'] = pd.DataFrame(test_spec).mean(axis = 0)
    tab_rf_rasar.loc[:,'test_f1'] = pd.DataFrame(test_f1).mean(axis = 0)

    params_df = pd.DataFrame(params_comb, index = ['mod' + str(i) for i in range(0,len(params_comb))])
    tab_rf_rasar = pd.concat([params_df, tab_rf_rasar], axis = 1)
    
    return tab_rf_rasar

















