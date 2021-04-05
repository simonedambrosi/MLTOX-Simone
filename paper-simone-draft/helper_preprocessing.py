# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from smiles_proc import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer

from time import ctime


# Pre-processing steps:

# - Step 1: Load Data
# - Step 2: Prefiltering (target endpoint LC50 and other for DF RASAR) and Pre-Aggregation 
# - Step 3: Feature Selection and Imputation (removing or imputing)
# - Step 4: Repeated Experiments Aggregation

# - Step 5: Extraction of SMILES, PubChem2D and molecular descriptors from CASRN (from separate chemical table)
# - Step 6: Transformation of chemical features

# - Removing possible duplicates (not required but done)


#######################################################################################################################################

# Step 1: Load Data
# La versione di Ecotox risale al 06/11/2020. Da questo ho usato 3 file denominati, rispettivamente: 
# [i]   species.txt
# [ii]  tests.txt
# [iii] results.txt

# Nel primo file ci sono le informazioni riguardante le specie di animali: tutta la denominazione tassonomica (da domain fino a variety), altre denominazioni tipiche (nome latino o nome comune), e anche il gruppo ecotox (fish, mammalians, birds, algae,...).

# Nel secondo file, \textit{tests.txt}, ci sono le informazioni riguardanti i test: condizioni di laboratorio (exposure, control, application frequency,...), CASRN del composto chimico e l'animale testato. Ogni riferimento tra tabelle si basa su valori interni inutili per i modelli. 
# Supponiamo di avere 2 test identificati attraverso due numeri 1 e 2; di avere due composti identificati come a e b, e di avere due specie identificate come A e B. 
# Nel test 1 è stato testato il composto a sulla specie A; nel test 2 è stato testato il composto b sulla specie B. 
# L'aggregazione delle tabelle delle informazioni su chemicals, species e tests si basa sulle chiavi interne.

# Allo stesso modo, i risultati dei test (tabella \textit{results.txt}) sono stati uniti ai precedenti dal momento in cui in questa tabella è presente il riferimento al test. 
# ESEMPIO: Il risultato con chiave 300 si riferisce al test 1 e quindi aggregato con quest'ultimo.
# In questa tabella sono presenti le seguenti informazioni: durata di osservazione, endpoint, effect, concentration level e type e le relative unità di misura.

# Per informazioni su Melting Point e Water Solubility mi devo riferire ad un altro dataset, CompTox, i cui valori sono predetti!


# Tests table dimensions:  (681605, 122)
# Species table dimensions:  (27125, 15)
# Results table dimensions:  (1006972, 137)


def load_raw_data(DATA_PATH_TESTS, DATA_PATH_RESULTS, DATA_PATH_SPECIES):
    
    print('Start!', ctime())
    print('Loading raw data')
    
    tests = pd.read_csv(DATA_PATH_TESTS, sep = '\|', engine = 'python')
    print('tests table loaded!')
    species = pd.read_csv(DATA_PATH_SPECIES, sep = '\|', engine = 'python')
    print('species table loaded!')
    results = pd.read_csv(DATA_PATH_RESULTS, sep = '\|', engine = 'python')
    print('results table loaded!')
    
    print('Tests table dimensions: ', tests.shape)
    print('Species table dimensions: ', species.shape)
    print('Results table dimensions: ', results.shape)
    
    return tests, species, results

#######################################################################################################################################

# Step 2: Prefiltering and Pre-Aggregation

# Una volta caricati i dati filtro i risultati sia su endpoint (prendo solo EC50 e LC50) che sugli effetti (prendo solo MOR mortalità).
# Tolgo le specie di animali che hanno il gruppo ecotox nullo e restringo sul regno animale solo sui pesci.

# ESEMPIO: ecotox_group assume i valori:  - Fish, Standard test Species
#                                         - Crustaceans, Standard Test Species
#                                         - Insects/Spiders
#                                         - Molluscs
#                                         - ...

# Inoltre, non considero neanche gli embrioni la cui informazione è contenuta nella tabella test, variabile "organism_lifestage".

# A questo punto ho i risultati di tutti i test con la coppia LC50/Mor o EC50/Mor, la tabella dei soli pesci e la tabella dei test filtrati.

# Pre-Aggregation: aggrego i test alle species sulla chiave comune "species_number", e successivamente unisco i risultati dei test aggregando sulla chiave comune coi risultati "test_id".


# Prefiltered dimensions:  (64341, 272)
# 272 potenziali caratteri rilevati su 64341 esperimenti, molti sono ripetuti...


def prefilter(species, tests, results):
    print('Prefiltering Step...', ctime())
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
    
    print('Dimensions after prefiltering step: ', results_prefilter.shape)
    return results_prefilter

#######################################################################################################################################
# Step 3: Feature Selection and Imputation
# Per decidere quali feature estrarre ho usato la funzione nel general_helper.py denominata "null_output_counts" che prendendo in input un pandas dataframe calcola la percentuale di valori nulli (Null, Not Available e Not Reported) e il numero di modalità che possono assumere.

# Ho ristretto quindi alle sole features che avessero meno del 30% di valori nulli e più di 1 singola modalità. Se avessi imposto una percentuale di valori mancanti maggiore (50%) le features aggiuntive avrebbero dimezzato il dataset dopo l'eliminazione dei record con valori mancanti, riducendolo alla fine a troppe poche osservazioni. 
# Imponendo, invece, il 30% come soglia si ottiene un quantitativo di informazione sufficiente.

# Le variabili più interessanti sono nella lista "keep_columns" (19 variabili).

# Successivamente alla selezione, ho proceduto all'imputazione o rimozione dei valori mancanti.


# CONCENTRATION FEAT.
# Ho iniziato questa fase dalle varibili riguardanti la concentrazione (conc1_mean è la variabile target, di tipo stringa).

# - Se la variabile target ha un valore nullo, il record viene eliminato.
# - Se la variabile target ha un asterisco "*" all'interno del valore, l'asterisco viene tolto e il valore ripristinato.
# - Se la variabile target assume il valore ">100000", questo viene tolto (1 solo caso).
# - La variabile target viene trasformata in tipo float.

# - Se la variabile 'conc1_unit', ossia l'unità di misura della concentrazione, ha all'interno della misura le lettere "AI" (Active Ingredient), queste vengono rimosse, e l'unità di misura viene ripristinata.
# - I valori della variabile target vengono trasformati tutti in un'unica unità di misura, ossia mg/L, applicando le dovute trasformazioni.
# - La variabile 'conc1_unit' (tutta mg/L) perde di significato e viene rimossa.

# - Se il tipo di concentrazione (Active ingredient, dissolved, formulation,...) ha un valore Not Coded (NC) o Not Reported, l'intero record viene tolto.


# TEST FEAT.
# Le variabili da imputare/aggiustare sono: "exposure_type", "test_location", "control_type", "media_type" e "application_freq_unit".
# In ordine, per exposure_type:
# - I valori con il simbolo "/" vengono ripristinati senza questo simbolo.
# - I valori che assumono la modalità "AQUA - NR", ossia esposizione in acqua non riportata, vengono imputati con il generico valore "AQUA".
# - I valori Not Reported (circa 5000, ossia lo 0.089% del totale) vengono imputati con il valore generico "AQUA" ossia esposizione in acqua.

# Per test_location:
# - Questa variabile viene rimossa perchè sbilanciata: più del 98% del totale di esperimenti sono stati eseguiti in laboratorio, mentre il restante 2% sono stati eseguiti su un "suolo artificiale" o "naturale".

# Per control_type:
# - I valori con il simbolo "/" vengono ripristinati senza questo simbolo.
# - Creata classe generica "Unknown" per non scartare questa variabile.

# Per media_type:
# - I valori con il simbolo "/" vengono ripristinati senza questo simbolo.
# - Se questa variabile assume una modalità tra ['NR', 'CUL', 'NONE', 'NC'], dove CUL sta per Culture, l'intero record viene rimosso.

# Per application_freq_unit:
# - I valori mancanti, Not Reported e Not Coded, vengono imputati con la classe più frequente: "X", ossia la concentrazione del composto è unitaria (il composto viene dato tutto in una sola volta al pesce).


# DURATION FEAT.
# Per obs_duration_mean, unica feature sulla durata dell'esperimento:
# - Con l'aiuto di obs_duration_unit, gli esperimenti hanno tutti la durata compresa tra [24, 48, 72, 96] ore.


# SPECIES FEAT.
# - Le features "species" e "genus" hanno i valori nulli nelle stesse posizioni. Eliminando i record si ottiengono queste due caratteristiche pulite.
# - Stesso discorso per "family", "tax_order" e "class".



# Alla fine di questa fase, si hanno 50160 esperimenti e 16 caratteristiche utili.


def select_impute_features(prefiltered_results):
    
    print('Selection and Imputation Step...', ctime())
          
    keep_columns = ['obs_duration_mean', 'obs_duration_unit',
       'endpoint', 'effect', 'measurement', 'conc1_type', 'conc1_mean',
       'conc1_unit', 'test_cas', 'test_location', 'exposure_type',
       'control_type', 'media_type', 'application_freq_unit', 'class',
       'tax_order', 'family', 'genus', 'species']
    
    db = prefiltered_results.copy()
    db = db[keep_columns]
    
    print('Features:')
    for i in keep_columns:
          print(i, end = ', ')
    print()
    db = impute_conc(db)
    
    db = impute_test_feat(db)
    
    db = impute_duration(db)
    
    db = impute_species(db)
    
    print('Dimension after imputation step:', db.shape)
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


#######################################################################################################################################
# Step 4: Repeated Experiment Aggregation
# Gli esperimenti sono ripetuti, ma come considerare un esperimento come ripetuto?
# Ho deciso che due esperimenti sono ripetuti se condividono le seguenti caratteristiche:
# - Stesso pesce: ho unito tutta il nome tassonomico del pesce in un'unica feature, e ho creato un dataset a parte per l'aggregazione successiva.
# - Stesso composto: ossia il CASRN
# - Stessa durata, tipo di esposizione, controllo finale, acqua in cui è stato testato il pesce, frequenza di applicazione.

# Raggruppando in base a queste caratteristiche, ho preso il valore mediano e riaggregato l'informazione riguardante i pesci.

# Si perdono le informazioni endpoint, effect e measurement. Le prime due è ovvio, tutti gli esperimenti si riferiscono a LC50 o EC50, con effetto Mortality. "measurement" non ha particolare rilevanza dato che il 99% dei valori assumono la modalità "MORT", ossia mortalità. Circa un centinaio di esperimenti sono stati "misurati" con la sopravvivenza, "SURV".

# Alla fine di questa aggregazione si hanno 28815 esperimenti (aggregati), con 14 caratteristiche rilevate (compresa l'aggregazione della tassonomia --> "fish").


def repeated_experiments(imputed_db):
          
    print('Aggregation of repeated experiments', ctime())
    db = imputed_db.copy()
    db['fish'] = db['class'] + ' ' + db['tax_order'] + ' ' + db['family'] + ' ' + db['genus'] + ' ' + db['species']
    
    db_species = db[['class', 'tax_order', 'family', 'genus', "species", 'fish']]
    db_species = db_species.groupby("fish").first()
    
    final_db = db.groupby(by = ['test_cas', 'obs_duration_mean', 'conc1_type', 'fish', 'exposure_type',
                     'control_type', 'media_type', 'application_freq_unit']).agg('median').reset_index()
    final_db = final_db.merge(db_species, on='fish')
    
    print('Dimension after repeated experiments aggregation:', final_db.shape)
    return final_db

#######################################################################################################################################
''' Step 5: Extraction of SMILES, PubChem2D and molecular descriptors from CASRN 
Uso il file "smiles_proc.py" per estrarre tutte le informazioni dal CASRN.

Se si vuole evitare di calcolarsi ex-novo tutti gli smiles, i pubchem e i descrittori molecolari (operazione di circa 2 ore) si può usare la funzione alternativa "process_chemicals" che prende in input un dataset con già queste info estratte.
'''

def extract_chemical_data(final_db):
    chem = pd.DataFrame(final_db.test_cas.unique(), columns = ['test_cas'])
    
    print('Remark: Stable internet connection required for smiles and pubchem extraction!')
        
    print('Smiles extraction...', ctime())
    chem['smiles'] = chem.test_cas.apply(find_smiles)
    chem['smiles'].fillna('NaN', inplace = True)
    
    print('Pubchem2d extraction...', ctime())
    chem['pubchem2d'] = chem.smiles.apply(find_pubchem)
        
    print('Molecular Descriptors extraction...', ctime())
    chem = adding_smiles_features(chem)
    
    to_drop_nofeat = chem[(chem == 'NaN').any(axis = 1)].index
    chem.drop(index = to_drop_nofeat, inplace = True)
    
    to_drop_null = chem[chem.isnull().any(axis = 1)].index
    chem.drop(index = to_drop_null, inplace = True)
    
    return chem


def process_chemicals(DATA_PATH_CHEMICAL_FEATURES):
    
    chem_feat = pd.read_csv(DATA_PATH_CHEMICAL_FEATURES).drop(columns = ['Unnamed: 0'])
    
    chem_feat = adding_smiles_features(chem_feat)
    to_drop_nofeat = chem_feat[chem_feat['bonds_number'] == 'NaN'].index
    chem_feat.drop(index = to_drop_nofeat, inplace = True)
    
    to_drop_null = chem_feat[chem_feat.isnull().any(axis = 1)].index
    chem_feat.drop(index = to_drop_null, inplace = True)
    
    return chem_feat

#######################################################################################################################################
# Step 4: Transformation of chemical features

# Alcune variabili necessitano di trasformazioni per regolarizzare le proprie distribuzioni. Queste caratteristiche sono: "bonds_number", "atom_number" e "Mol", in aggiunta anche "WaterSolubility".
# La trasformazione è logaritmica e poi MinMax. 
# Per "WaterSolubility" ho usato la trasformazione Box-Cox per normalizzare la distribuzione.

def process_smiles_features(chemical_features):
    
    db = chemical_features.copy()
    
    # Bonds Number
    db.bonds_number = db.bonds_number.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["bonds_number"]])
    db[["bonds_number"]] = minmax.transform(db[["bonds_number"]])
    
    # Atom Number
    db.atom_number = db.atom_number.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["atom_number"]])
    db[["atom_number"]] = minmax.transform(db[["atom_number"]])
    
    # Molecular Weight
    db.Mol = db.Mol.apply(lambda x: np.log1p(x))
    
    minmax = MinMaxScaler()
    minmax.fit(db[["Mol"]])
    db[["Mol"]] = minmax.transform(db[["Mol"]])
    
    # Water Solubility
    pt = PowerTransformer(method = 'box-cox')

    pt.fit(db.WaterSolubility.values.reshape(-1, 1))
    db[['WaterSolubility']] = pt.transform(db.WaterSolubility.values.reshape(-1, 1)).ravel()
    
    return db







