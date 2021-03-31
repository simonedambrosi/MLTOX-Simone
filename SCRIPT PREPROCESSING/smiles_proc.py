from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDConfig
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdchem
from rdkit.Chem import MolFromSmarts
from rdkit.Chem import MolFromSmiles

import cirpy
import pubchempy as pcp

import pandas as pd
import numpy as np

from time import ctime

def atom_number(smile):
    '''
    Given the Smile, this function counts the number of atoms for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - number of atoms (int): number of atoms in the chemical
    '''
    return sum(1 for c in smile if c.isupper())

def alone_atom_number(smile):
    '''
    Given the Smile, this function counts the number of atoms, which are seperated from other atoms, in each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - number of separeted atoms (int): number of separeted atoms in the chemical 
    ''' 
    return smile.count('[') 

def count_doubleBond(smile):
    '''
    Given the smile, this function count the number of double bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - number of double bonds (int): number of double bonds in the chemical 
    '''
    return smile.count('=') 

def count_tripleBond(smile):
    '''
    Given the smile, this function count the number of triple bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - number of triple bonds (int): number of triple bonds in the chemical 
    '''
    return smile.count('#') 
    
def bonds_number(smile):
    '''
    Given the smile, this function count the number of bonds for each chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - bonds number (int): number of bonds in the chemical (NaN if not found)
    '''
    m = Chem.MolFromSmiles(smile)
    try:
        return rdchem.Mol.GetNumBonds(m)
    except:
        return 'NaN'
    
def ring_number(smile):
    '''
    Given the smile, this function count the number of ring in each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - ring_number (int): number of ring in the chemical (NaN if not found)
    '''
    m = Chem.MolFromSmiles(smile)
    try:
        f = rdchem.Mol.GetRingInfo(m)
        return f.NumRings()
    except:
        return 'NaN'

def Mol(smile):
    '''
    Given the smile, this function compute the Mol for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - Mol (float): mol number of the chemical (NaN if not found)
    '''
    
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Descriptors.MolWt(m)
    except:
        return 'NaN'
    
def MorganDensity(smile):
    '''
    Given the Smile, this function compute the Morgan density for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - Morgan Density (float): Morgan density of the chemical (NaN if not found)
    '''
    smile = str(smile)
    m = Chem.MolFromSmiles(smile)
    try:
        return Descriptors.FpDensityMorgan1(m)
    except:
        return 'NaN'

def LogP(smile):
    '''
    Given the Smile, this function compute the partition coefficient for each Chemical
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - LogP (float): partition coefficient of the chemical (NaN if not found)
    '''
    smile = str(smile)
    try:
        m = Chem.MolFromSmiles(smile)
        return Crippen.MolLogP(m)
    except:
        return 'NaN'    

def OH_count(smile):
    '''
    Given the SMILES, this function compute the number of OH group in the chemical.
    Inputs:
        - smiles (str)
    Outputs:
        - Count of OH group in the molecule (int) (NaN if not found)
    '''
    try:
        m = MolFromSmiles(smile)
        patt = MolFromSmarts('[OX2H]')
        return len(m.GetSubstructMatches(patt))
    except:
        return 'NaN'
        
def to_cas(num):
    ''' 
    Transform an integer CAS into a CAS Registry Number (format XXXXX-XX-X)
    Inputs:
        - num (int): CAS in integer format
    Outputs:
        - Cas_number (string): cas in format XXXXXX-XX-X
    '''
    Cas_number = str(num)
    Cas_number = Cas_number[:-3]+ '-' + Cas_number[-3:-1] +'-' + Cas_number[-1]
    return Cas_number

def find_pubchem(smiles):
    '''
    Find PubChem2D Fingerprint using PubchemPy
    Inputs:
        - smile (string): original SMILES code
    Outputs:
        - PubChem2D (string): pubchem2d fingerprint in string format (NaN if not found)
    '''
    try:
        return pcp.get_compounds(smiles, 'smiles')[0].cactvs_fingerprint
    except:
        return 'NaN'

def find_smiles(cas):
    '''
    Find SMILES representation using CirPy
    Inputs:
        - Cas_number (str): CAS number in original format
    Outputs:
        - SMILES (str): original SMILES code
    '''

    return cirpy.resolve(cas, 'smiles')

def adding_smiles_features(dataframe):
    '''
    Given the SMILES, this function apply all other function to extract each molecular descriptors. 
    As regarding MP and WS, the paths of CompTox files have to be replaced!
    Input: 
        - SMILES (str)
    Output:
        - All molecular descriptors.
    '''
    
    print("Finding atom number...")
    dataframe['atom_number'] = dataframe['smiles'].apply(atom_number)

    print("Finding number of alone atoms...")
    dataframe['alone_atom_number'] = dataframe['smiles'].apply(alone_atom_number)

    print("Finding single bounds number...")
    dataframe['bonds_number'] = dataframe['smiles'].apply(bonds_number)

    print("Finding double bounds number...")
    dataframe['doubleBond'] = dataframe['smiles'].apply(count_doubleBond)

    print("Finding triple bounds number...")
    dataframe['tripleBond'] = dataframe['smiles'].apply(count_tripleBond)

    print("Finding ring number...")
    dataframe['ring_number'] = dataframe['smiles'].apply(ring_number)

    print("Finding Molecular Weight...")
    dataframe['Mol'] = dataframe['smiles'].apply(Mol)

    print("Finding morgan density...")
    dataframe['MorganDensity'] = dataframe['smiles'].apply(MorganDensity)

    print("Finding partition number (LogP)...")
    dataframe['LogP'] = dataframe['smiles'].apply(LogP)
    
    print("Finding number of OH group...")
    dataframe['oh_count'] = dataframe['smiles'].apply(OH_count)
    
    print("Finding Melting Point and Water Solubility using CompTox Database...")
    print("Loading and merging CompTox Database...", ctime())
    
    
    ###########################################################################
    # Paths needs to be replaced whenever files are not in the folder #
    ###########################################################################
    
    properties = pd.concat([pd.read_excel(open("data/DSSToxQueryWPred1.xlsx", 'rb'), # here
                  usecols =  ['Substance_CASRN', 'NCCT_MP', 'NCCT_WS'], engine = 'openpyxl'),
                  pd.read_excel(open("data/DSSToxQueryWPred2.xlsx", 'rb'), # here
                  usecols =  ['Substance_CASRN', 'NCCT_MP', 'NCCT_WS'], engine = 'openpyxl'),
                  pd.read_excel(open("data/DSSToxQueryWPred3.xlsx", 'rb'), # here
                  usecols =  ['Substance_CASRN', 'NCCT_MP', 'NCCT_WS'], engine = 'openpyxl'),
                  pd.read_excel(open("data/DSSToxQueryWPred4.xlsx", 'rb'), # here
                  usecols =  ['Substance_CASRN', 'NCCT_MP', 'NCCT_WS'], engine = 'openpyxl')], axis = 0)
    
    
    dataframe = dataframe.merge(properties, left_on = 'test_cas', right_on = 'Substance_CASRN')
    dataframe.drop(columns = 'Substance_CASRN', inplace = True)
    dataframe.rename(columns = {'NCCT_MP': 'MeltingPoint', 'NCCT_WS': 'WaterSolubility'}, inplace = True)
    
    print('End Molecular Descriptors extraction!', ctime())
    
    return dataframe