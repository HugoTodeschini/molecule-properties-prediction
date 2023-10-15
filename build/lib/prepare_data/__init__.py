import numpy as np
import pandas as pd

from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops

def fingerprint_features(smile_string, radius=2, size=2048):
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                          nBits=size,
                                                          useChirality=True,
                                                          useBondTypes=True,
                                                          useFeatures=False
                                                          )

def prepare_dataframe(df_data):
    df_features = df_data['smiles'].apply(fingerprint_features)
    df_features = df_features.apply(np.array)
    return(df_features)

def prepare_smile(smile : str):
    return(np.array(fingerprint_features(smile)))