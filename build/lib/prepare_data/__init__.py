import numpy as np

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

#Function to convert a dataframe with molecule's smile in a dataframe to train a model
def prepare_dataframe(df_data):
    df_features = df_data['smiles'].apply(fingerprint_features)
    df_features = df_features.apply(np.array)
    return(df_features)

#Function to convert smile in a numpy array
def prepare_smile(smile : str):
    return(np.array(fingerprint_features(smile)))