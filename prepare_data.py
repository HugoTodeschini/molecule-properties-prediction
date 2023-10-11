import numpy as np
import pandas as pd

from feature_extractor import fingerprint_features

def prepare_dataframe(df_data):
    df_features = df_data['smiles'].apply(fingerprint_features)
    df_features = df_features.apply(np.array)
    return(df_features)

def prepare_smile(smile : str):
    return(np.array(fingerprint_features(smile)))