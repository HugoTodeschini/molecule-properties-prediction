import numpy as np
import pandas as pd

from feature_extractor import fingerprint_features

def prepare_dataframe(df_data):
    df_features = df_data['smiles'].apply(fingerprint_features)
    df_features = df_features.apply(np.array)
    return(df_features)

def prepare_smile(smile : str):
    return(np.array(fingerprint_features(smile)))

"""def split_data(df_data):
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(df_features.tolist()), df_single['P1'], test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    return(df_features)"""

"""def train_model(model):
    return(model)"""

"""def evaluate_model(model):
    return(model)"""