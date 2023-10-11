import numpy as np
import pandas as pd

from feature_extractor import fingerprint_features

from sklearn.model_selection import train_test_split, KFold

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau

df_single = pd.read_csv('../dataset_single.csv')
df_multi = pd.read_csv('../dataset_multi.csv')

df_features = df_single['smiles'].apply(fingerprint_features)
df_features = df_features.apply(np.array)

#We split our data between training,test and validation datasets
X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(df_features.tolist()), df_single['P1'], test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

def neural_model_1(x, y, x_test, y_test, neurons):
    """
    Neural network model
    
    Inputs
    x: descriptors values for training and validation
    y: properties values for training and validation
    x_test: descriptors values for test
    y_test: properties values for test
    
    
    Outputs
    model: trained neural network model
    score: a list with the score values for each fold
    """
    np.random.seed(1)
    score = []
    kfold = KFold(n_splits=5, shuffle=True)
    
    model = Sequential()
    model.add(Dense(neurons, input_dim=x.shape[1], activation='relu'))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    rlrop = ReduceLROnPlateau(monitor='accuracy', factor=0.01, patience=10)
        
    for train, validation in kfold.split(x, y):

        model.fit(x.iloc[train], y.iloc[train], 
                      epochs=100,
                      batch_size=128,
                      callbacks=[rlrop],
                      verbose=0,
                      validation_data=(x.iloc[validation], y.iloc[validation]))

        score.append(model.evaluate(x_test, y_test))
    
    return model, score

model1 = neural_model_1(X_train, y_train, X_test, y_test, 64)[0]

model1.save("model1.keras")