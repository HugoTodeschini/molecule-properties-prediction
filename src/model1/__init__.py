import argparse
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau

from prepare_data import prepare_smile,prepare_dataframe

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

#Train neural_model_1 to predict P1 class
def train():

    #User can choose the number of neurons by layer of the neural network
    parser = argparse.ArgumentParser()
    parser.add_argument("-neurons", help="number of neurons by layer of the neural network",type=int)
    args = parser.parse_args()

    df_single = pd.read_csv('../dataset_single.csv')
    df_features = prepare_dataframe(df_single)

    #We split our data between training,test and validation datasets
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(df_features.tolist()), df_single['P1'], test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    #We save our validation dataset which will be used to evaluate our model
    X_val.to_csv('../X_val.csv')
    y_val.to_csv('../y_val.csv')

    model1 = neural_model_1(X_train, y_train, X_test, y_test, args.neurons)[0]
    model1.save("model1.keras")

    return({"Model1": model1})

#Evaluates the model's performance 
def evaluate():
    
    X_val = pd.read_csv('../X_val.csv',index_col = 0)
    y_val = pd.read_csv('../y_val.csv',index_col = 0)
    model1 = tf.keras.models.load_model('model1.keras')

    predictions = pd.DataFrame(model1.predict(X_val),columns=['Prediction'])
    predictions["P1"] = 0
    predictions.loc[predictions['Prediction'] > 0.5, "P1"] = 1

    #Calculate and return accuracy, recall, precision and f1_score of the model
    accuracy = accuracy_score(y_val, predictions["P1"])
    recall = recall_score(y_val, predictions["P1"])
    precision = precision_score(y_val, predictions["P1"])
    f1 = f1_score(y_val, predictions["P1"])
    metrics = {"accuracy": accuracy, "recall":recall, "precision":precision,"f1_score": f1}

    return(metrics)

#Function which uses trained model1 to predict P1
def predict():
    
    #User passes as an argument the molecule's "smile" whose P1 it wants to predict
    parser = argparse.ArgumentParser()
    parser.add_argument("-smile", help="smile to predict",type=str)
    args = parser.parse_args()

    model1 = tf.keras.models.load_model('model1.keras')

    smile_array = prepare_smile(args.smile)
    prediction = model1.predict(pd.DataFrame(smile_array).T)[0][0]
    if prediction > 0.5:
        P1_predicted = 1
    else:
        P1_predicted = 0

    return({"P1": P1_predicted})