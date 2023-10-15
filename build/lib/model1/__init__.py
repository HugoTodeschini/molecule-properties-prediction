import argparse
import tensorflow as tf
import pandas as pd

from prepare_data import prepare_smile

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', action="store", help='Provides name')
    args = parser.parse_args()

    return("train:",str(args))

def evaluate():
    return("evaluate model 1")

def predict():
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