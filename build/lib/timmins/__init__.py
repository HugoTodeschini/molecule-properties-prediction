import argparse
import tensorflow as tf

#from prepare_data import prepare_smile

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', action="store", help='Provides name')
    args = parser.parse_args()

    return("train:",str(args))

def evaluate():
    return("evaluate")

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', action="store", help='Provides name')
    smile = parser.parse_args()

    model1 = tf.keras.models.load_model('model1.keras')

    smile_array = prepare_smile(smile)
    prediction = model1.predict(pd.DataFrame(smile_array).T)[0][0]
    return({"value": prediction})