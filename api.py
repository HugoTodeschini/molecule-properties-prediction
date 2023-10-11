import tensorflow as tf
import pandas as pd
from flask import Flask, jsonify
from prepare_data import prepare_smile

model1 = tf.keras.models.load_model('model1.keras')

app = Flask(__name__)

@app.route('/predict/<smile>', methods=['POST'])
def predict_P1(smile):
    smile_array = prepare_smile(smile)
    prediction = model1.predict(pd.DataFrame(smile_array).T)[0][0]
    if prediction > 0.5:
        P1_predicted = 1
    else:
        P1_predicted = 0
    return(jsonify({"P1_predicted": P1_predicted}))


@app.route('/', methods=['GET'])
def index():
    return("test")

if __name__ == "__main__":
    app.run()