# molecule-properties-prediction
Deep learning models to predict molecule properties based on their "smiles"

Description: This python package allows you to train and test a deep learning model that predicts the P1 property of molecules from their “smile”.

Prerequisites:

To use this package put the "dataset_single.csv" file int folder which contains this package.
This package has been develpped with python 3.6.13

Installation:

Install all packages with the command:
pip install -r requirements.txt

To package the code use "python setup.py install" in the command line at the root of the project

You can now use 3 commands:

The command "servier_train" which allows you to train a model to predict P1 property.
You must pass as an argument a number which will determine the number of neurons that each layer of the network has.
ex: "servier_train -neurons 64"

Then you can use "servier_evaluate" to evaluate your model performances.
This command will show you the accuracy, precision, recall and f1_score of your model on a validation dataset.
There is not argumants on this command.
ex: "servier_evaluate"

You can also use "servier_predict" to make a prediction with your model.
Pass as an argument the smile of the molecule on which you want to make the prediction.
ex: "servier_predict -smile "Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1""

Flask API:

To launch the flask API of the model execute the "api.py" file with the "python api.py" command.
The api has one route "predict" which allows you to make a prediction with the trained model.
Once the api launched you can try it with the command: "curl -X POST http://127.0.0.1:5000/predict/NC(=O)NC(Cc1ccccc1)C(=O)O".
You can replace "NC(=O)NC(Cc1ccccc1)C(=O)O" with the smile of the molecule that you want.
The API will return the predicted P1 property of the molecule.

Docker:

You can build and launch the docker image of the projetct with "docker build -t servier ." and "docker run -p 5000:5000 servier"
The docker image launch the Flask API.
You can try it with "curl -X POST http://127.0.0.1:5000/predict/NC(=O)NC(Cc1ccccc1)C(=O)O"