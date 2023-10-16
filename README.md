# molecule-properties-prediction
Deep learning models to predict molecule properties based on their "smiles"

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

API:

To launch the flask API execute the "api.py" file