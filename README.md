# molecule-properties-prediction
Deep learning models to predict molecule properties based on their "smiles"

Command:

Package
python setup.py install
servier_train -neurons 64
servier_evaluate
servier_predict -smile "Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1"