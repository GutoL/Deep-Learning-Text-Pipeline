import mlflow
logged_model = 'runs:/275ec5bc91954fae90b3b6468e6b14cb/models/FacebookAI/roberta-base'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('test_dataset.csv')

predictions = loaded_model.predict(pd.DataFrame(data.head(5)))

print(predictions)