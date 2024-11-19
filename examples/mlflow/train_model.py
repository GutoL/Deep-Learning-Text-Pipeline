
import sys
sys.path.append("/home/guto/Documents/GitHub/Deep-Learning-Text-Pipeline/")

import statistics
import torch
import torch.nn.functional as functional
import gc

import torch
from transformers import AutoTokenizer  # Adjust based on the tokenizer you're using

from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

import mlflow
import mlflow.pytorch


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name, model, text_column):
        self.model_name = model_name
        self.model = model
        self.text_column = text_column

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, context, input_df):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Tokenize input DataFrame text
        tokens = tokenizer(
            input_df[self.text_column].tolist(), 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Move tokens to the same device as the model
        tokens = {key: value.to(self.device) for key, value in tokens.items()}

        # Set model to evaluation mode and make predictions
        self.model.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokens)
        
        logits = outputs.logits
        logits = functional.softmax(logits, dim=-1)

        return logits # Adjust output handling as needed

def train_log_model(experiment_name, model_name, data_handler, training_parameters, parameters_to_track, save_model):
    
    if model_name is None:
        model_name = 'FacebookAI/roberta-base'  # 'bert-base-uncased'

    print('*** Model:', model_name)


    language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                                        text_column = data_handler.text_column,
                                                        processed_text_column=data_handler.get_text_column_name(),
                                                        label_column=data_handler.label_column,
                                                        new_labels=training_parameters['labels'],
                                                        output_hidden_states=True)


    # model_name = model_name.replace('/', '_')

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        
        # Log experiment metadata
        mlflow.log_param("model_name", model_name)

        metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

        for parameter in parameters_to_track:
            mlflow.log_param(parameter, training_parameters[parameter])

        for metric in metrics:
            mlflow.log_metric(metric, statistics.mean(metrics[metric]))
        
        # run_ids = []
        # artifact_paths = []
        if save_model:
            artifact_path = f"models/{model_name}"
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=ModelWrapper(model_name, model, data_handler.get_text_column_name())
            )
            # run_ids.append(mlflow.active_run().info.run_id)
            # artifact_paths.append(artifact_path)


        print(metrics)