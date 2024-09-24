# from collections import Counter
import torch
import torch.nn.functional as functional
from torch.utils.data import TensorDataset, DataLoader
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
from codes.language_model_handlers.ml_based_language_model_handler import MachineLearningLanguageModelHandler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from icecream import ic


class EnsembleFeaturesLlm():
    
    def __init__(self, features_names=None):
        
        ## LLM parameters 
        self.max_length = 128
        self.batch_size = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## ML parameters
        if features_names:
            self.features_names = features_names
        
        self.embeddings_for_ml_models = None
    
    def compute_metrics(self, predictions, labels):
    
        average_mode = 'weighted'
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average_mode),
            'recall': recall_score(labels, predictions, average=average_mode),
            'f1': f1_score(labels, predictions, average=average_mode)
        }
    

    def tokenize_dataset(self, data, tokenizer):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=self.max_length, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    def prepare_dataset(self, data, tokenizer, shuffle=True):
        """
        Prepares a dataset for training or evaluation by tokenizing the input texts,
        converting the labels to tensors, and creating a DataLoader.

        Args:
            data (pd.DataFrame): The input data containing text and label columns.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to convert text to tokens.
            shuffle (bool): Whether to shuffle the dataset. Default is True.

        Returns:
            DataLoader: A DataLoader object for the dataset.

        Attributes:
            text_column (str): The name of the column in `data` containing the text data.
            label_column (str): The name of the column in `data` containing the labels.
            device (torch.device): The device (CPU/GPU) to move the tensors to.
            batch_size (int): The batch size for the DataLoader.
        """
        # Tokenize the text data
        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list(), tokenizer)

        # Convert labels to LongTensor
        Y = torch.LongTensor(data[self.label_column].to_list())
        
        # Move tensors to the specified device
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        # Create a TensorDataset from the input tensors and labels
        dataset = TensorDataset(input_ids, att_masks, Y)

        # Create a DataLoader with the specified batch size and shuffle option
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return data_loader
    
    def _generate_model_predictions(self, tokenizer, model, data, convert_output_to_probability, 
                                    text_column='text', label_column='label', machine_learning_model=None):
        """
        Makes predictions using a given model and tokenizer on the provided dataset.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer for processing text data.
            model (transformers.PreTrainedModel): The model to use for predictions.
            data (pd.DataFrame): The input data containing text and label columns.
            convert_output_to_probability (bool): Whether to convert the model output to probabilities.
            text_column (str): The name of the column containing text data. Default is 'text'.
            label_column (str): The name of the column containing label data. Default is 'label'.

        Returns:
            tuple: Containing texts, true_vals, predictions, and loss_val_total.
        """

        # Set text and label column names
        self.text_column = text_column
        self.label_column = label_column

        # Prepare the dataset and dataloader
        dataloader_test = self.prepare_dataset(tokenizer=tokenizer, data=data, shuffle=False)

        # Set model to evaluation mode
        model.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        texts = []
        j = 0

        # Iterate through batches of data
        for batch in dataloader_test:
            
            # Move each tensor in the batch to the specified device (CPU/GPU)
            batch = tuple(b.to(self.device) for b in batch)
            
            # Collect the original text for each item in the batch
            for i in range(len(batch[0])):
                texts.append(data.iloc[j][self.text_column])
                j += 1

            # Prepare the inputs for the model
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }

            # Perform forward pass without gradient computation
            with torch.no_grad():
                outputs = model(**inputs)

            # Compute loss and logits
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            # Get true labels
            label_ids = inputs['labels'].cpu().numpy()
            true_vals.append(label_ids)

            # Convert logits to probabilities if specified
            if convert_output_to_probability:
                logits = functional.softmax(logits, dim=-1)
            
            # Detach logits and move to CPU
            logits = logits.detach().cpu().numpy()
            predictions.append(logits)

        return texts, true_vals, predictions, loss_val_total


    
    def default_weighted_average_predictions(self, predictions, weights):
        if len(predictions) != len(weights):
            raise ValueError("The number of predictions and weights must be the same")
        
        # Convert the list of predictions to a numpy array for easier manipulation
        predictions_array = np.array(predictions)
        
        # Perform weighted average
        weighted_preds = np.average(predictions_array, axis=0, weights=weights)
        
        return weighted_preds.tolist()
    
    def dynamic_weighted_average_predictions(self, predictions, weights_dictionary):
        if len(predictions) != len(weights_dictionary):
            raise ValueError("The number of predictions and weights must be the same")
        
        # Convert the list of predictions to a numpy array for easier manipulation
        predictions_array = np.array(predictions)

        predictions = []

        for i, model_name in enumerate(weights_dictionary):
            weights = np.array(list(weights_dictionary[model_name].values()))
            
            weighted_preds = predictions_array[i]*weights
            
            predictions.append(weighted_preds)
        
        return np.average(np.array(predictions), axis=0)
            

    def accuracy_per_class(self, true_labels, predictions):
        """
        Calculate the accuracy per class.

        Parameters:
        true_labels (array-like): Array of true labels with shape (num_samples,)
        predictions (array-like): Array of predicted probabilities or class labels with shape (num_samples, num_classes) or (num_samples,)

        Returns:
        dict: A dictionary with class labels as keys and their respective accuracies as values
        """
        # If predictions are probabilities, convert them to class labels
        if len(predictions.shape) == 2:
            predicted_labels = np.argmax(predictions, axis=1)
        else:
            predicted_labels = predictions
        
        # Initialize variables to store the number of correct predictions and the total number of instances per class
        class_correct_predictions = {}
        class_total_instances = {}
        
        # Get the unique class labels
        unique_classes = np.unique(true_labels)
        
        for cls in unique_classes:
            # Identify instances of the current class
            class_indices = (true_labels == cls)
            
            # Calculate the number of correct predictions for the current class
            correct_predictions = np.sum(predicted_labels[class_indices] == true_labels[class_indices])
            
            # Calculate the total number of instances for the current class
            total_instances = np.sum(class_indices)
            
            # Store the results
            class_correct_predictions[cls] = correct_predictions
            class_total_instances[cls] = total_instances
        
        # Calculate accuracy per class
        class_accuracies = {cls: class_correct_predictions[cls] / class_total_instances[cls] for cls in unique_classes}
        
        return class_accuracies
    
    def weighted_voting(self, predictions, weights):

        predictions_array = np.array(predictions)

        predictions_array = np.argmax(predictions_array, axis=2) # checking the class with the highest probability per model
        
        def most_frequent_weighted(arr, weights):
            unique, counts = np.unique(arr, return_counts=True)
            max_count_indices = np.where(counts == counts.max())[0]
            
            if len(max_count_indices) == 1:
                return unique[max_count_indices[0]]
            else:
                # There's a tie, use weights to decide
                max_count_values = unique[max_count_indices]
                tied_indices = [np.where(arr == val)[0] for val in max_count_values]
                weighted_sums = [sum(weights[idx] for idx in indices) for indices in tied_indices]
                return max_count_values[np.argmax(weighted_sums)]
        
        transposed_matrix = predictions_array.T
        result = np.apply_along_axis(lambda col: most_frequent_weighted(col, weights), axis=1, arr=transposed_matrix)

        return result

    
    def perform_ensemble_llms(self, data, models_names, ensemble_list, text_column='text', label_column='label', path_saved_models=''):
        """
        Performs ensemble predictions using multiple language models.

        Args:
            data (pd.DataFrame): The input data containing text and label columns.
            models_names (list): List of tuples with model names to be used in the ensemble. Each tuple contains the LLM name and optionally an ML model name.
            ensemble_list (list): list of types of ensemble method to use. Options are:
                                - 'dynamic_weighted_average_predictions'
                                - 'default_weighted_average_predictions'
                                - 'weighted_voting'
            text_column (str): The name of the column containing text data. Default is 'text'.
            label_column (str): The name of the column containing label data. Default is 'label'.
            path_saved_models (str): Path to the directory containing saved models. Default is an empty string.

        Returns:
            tuple: Containing final ensemble predictions and individual model predictions.
        """
        
        predictions = {}
        weights_per_model = {}

        # Iterate over each model specified in models_names
        for llm_name, ml_model_name in models_names:
            # Replace '/' with '_' in LLM name for valid file path
            llm_name = llm_name.replace('/', '_')

            if ml_model_name is not None:
                # Generate predictions using the ML model
                pred = self.generate_predictions_ml(data, ml_model_name, llm_name, text_column, label_column, path_saved_models)
                model_name = llm_name + '+' + ml_model_name  # Combine LLM and ML model names
            else:
                # Generate predictions using the LLM
                pred = self.generate_predictions_llm(data, llm_name, text_column, label_column, path_saved_models)
                pred = np.concatenate(pred, axis=0)
                model_name = llm_name  # Use only LLM name
            
            # ic(pred)  # Log predictions for debugging

            # Store predictions
            predictions[model_name] = pred

            # Calculate and store weights for the model based on accuracy per class
            weights_per_model[model_name] = self.accuracy_per_class(data[label_column].values, predictions[model_name])


        final_predictions = {}

        for ensemble_type in ensemble_list:
            # Different ensemble types
            if ensemble_type == 'dynamic_weighted_average_predictions':
                # Perform dynamic weighted average of predictions
                ensemble_predictions = self.dynamic_weighted_average_predictions(list(predictions.values()), weights_per_model)
            
            elif ensemble_type == 'default_weighted_average_predictions':
                # Perform default weighted average of predictions
                weights_list = [np.mean(list(weights_per_model[name].values())) for name in weights_per_model]
                ensemble_predictions = self.default_weighted_average_predictions(list(predictions.values()), weights=weights_list)
            
            elif ensemble_type == 'weighted_voting':
                # Perform weighted voting
                weights_list = [np.mean(list(weights_per_model[name].values())) for name in weights_per_model]
                # weights_list = [1] * len(models_names)
                ensemble_predictions = self.weighted_voting(list(predictions.values()), weights_list)

            final_predictions[ensemble_type] = ensemble_predictions

        return final_predictions, predictions

    

    def generate_predictions_ml(self, data, ml_model_name, llm_name, text_column, label_column, path_saved_models):
        """
        Generates predictions using a machine learning model and a language model.

        Args:
            data (pd.DataFrame): The input data containing text and label columns.
            ml_model_name (str): The name of the machine learning model to use.
            llm_name (str): The name of the language model to use.
            text_column (str): The name of the column containing text data.
            label_column (str): The name of the column containing label data.
            path_saved_models (str): Path to the directory containing saved models.

        Returns:
            np.ndarray: Predictions generated by the machine learning model.
        """
        
        # Initialize the machine learning language model handler with given parameters
        ml_language_model_handler = MachineLearningLanguageModelHandler(
            llm_name=llm_name,
            ml_model_name=ml_model_name,
            text_column=text_column,
            processed_text_column=text_column,
            label_column=label_column,
            batch_size=64,
            new_labels=[],
            output_hidden_states=True
        )

        # Load the language model from the specified path
        ml_language_model_handler.load_llm_model(path=path_saved_models, name_file=llm_name)
        
        # Load the machine learning model from the specified path
        ml_language_model_handler.load_ml_model(path=path_saved_models, ml_model_name=ml_model_name)

        # Check if embeddings are already generated and cached
        if self.embeddings_for_ml_models is None:
            # Generate embeddings using the language model
            self.embeddings_for_ml_models = (llm_name, ml_language_model_handler.generate_embeddings(data=data))
        else:
            # If cached embeddings are from a different language model, regenerate embeddings
            if self.embeddings_for_ml_models[0] != llm_name:
                self.embeddings_for_ml_models = (llm_name, ml_language_model_handler.generate_embeddings(data=data))

        # Evaluate the machine learning model using the generated embeddings
        _, predictions = ml_language_model_handler.evaluate_ml_model(
            X_test=self.embeddings_for_ml_models[1], 
            y_test=data[label_column],
            convert_to_one_hot_encoding=True
        )

        return predictions

    def generate_predictions_llm(self, data, llm_name, text_column, label_column, path_saved_models):
        """
        Generates predictions using a language model.

        Args:
            data (pd.DataFrame): The input data containing text and label columns.
            llm_name (str): Name of the language model to be used for predictions.
            text_column (str): The name of the column containing text data.
            label_column (str): The name of the column containing label data.
            path_saved_models (str): Path to the directory containing saved models.

        Returns:
            np.ndarray: The predictions from the language model.
        """
        
        # Initialize the language model handler
        language_model_manager = PytorchLanguageModelHandler(
            model_name=llm_name,
            text_column=text_column,
            processed_text_column=None,
            label_column=label_column,
            new_labels=[],
            output_hidden_states=True,
            create_tokenizer=False
        )
        
        # Load the pre-trained language model
        language_model_manager.load_llm_model(path=path_saved_models, name_file=llm_name.replace('/', '_'))

        # Generate predictions using the loaded model
        _, _, predictions, _ = self._generate_model_predictions(
            tokenizer=language_model_manager.tokenizer,
            model=language_model_manager.model,
            data=data,
            convert_output_to_probability=True,
            text_column=text_column
        )

        return predictions


