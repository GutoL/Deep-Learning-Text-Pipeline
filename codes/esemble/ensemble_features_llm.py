# from collections import Counter
import torch
import torch.nn.functional as functional
from torch.utils.data import TensorDataset, DataLoader
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
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
    
    def tokenize_dataset(self, data, tokenizer):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=self.max_length, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    def prepare_dataset(self, data, tokenizer, shuffle=True):

        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list(), tokenizer)

        Y = torch.LongTensor(data[self.label_column].to_list())
        
        # move on device (GPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        dataset = TensorDataset(input_ids, att_masks, Y)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return data_loader
    
    def model_prediction(self, tokenizer, model, data, convert_output_to_probability, text_column='text', label_column='label'):
        self.text_column = text_column
        self.label_column = label_column

        dataloader_test = self.prepare_dataset(tokenizer=tokenizer, data=data, shuffle=False)

        model.eval()
    
        loss_val_total = 0
        predictions, true_vals = [], []

        texts = []
        
        j=0

        for batch in dataloader_test:
            
            batch = tuple(b.to(self.device) for b in batch)       
            
            for i in range(len(batch[0])):
                
                texts.append(data.iloc[j][self.text_column])                

                j += 1

            inputs = {
                        'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[2],
                     }
    
            with torch.no_grad():        
                outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()
    
            label_ids = inputs['labels'].cpu().numpy()
            true_vals.append(label_ids)

            if convert_output_to_probability:
                logits = functional.softmax(logits, dim=-1)
                
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

    def compute_metrics(self, eval_pred):
        
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        average_mode = 'weighted'
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average_mode),
            'recall': recall_score(labels, predictions, average=average_mode),
            'f1': f1_score(labels, predictions, average=average_mode)
        }
    
    def perform_ensemble_llms(self, data, models_names, dynamic_weights):

        text_column = 'text'
        label_column = 'label'
        path = ''

        predictions = {}
        weights_per_model = {}

        for model_name in models_names:
            language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                                            text_column = text_column,
                                                            processed_text_column=None,
                                                            label_column=label_column,
                                                            new_labels=[],
                                                            output_hidden_states=True)
            
            language_model_manager.load_model(path=path+'saved_models/', name_file='multi_class_'+model_name.replace('/', '_'))

            _, _, pred, _ = self.model_prediction(tokenizer=language_model_manager.tokenizer, model=language_model_manager.model, 
                                         data=data, convert_output_to_probability=True, text_column=text_column)
            
            predictions[model_name] = np.concatenate(pred, axis=0)

            weights_per_model[model_name] = self.accuracy_per_class(data[label_column], predictions[model_name])

        if dynamic_weights:
            final_predictions = self.dynamic_weighted_average_predictions(list(predictions.values()), weights_per_model)
        else:
            weights_list = []
            for name in weights_per_model:
                weights_list.append(np.mean(list(weights_per_model[name].values())))
            
            # weights_list = [1,1,1]
            ic(weights_list)

            final_predictions = self.default_weighted_average_predictions(list(predictions.values()), weights=weights_list)

        return final_predictions, predictions   

        

    # def perform_ensemble_ml_and_llm(self, data, llm_tokenizer, llm, feature_ml):

    #     self.tokenizer = llm_tokenizer
        
    #     _, _, llm_predictions, _ = self.model_prediction(model=llm, data=data, convert_output_to_probability=True, text_column='text')
    #     _, _, ml_predictions = feature_ml.evaluate_model(data)


    #     llm_predictions = np.concatenate(llm_predictions, axis=0)
        
    #     # llm_predictions = np.argmax(llm_predictions, axis=1)
    #     # ml_predictions = np.argmax(ml_predictions, axis=1)

    #     predictions = [llm_predictions.tolist(), ml_predictions]

    #     final_predictions = self.weighted_average_predictions(predictions=predictions, weights=[0.6, 0.4]) # LLM ML

    #     return final_predictions

    # def weighted_average_predictions(self, predictions, weights):
        
    #     if len(predictions) != len(weights):
    #         raise ValueError("The number of predictions and weights must be the same.")
        
    #     num_models = len(predictions)
    #     num_samples = len(predictions[0])
        
    #     # Ensure all prediction lists are the same length
    #     for pred in predictions:
    #         if len(pred) != num_samples:
    #             raise ValueError("All prediction lists must be of the same length.")
        
    #     # Initialize a list to hold the final predictions
    #     final_predictions = []

    #     # Transpose the list of predictions to iterate over each sample
    #     transposed_predictions = list(zip(*predictions))
        
    #     # Iterate over each sample's predictions
    #     for sample_predictions in transposed_predictions:
    #         weighted_counts = Counter()
    #         for i in range(num_models):
    #             weighted_counts[sample_predictions[i]] += weights[i]
            
    #         # Find the class with the highest weighted count
    #         final_prediction = weighted_counts.most_common(1)[0][0]
    #         final_predictions.append(final_prediction)

    #     return final_predictions
    