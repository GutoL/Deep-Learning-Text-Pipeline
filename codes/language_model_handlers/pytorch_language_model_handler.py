# https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
# https://www.intodeeplearning.com/bert-multiclass-text-classification/

import torch
import numpy as np 
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch.nn.functional as functional
from tqdm import tqdm
from transformers.trainer_pt_utils import get_parameter_names
import pandas as pd
from transformers import EvalPrediction

import matplotlib.pyplot as plt

from codes.language_model_handlers.language_model_handler import LanguageModelHandler

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Check if the training process should stop early based on validation loss.

        Args:
            validation_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        # If the current validation loss is lower than the minimum recorded loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss  # Update minimum loss
            self.counter = 0  # Reset counter

        # If the validation loss has increased beyond the minimum loss plus a defined delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1  # Increment the counter
            if self.counter >= self.patience:  # Check if patience limit has been reached
                return True  # Indicate that training should stop

        return False  # Indicate that training should continue

class PytorchLanguageModelHandler(LanguageModelHandler):
    def __init__(self, model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, 
                 batch_size=32, text_size_limit=512, create_tokenizer=True):
        super().__init__(model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states, batch_size, text_size_limit)
        self.handler_type = 'pytorch'

        if create_tokenizer:
            self.create_tokenizer()
    
    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']

    def train_evaluate_model(self, training_parameters):
        """
        Train and evaluate the model using the specified parameters.

        Args:
            training_parameters (dict): Dictionary containing training settings and dataset info.

        Returns:
            dict: Metrics results averaged over repetitions.
            model: The trained model.
        """
        
        # Prepare paths for saving the model
        if training_parameters['model_file_name']:
            path_to_model = '/'.join(training_parameters['model_file_name'].split('/')[:-1]) + '/'
            model_name_file = training_parameters['model_file_name'].split('/')[-1]

        # Set number of labels based on the training dataset
        self.num_labels = len(training_parameters['dataset_train'][self.label_column].value_counts())

        # Prepare the training dataset
        dataloader_train = self.prepare_dataset(training_parameters['dataset_train'], shuffle=True)
        
        epochs = training_parameters['epochs']
        metrics_results = {}

        # Repeat training for a specified number of times
        for x in range(training_parameters['repetitions']):
            self.create_llm()

            # Prepare optimizer with weight decay
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": training_parameters['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=training_parameters['learning_rate'],
                betas=(training_parameters['betas'][0], training_parameters['betas'][1])
            )

            # Learning rate scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=len(dataloader_train) * epochs
            )

            train_loss_per_epoch = []
            test_loss_per_epoch = []

            best_val_loss = float('inf')
            early_stopper = EarlyStopper(patience=training_parameters['patience'], min_delta=training_parameters['min_delta'])

            # Training loop
            for epoch_num in range(epochs):
                epoch_num += 1
                print('Epoch:', epoch_num)

                self.model.train()
                loss_train_total = 0
                
                for step_num, batch_data in enumerate(tqdm(dataloader_train, desc='Training')):
                    batch = tuple(b.to(self.device) for b in batch_data)

                    inputs = {
                        'input_ids': batch[0], 
                        'attention_mask': batch[1], 
                        'labels': batch[2]
                    }

                    outputs = self.model(**inputs)
                    loss = training_parameters['loss_function'](outputs.logits, inputs['labels'].long())
                    
                    loss_train_total += loss.item()

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                train_loss_per_epoch.append(loss_train_total / (step_num + 1))

                # Evaluate the model
                loss_val_avg, metrics, _ = self.evaluate_model(test_dataset=training_parameters['dataset_test'])
                test_loss_per_epoch.append(loss_val_avg)

                print(f'Test performance after epoch {epoch_num}:', metrics)
                # print(f'Current los:{loss_val_avg}, best loss: {best_val_loss}')

                # Save the model if it has the best validation loss
                if (loss_val_avg < best_val_loss) and training_parameters['model_file_name']:
                    best_val_loss = loss_val_avg
                    self.save_model(path=path_to_model, name_file=model_name_file)

                    with open(path_to_model + model_name_file + '/epoch_number.txt', "w") as fp:
                        fp.write(str(epoch_num))

                if early_stopper.early_stop(loss_val_avg):             
                    break

            # Collect metrics for averaging
            for m in metrics:
                if m in metrics_results:
                    metrics_results[m].append(metrics[m])
                else:
                    metrics_results[m] = [metrics[m]]

        # Plot training and test loss
        epochs_range = range(1, epochs + 1)
        fig, ax = plt.subplots()
        ax.plot(epochs_range, train_loss_per_epoch, label='Training loss')
        ax.plot(epochs_range, test_loss_per_epoch, label='Test loss')
        ax.set_title('Training and Test loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

        # Print average metrics
        for metric in metrics_results:
            print(metric, np.mean(metrics_results[metric]))

        return metrics_results, self.model

    

    def evaluate_model(self, test_dataset):
        """
        Evaluate the model on the test dataset.

        Args:
            test_dataset (DataFrame): The dataset used for evaluation.

        Returns:
            tuple: Average loss, evaluation metrics, and a DataFrame containing texts, true labels, and predictions.
        """
        
        # Initialize a DataFrame to store classifications
        classifications_df = pd.DataFrame()

        # Get predictions, true values, and total loss from model prediction
        texts, true_vals, predictions, loss_val_total = self.model_prediction(
            data=test_dataset, 
            convert_output_to_probability=True
        )

        # Calculate average loss
        loss_val_avg = loss_val_total / len(test_dataset)

        # Process predictions and true values
        predictions = np.concatenate(predictions, axis=0).argmax(-1)
        true_vals = np.concatenate(true_vals, axis=0)

        # Populate the classifications DataFrame
        classifications_df['texts'] = texts
        classifications_df['labels'] = true_vals  # True labels
        classifications_df['predictions'] = predictions  # Model predictions

        # Compute evaluation metrics
        metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=true_vals))

        return loss_val_avg, metrics, classifications_df

    
    def model_prediction(self, data, convert_output_to_probability):
        """
        Run predictions on the given dataset using the model.

        Args:
            data (DataFrame): The dataset to evaluate.
            convert_output_to_probability (bool): Whether to convert logits to probabilities.

        Returns:
            tuple: Texts, true labels, predictions, and total loss.
        """
        
        # Prepare the test dataset
        dataloader_test = self.prepare_dataset(data, shuffle=False)

        self.model.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        texts = []

        j = 0  # Index for accessing rows in the DataFrame

        for batch in dataloader_test:
            batch = tuple(b.to(self.device) for b in batch)       
            
            for i in range(len(batch[0])):
                # Append the original text to the texts list
                texts.append(data.iloc[j][self.text_column])                
                j += 1

            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2],
            }

            with torch.no_grad():        
                outputs = self.model(**inputs)

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
