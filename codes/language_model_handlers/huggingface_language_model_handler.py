import numpy as np 

from tqdm import tqdm
from datasets import Dataset
import evaluate
from copy import deepcopy
from transformers import Trainer, EarlyStoppingCallback
from transformers import pipeline #, BertModel, BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from codes.language_model_handlers.language_model_handler import LanguageModelHandler


class HuggingfaceLanguageModelHandler(LanguageModelHandler):
    
    def __init__(self, model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        super().__init__(model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states, batch_size, text_size_limit)
        self.handler_type == 'hugging_face'
        self.metric_accuracy = evaluate.load("accuracy")
        self.metric_precision = evaluate.load("precision")
        self.metric_recall = evaluate.load("recall")
        self.metric_f1 = evaluate.load("f1")

    def tokenize_dataset(self, data):
        return self.tokenizer(data[self.text_column],
                              truncation=True,
                              # padding=True,
                              # return_tensors='pt',
                              max_length= 42, # 32
                              padding="max_length"
                             )

    def add_new_tokens_to_tokenizer(self, new_tokens):
        """
        Adds new tokens to the tokenizer and resizes the model's embedding layer if necessary.

        Args:
            new_tokens (list): A list of new tokens to add to the tokenizer.

        Returns:
            int: The number of tokens added to the tokenizer.
        """
        if self.tokenizer is not None:
            # Add new tokens to the tokenizer
            number_of_tokens_added = self.tokenizer.add_tokens(new_tokens=new_tokens)

            if self.model is not None:
                print('### Resizing the model embeddings layer...')
                # Resize the model's embedding layer to accommodate new tokens
                self.model.resize_token_embeddings(len(self.tokenizer))

            return number_of_tokens_added

    def prepare_dataset(self, data):
        """
        Prepares the dataset for training by tokenizing and converting it to the required format.

        Args:
            data (pd.DataFrame): The input dataframe containing the data to prepare.

        Returns:
            Dataset: A tokenized dataset ready for use with the model.
        """
        
        # Determine the number of unique labels in the dataset
        self.num_labels = len(data[self.label_column].value_counts())
        
        # Convert the pandas dataframe to a Hugging Face Dataset
        hg_data = Dataset.from_pandas(data)        

        # Tokenize the dataset using the predefined tokenization method
        tokenized_dataset = hg_data.map(self.tokenize_dataset)
        
        return tokenized_dataset


    def evaluate_model(self, dataloader_val):
        """
        Evaluates the model on the provided validation dataloader.

        Args:
            dataloader_val (DataLoader): The validation DataLoader containing input data.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The predicted logits from the model.
                - np.ndarray: The true label IDs corresponding to the input data.
        """
        
        # Set the model to evaluation mode
        self.model.eval()
        
        predictions, true_vals = [], []  # Initialize lists to store predictions and true values
        
        # Iterate over batches in the validation dataloader
        for batch in dataloader_val:
            # Move batch data to the specified device (CPU/GPU)
            batch = tuple(b.to(self.device) for b in batch)
            
            # Prepare inputs for the model
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[2]
            }

            # Disable gradient calculation for evaluation
            with torch.no_grad():        
                outputs = self.model(**inputs)

            logits = outputs['logits']  # Get the logits from the model's output
            
            # Detach the logits from the computation graph and move to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()  # Get true labels

            # Append logits and true labels to their respective lists
            predictions.append(logits)
            true_vals.append(label_ids)
        
        # Concatenate predictions and true values across all batches
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return predictions, true_vals  # Return the predictions and true labels

    
    def train_evaluate_model(self, training_parameters):
        """
        Trains and evaluates the model based on the provided training parameters.

        Args:
            training_parameters (dict): A dictionary containing the necessary training configurations and datasets.

        Returns:
            tuple: A tuple containing:
                - dict: The computed metrics (accuracy, precision, recall, F1 score).
                - MyTrainer: The trainer instance used for training the model.
        """
        
        results_summary = {}
        detailed_metrics = ['eval_accuracy', 'eval_precision', 'eval_recall', 'eval_f1']

        # Prepare the training and test datasets
        self.tokenized_dataset_train = self.prepare_dataset(training_parameters['dataset_train'])
        self.tokenized_dataset_test = self.prepare_dataset(training_parameters['dataset_test'])

        # Create data loaders and model
        self.create_dl_model()

        # Create a deep copy of the model to avoid modifying the original during training
        model = deepcopy(self.model)

        # Create the AdamW optimizer with weight decay
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        # Group parameters for optimizer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters],
                "weight_decay": training_parameters['training_args'].weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_parameters['training_args'].learning_rate,
            betas=(training_parameters['training_args'].adam_beta1, training_parameters['training_args'].adam_beta2),
            eps=training_parameters['training_args'].adam_epsilon
        )

        # Initialize the trainer
        self.trainer = MyTrainer(
            model=model, 
            args=training_parameters['training_args'], 
            train_dataset=self.tokenized_dataset_train, 
            eval_dataset=self.tokenized_dataset_test, 
            loss_function=training_parameters['loss_function'],
            optimizers=(optimizer, None),
            # compute_metrics=self.compute_metrics, 
            # tokenizer=self.tokenizer
        )

        # Set up early stopping if specified in the training parameters
        if training_parameters['early_stopping_patience']:
            self.trainer.callbacks = [EarlyStoppingCallback(early_stopping_patience=training_parameters['early_stopping_patience'])]

        # Train the model
        self.trainer.train()

        # Update the model with the trained version from the trainer
        self.model = self.trainer.model

        # Evaluate the model on the test dataset
        predictions, true_vals = self.evaluate_model(dataloader_val=self.prepare_dataset_for_embeddings(training_parameters['dataset_test']))

        # Extract predicted classes from logits
        predictions = predictions[0].argmax(-1)  # Extract the predicted class indices

        # Compute evaluation metrics
        metrics = self.compute_metrics((np.argmax(predictions, axis=-1), true_vals))

        return metrics, self.trainer  # Return the computed metrics and trainer instance


    def data_loader(self, dataframe, column):
        """
        Generator function that yields processed text from a specified column of a dataframe.

        Args:
            dataframe (pd.DataFrame): The input dataframe containing text data.
            column (str): The name of the column from which to extract text.

        Yields:
            str or list: The original text or a truncated list of words if the text exceeds the size limit.
        """
        # Iterate over each row in the dataframe
        for row in dataframe.values:
            text = row[column]  # Getting the text from the specified column

            print(text)  # Print the text for debugging purposes
            
            # Check if the text exceeds the specified word limit
            if len(text.split()) > self.text_size_limit:
                # Yield the first 'text_size_limit' words as a list
                yield text.split()[:self.text_size_limit]
            else:
                # Yield the original text if it doesn't exceed the limit
                yield text    
            
            yield text  # Yield the original text again (this seems redundant)


    def plot_embeddings(self, file_name, embeddings_results, labels, algorithm='PCA', all_together=False):
        """
        Plots the embeddings using a specified dimensionality reduction algorithm.

        Args:
            file_name (str): The name of the file to save the plot.
            embeddings_results (dict): A dictionary containing embeddings for different models.
            labels (list): The labels corresponding to the embeddings.
            algorithm (str): The algorithm to use for dimensionality reduction (default is 'PCA').
            all_together (bool): If True, plots all embeddings together; otherwise, plots them separately.
        """
        
        # Initialize an empty DataFrame to store embeddings
        embeddings_df = pd.DataFrame()

        # Iterate over each model's embeddings
        for model_name in embeddings_results:
            X = embeddings_results[model_name]  # Get the embeddings for the current model

            # Reduce dimensionality of the embeddings
            reduced_data = self.reduce_embeddings_dimentionality(X, algorithm)

            # Create a DataFrame for the current model's embeddings
            df = pd.DataFrame()
            df['model'] = [model_name] * X.shape[0]  # Assign model names
            df['model'] = df['model'].apply(lambda i: str(i))  # Ensure model names are strings

            df['label'] = labels  # Assign labels to the DataFrame

            df['first_dimension'] = reduced_data[:, 0]  # First dimension of reduced data
            df['second_dimension'] = reduced_data[:, 1]  # Second dimension of reduced data

            # Concatenate the current model's DataFrame to the overall embeddings DataFrame
            embeddings_df = pd.concat([embeddings_df, df])

        # Plot the embeddings based on the 'all_together' flag
        if all_together:
            self.plot_embbedings_together(embeddings_df, file_name)
        else:
            self.plot_embbedings_separated(embeddings_df, file_name)



            
    def plot_embbedings_together(self, embeddings_df):
        plt.figure(figsize=(16,10))

        # Automatically assign colors and shapes
        unique_models = embeddings_df['model'].unique()
        unique_labels = embeddings_df['label'].unique()

        color_dict = {model: plt.cm.tab10(i) for i, model in enumerate(unique_models)}
        symbols = ['x', 'o'] #['v', '^', 's', 'D', 'o', '<', '>', 'p', '*']
        shape_dict = {label: marker for label, marker in zip(unique_labels, symbols)}

        # Scatter plot
        for model, group_model in embeddings_df.groupby('model'):
            for label, group_label in group_model.groupby('label'):
                plt.scatter(group_label['first_dimension'], group_label['second_dimension'],
                            label=f'{model} - {label}',
                            color=color_dict.get(model, 'black'),
                            marker=shape_dict.get(label, 'o'),
                            alpha=0.3)

        # Customize the plot
        # plt.title('Scatter Plot with Models and Labels')
        plt.xlabel('First Dimension')
        plt.ylabel('Second Dimension')
        plt.legend()
        plt.grid(False)

        # disabling xticks by Setting xticks to an empty list
        plt.xticks([])

        # disabling yticks by setting yticks to an empty list
        plt.yticks([])

        plt.show()


    def plot_embbedings_separated(self, embeddings_df, file_name):
        # Automatically assign colors and shapes
        unique_models = embeddings_df['model'].unique()
        unique_labels = embeddings_df['label'].unique()

        # color_dict = {model: plt.cm.tab10(i) for i, model in enumerate(unique_models)}
        color_dict = {label: sns.color_palette("husl", n_colors=len(unique_labels))[i] for i, label in enumerate(unique_labels)}
        shape_dict = {label: marker for label, marker in zip(unique_labels, ['x', 'o', 's', 'D', 'v', '<', '>', 'p', '*'])}

        if len(unique_models) == 1:
            number_of_plots = len(unique_models)+1

        # Set the size of each subplot
        fig, axes = plt.subplots(1, number_of_plots, figsize=(15, 5))  # Adjust the figsize as needed

        # Create subplots for each model
        for ax, model in zip(axes, unique_models):
            ax.set_title(model)

            for label, group_label in embeddings_df[embeddings_df['model'] == model].groupby('label'):
                ax.scatter(group_label['first_dimension'], group_label['second_dimension'],
                          label=label,
                          color=color_dict.get(label, 'black'),
                          marker=shape_dict.get(label, 'o'))

            ax.set_xlabel('First Dimension')
            ax.set_ylabel('Second Dimension')
            ax.legend()
            ax.grid(False)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # disabling xticks by Setting xticks to an empty list
        plt.xticks([])

        # disabling yticks by setting yticks to an empty list
        plt.yticks([])

        # Show/save the plots
        # plt.show()
        plt.savefig(file_name)

class MyTrainer(Trainer):
    def __init__(self, loss_function=None, **kwds):
        super(MyTrainer, self).__init__(**kwds)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        
        labels = inputs.get('labels')
        
        outputs = model(**inputs)
        logits = outputs.get('logits')

        loss = self.loss_function(logits.squeeze(), labels.squeeze())

        return (loss, outputs) if return_outputs else loss