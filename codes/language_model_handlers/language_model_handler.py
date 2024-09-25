import os
import torch
from abc import abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification #, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd 
from torch.utils.data import TensorDataset, DataLoader # RandomSampler #, SequentialSampler
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import colorcet as cc
import numpy as np


plt.rcParams['figure.dpi'] = 300

class LanguageModelHandler():
    def __init__(self, model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=128, random_state=42):
        self.handler_type = None
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.pipeline = None
        self.zero_shot_pipeline = None
        self.output_hidden_states = output_hidden_states
        self.random_state = random_state
        
        self.processed_text_column = processed_text_column
        self.text_column = text_column
        self.label_column = label_column

        self.num_labels = len(new_labels)
        self.new_labels = new_labels
        
        self.max_length = text_size_limit
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.create_tokenizer()
        

    def test_gpu(self):
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        # Storing ID of current CUDA device
        cuda_id = torch.cuda.current_device()
        print(f"ID of current CUDA device: {torch.cuda.current_device()}")
        print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

    def create_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer

    def create_llm(self):
        """
        Creates and initializes a model for sequence classification.

        This method loads a pretrained model from the specified model name, sets the number of output classes,
        and moves the model to the specified device (CPU or GPU).

        Returns:
            model: The initialized sequence classification model.
        """
        self.create_tokenizer()

        print('Creating model:', self.model_name)  # Log the model name being created
        print('Device:', self.device)  # Log the device being used (CPU/GPU)
        print('Number of output classes:', self.num_labels)  # Log the number of output classes

        dropout_value = 0.2  # Set dropout value (currently unused)

        try:
            # Attempt to load the pretrained model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=self.output_hidden_states,
                id2label=self.new_labels
                # hidden_dropout_prob=dropout_value  # Hidden dropout probability (commented out)
            )
        except:
            # Handle potential errors and attempt to load the model with mismatched sizes ignored
            print('Error to import the model, ignore mismatched sizes')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                output_attentions=False,
                output_hidden_states=self.output_hidden_states,
                ignore_mismatched_sizes=True,
                id2label=self.new_labels
                # hidden_dropout_prob=dropout_value  # Hidden dropout probability (commented out)
            )

        # Move the model to the specified device (CPU or GPU)
        self.model.to(self.device)

        return self.model  # Return the initialized model


    def add_new_tokens_to_tokenizer(self, new_tokens):
        """
        Adds new tokens to the tokenizer and resizes the model's embeddings layer accordingly.

        Args:
            new_tokens (list): A list of tokens to be added to the tokenizer.

        Returns:
            int: The number of tokens added to the tokenizer, or None if the tokenizer is not initialized.
        """
        
        # Check if the tokenizer is initialized
        if self.tokenizer is not None:
            # Add new tokens to the tokenizer
            number_of_tokens_added = self.tokenizer.add_tokens(new_tokens=new_tokens)

            # Check if the model is initialized
            if self.model is not None:
                print('### Resizing the model embeddings layer...')  # Log resizing action
                # Resize the model's embedding layer to accommodate the new tokens
                self.model.resize_token_embeddings(len(self.tokenizer))

            return number_of_tokens_added  # Return the number of tokens added


    def save_model(self, path, name_file):
        name_file = name_file.replace('/','_')
        
        ## Save tokenizer
        self.tokenizer.save_pretrained(path+name_file)

        ## Save model
        # model_path = os.path.join(path, name_file, name_file+'.pth')
        # torch.save(self.model, model_path)
        self.model.save_pretrained(path+name_file)
        

    def load_llm_model(self, path, name_file):
        """
        Loads a pre-trained language model and its tokenizer from the specified path.

        Args:
            path (str): The directory path where the model and tokenizer are saved.
            name_file (str): The name of the model file. This is appended to the path to locate the model files.

        Returns:
            tuple: A tuple containing the loaded tokenizer and model.
        """

        # Construct the full path to the model directory
        model_name = path + name_file

        # Load the tokenizer from the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model from the pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Move the model to the specified device (e.g., GPU or CPU)
        self.model.to(self.device)

        return self.tokenizer, self.model


    def compute_metrics(self, eval_pred):
        """
        Computes evaluation metrics for model predictions.

        This method calculates accuracy, precision, recall, and F1 score based on the provided predictions and labels.

        Args:
            eval_pred: An object containing the model predictions and the true labels.

        Returns:
            dict: A dictionary containing the computed metrics (accuracy, precision, recall, and F1 score).
        """
        
        # Extract predictions and true labels from the evaluation prediction object
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        
        average_mode = 'weighted'  # Set the averaging mode for precision, recall, and F1 score

        return {
            'accuracy': accuracy_score(labels, predictions),  # Calculate accuracy
            'precision': precision_score(labels, predictions, average=average_mode),  # Calculate precision
            'recall': recall_score(labels, predictions, average=average_mode),  # Calculate recall
            'f1': f1_score(labels, predictions, average=average_mode)  # Calculate F1 score
        }

    def one_hot_encode(self, input_list):
        """
        Perform one-hot encoding on a list of integers.
        
        Args:
        input_list (list of int): List of integers to be one-hot encoded.
        
        Returns:
        np.ndarray: A 2D numpy array where each row is the one-hot encoded version of the corresponding integer in input_list.
        
        Example:
        >>> one_hot_encode([2, 3, 0, 1])
        array([[0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0]])
        """
        
        # Find the number of unique classes by getting the maximum value in the input list
        # and adding 1 (since class labels are assumed to start from 0)
        num_classes = max(input_list) + 1
        
        # Initialize a 2D numpy array of zeros with shape (number of elements in input_list, num_classes)
        one_hot_encoded = np.zeros((len(input_list), num_classes), dtype=int)
        
        # Iterate over the input list and set the corresponding index in the one-hot encoded array to 1
        for idx, val in enumerate(input_list):
            one_hot_encoded[idx, val] = 1
            
        return one_hot_encoded
    
    ### EMBEDDINGS METHODS
    def sentences_to_embedding_standard(self, sentences, model_names=None):
        """
        Computes embeddings for a list of sentences using specified models.

        Args:
            sentences (list): A list of sentences to encode.
            model_names (list, optional): A list of model names to use for encoding. 
                                        If None, defaults to using self.model_name.

        Returns:
            dict: A dictionary where keys are model names and values are the corresponding embeddings.
        """
        
        embeddings_results = {}  # Initialize a dictionary to store embeddings

        # Use the default model name if none are provided
        if model_names is None:
            model_names = [self.model_name]

        for model_name in model_names:
            # Load the SentenceTransformer model
            model = SentenceTransformer(model_name_or_path=model_name, device='cpu')  # Change device to 'gpu' if needed

            # Generate embeddings for the sentences
            embeddings = model.encode(sentences)
            embeddings_results[model_name] = embeddings  # Store embeddings in the results dictionary

        return embeddings_results  # Return the embeddings results


    def tokenize_dataset(self, data):
        """
        Tokenizes a list of texts into input IDs and attention masks.

        Args:
            data (list): A list of texts to tokenize.

        Returns:
            tuple: A tuple containing:
                - input_ids (torch.Tensor): The tokenized input IDs.
                - attention_mask (torch.Tensor): The attention masks for the tokenized inputs.
        """
        
        # Use the tokenizer to encode the data
        encoded_dict = self.tokenizer.batch_encode_plus(
            data,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask']  # Return the input IDs and attention masks


    def prepare_dataset(self, data, shuffle=True):
        """
        Prepares the dataset for training or evaluation by creating input IDs, attention masks, 
        and labels, then loading them into a DataLoader.

        Args:
            data (pd.DataFrame): The input DataFrame containing the text and labels.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

        Returns:
            DataLoader: A DataLoader object containing the prepared dataset.
        """
        
        # Tokenize the dataset to get input IDs and attention masks
        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list())

        # Convert labels to a LongTensor
        if data[self.label_column].dtype == object: # if the labels are in the string format, convert to int
            data[self.label_column] = pd.factorize(data[self.label_column])[0]

        Y = torch.LongTensor(data[self.label_column].to_list())
        
        # Move tensors to the specified device (GPU or CPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        # Create a TensorDataset from the input IDs, attention masks, and labels
        dataset = TensorDataset(input_ids, att_masks, Y)

        # Create a DataLoader for batching the dataset
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return data_loader  # Return the DataLoader


    def reduce_embeddings_dimentionality(self, X, algorithm='PCA'):
        """
        Reduces the dimensionality of the given embeddings using the specified algorithm.

        Args:
            X (np.ndarray): The input embeddings to be reduced. Can be of shape (n_samples, n_features) 
                            or (n_samples, height, width) for images.
            algorithm (str): The dimensionality reduction algorithm to use. Options are 'PCA', 'TSNE', 
                            and 'MDS'. Defaults to 'PCA'.

        Returns:
            np.ndarray: The transformed embeddings with reduced dimensions.
        """
        n_components = 2  # Number of components for dimensionality reduction

        # Reshape the input if it has more than 2 dimensions
        if len(X.shape) > 2:
            nsamples, nx, ny = X.shape
            X = X.reshape((nsamples, nx * ny))  # Flatten the dimensions

        # Initialize the dimensionality reduction object based on the chosen algorithm
        if algorithm == 'PCA':
            dim_reduction_obj = PCA(n_components=n_components)

        elif algorithm == 'TSNE':
            dim_reduction_obj = TSNE(n_components=n_components, verbose=0, perplexity=40, n_iter=300)

        elif algorithm == 'MDS':
            dim_reduction_obj = MDS(n_components=n_components, metric=True, random_state=42)

        # Fit and transform the data using the selected algorithm
        return dim_reduction_obj.fit_transform(X)  # Return the reduced embeddings

    
    
    def get_bert_encoded_data_in_batches(self, df):
        """
        Generates batches of BERT-encoded data from a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing text and labels.

        Yields:
            tuple: A tuple containing:
                - (torch.Tensor, torch.Tensor): A tuple of input IDs and attention masks for the batch.
                - torch.LongTensor: The corresponding labels for the batch.
        """
        
        # Create a list of tuples containing text and labels from the DataFrame
        data = [(row[self.text_column], row[self.label_column]) for _, row in df.iterrows()]
        
        # Create a sequential sampler for the data
        sampler = torch.utils.data.sampler.SequentialSampler(data)
        
        # Create a batch sampler to yield batches of specified size
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=self.batch_size if self.batch_size > 0 else len(data),
            drop_last=False
        )
        
        # Iterate over batches from the batch sampler
        for batch in batch_sampler:
            # Encode the batch of texts using the tokenizer
            encoded_batch_data = self.tokenizer.batch_encode_plus(
                [data[i][0] for i in batch],  # Extract texts for the current batch
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt'
            )

            # Clone the input IDs and attention masks to detach from the computation graph
            seq = encoded_batch_data['input_ids'].clone().detach()
            mask = encoded_batch_data['attention_mask'].clone().detach()

            # Yield the encoded batch data and corresponding labels
            yield (seq, mask), torch.LongTensor([data[i][1] for i in batch])  # Labels for the current batch


    def calculate_embeddings_model_layers(self, data, only_last_layer):
        """
        Calculates the embeddings from specified layers of the model for the given data.

        Args:
            data (pd.DataFrame): The input data containing texts and labels.
            only_last_layer (bool): If True, only the last layer's embeddings will be returned.

        Returns:
            dict: A dictionary containing:
                - 'hidden_states': Tensor of hidden states from the model layers.
                - 'masks': Tensor of attention masks used during encoding.
                - 'ys': Tensor of labels associated with the input data.
        """
        
        # Initialize tensors to hold masks and labels
        test_masks, test_ys = torch.zeros(0, self.max_length), torch.zeros(0, 1)
        test_hidden_states = None  # Placeholder for storing hidden states

        # Iterate through the data in batches
        for x, y in self.get_bert_encoded_data_in_batches(data):
            sent_ids, masks = x  # Extract input IDs and attention masks
            sent_ids = sent_ids.to(self.device)  # Move input IDs to the specified device
            masks = masks.to(self.device)  # Move attention masks to the specified device
            
            with torch.no_grad():  # Disable gradient calculation for inference
                model_out = self.model(sent_ids, masks)  # Get model output
                
                hidden_states = model_out.hidden_states[1:]  # Extract hidden states (skip the first layer)

                # If only the last layer is requested, filter to just the last hidden state
                if only_last_layer:
                    hidden_states = tuple([hidden_states[-1]])  # Keep only the last layer's hidden state
                
                # Concatenate masks and labels to the respective tensors
                test_masks = torch.cat([test_masks, masks.cpu()])  # Move masks to CPU and concatenate
                test_ys = torch.cat([test_ys, y.cpu().view(-1, 1)])  # Move labels to CPU and reshape

                # If hidden states haven't been initialized, initialize them
                if test_hidden_states is None:
                    test_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
                else:
                    # Concatenate hidden states across batches
                    test_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu()]) 
                                                for layer_hidden_state_all, layer_hidden_state_batch in zip(test_hidden_states, hidden_states))
        
        return {'hidden_states': test_hidden_states, 'masks': test_masks, 'ys': test_ys}  # Return the results

    
    def calculate_average_embeddings(self, hidden_states, masks, layers_ids):
        """
        Calculates the average embeddings for specified layers from the hidden states.

        Args:
            hidden_states (tuple): A tuple containing hidden states from the model layers.
            masks (Tensor): A tensor containing attention masks for the input data.
            layers_ids (list): A list of layer indices for which to compute average embeddings.

        Returns:
            dict: A dictionary where keys are layer indices and values are the averaged embeddings for those layers.
        """
        
        # Initialize a dictionary to hold averaged hidden states for specified layers
        all_averaged_layer_hidden_states = {}
        
        # Iterate through the hidden states for each layer
        for layer_i in range(len(hidden_states)):
            if layer_i in layers_ids:  # Check if the current layer is in the specified layer IDs
                # Calculate the average embeddings for the layer
                all_averaged_layer_hidden_states[layer_i] = torch.div(
                    hidden_states[layer_i].sum(dim=1),  # Sum hidden states across the sequence
                    masks.sum(dim=1, keepdim=True)      # Divide by the sum of the masks to get the average
                )
        
        return all_averaged_layer_hidden_states  # Return the dictionary of averaged embeddings

    def plot_embeddings_layers(self, data, results_path, filename, sample_size=None, algorithm='TSNE', labels_to_replace=None, number_of_layers_to_plot=2):
        """
        Plots embeddings for specified layers of the model.

        Args:
            data (DataFrame): The input data containing text and labels.
            results_path (str): Path to save the results.
            filename (str): Name of the file to save the plot.
            sample_size (int, optional): Number of samples to take from each label. Defaults to None.
            algorithm (str): Dimensionality reduction algorithm to use ('TSNE', etc.). Defaults to 'TSNE'.
            labels_to_replace (dict, optional): Mapping of labels to replace for visualization. Defaults to None.
            number_of_layers_to_plot (int): Number of layers to plot embeddings from. Defaults to 2.
        """
        
        # If a sample size is specified, sample from the data
        if sample_size:
            reduced_data = pd.DataFrame()

            for label_value in set(data[self.label_column].tolist()):
                temp_df = data[data[self.label_column] == label_value]
                
                # Adjust sample size if it exceeds available data
                if sample_size > temp_df.shape[0]:
                    sample_size = temp_df.shape[0]
                
                # Concatenate sampled data
                reduced_data = pd.concat([reduced_data, temp_df.sample(n=sample_size, random_state=self.random_state)])

            data = reduced_data  # Update data to the sampled version
        
        # Calculate hidden states and masks for the given data
        embeddings = self.calculate_embeddings_model_layers(data, only_last_layer=False)
        
        test_hidden_states = embeddings['hidden_states']  # Extract hidden states
        test_masks = embeddings['masks']  # Extract attention masks
        test_ys = embeddings['ys']  # Extract labels

        layers_to_visualize = []  # List to hold layers to visualize

        # Determine which layers to visualize: first N and last N layers
        for x in range(number_of_layers_to_plot):
            layers_to_visualize.append(x)  # Add first N layers
            layers_to_visualize.append(len(test_hidden_states) - x - 1)  # Add last N layers
        
        layers_to_visualize.sort()  # Sort the layer indices

        # Calculate average embeddings for the selected layers
        all_averaged_layer_hidden_states = self.calculate_average_embeddings(test_hidden_states, test_masks, layers_ids=layers_to_visualize)

        # If labels need to be replaced, map the new labels
        if labels_to_replace:
            test_ys = data[self.label_column].map(labels_to_replace).to_list()

        # Visualize the embeddings for the specified layers
        self.visualize_layerwise_embeddings(all_averaged_layer_hidden_states=all_averaged_layer_hidden_states, ys=test_ys, results_path=results_path, filename=filename, algorithm=algorithm)

            

    # Based on: https://medium.com/towards-data-science/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568
    def visualize_layerwise_embeddings(self, all_averaged_layer_hidden_states, ys, results_path, filename, algorithm):
        """
        Visualizes layer-wise embeddings using dimensionality reduction.

        Args:
            all_averaged_layer_hidden_states (dict): Averaged hidden states for each layer.
            ys (array-like): Labels corresponding to the embeddings.
            results_path (str): Path to save the visualization results.
            filename (str): Name of the file to save the plot.
            algorithm (str): Dimensionality reduction algorithm to use ('PCA', 'TSNE', etc.).
        """
        
        num_layers = len(all_averaged_layer_hidden_states)  # Number of layers to visualize
        plot_size_factor = 5  # Factor to scale the plot size

        # Create a figure with subplots for each layer
        fig = plt.figure(figsize=((plot_size_factor * num_layers * 2), plot_size_factor + 2))
        ax = [fig.add_subplot(int(num_layers / 2), num_layers, i + 1) for i in range(num_layers)]

        # Ensure ys is a list and reshape if necessary
        if type(ys) != list:
            ys = ys.numpy().reshape(-1)

        # Format the labels for better readability
        ys = [y.capitalize().replace('_', ' ') for y in ys]

        # Loop through each layer's averaged hidden states
        for i, layer_i in enumerate(all_averaged_layer_hidden_states):
            # Reduce the dimensionality of the embeddings
            layer_dim_reduced_vectors = self.reduce_embeddings_dimentionality(all_averaged_layer_hidden_states[layer_i].numpy(), algorithm=algorithm)
            
            # Create a DataFrame for the reduced vectors and their corresponding labels
            df = pd.DataFrame.from_dict({'x': layer_dim_reduced_vectors[:, 0], 'y': layer_dim_reduced_vectors[:, 1], 'label': ys})

            # Plot the embeddings using seaborn
            sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[i], palette=sns.color_palette(cc.glasbey, n_colors=self.num_labels))
            ax[i].set_title(f"Layer {layer_i + 1}")  # Title for each subplot

            # Remove axis values for cleaner visualization
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].legend([], [], frameon=False)  # Hide the legend for individual plots

        print('Visualizing layer-wise embeddings for', results_path + filename)

        # Add a global legend for the entire figure
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0, bbox_transform=fig.transFigure, ncol=3)

        # Save the figure to the specified path
        plt.savefig(results_path + filename, format='png', pad_inches=0)



    #### ABSTRACT METHODS

    # Function to train and evaluate the model
    @abstractmethod
    def train_evaluate_model(self, training_parameters):
        pass


    # def calculate_embeddings_local_model_with_batches(self, data):
        
    #     data_loader = self.prepare_dataset(data=data)
        
    #     X = np.array([])
        
    #     output_class = 'logits' # 'logits'  hidden_states
        
    #     for batch_data in tqdm(data_loader, desc='Data'):

    #             input_ids, att_mask, _ = [data for data in batch_data] # data.to(self.device)

    #             with torch.no_grad():
    #                 model_output = self.model(input_ids=input_ids, attention_mask=att_mask)

    #                 # Removing the first hidden state
    #                 # The first state is the input state
    #                 # model_output[output_class][1:][-1]

    #                 embeddings = model_output[output_class]

    #                 if X.shape[0] == 0:
    #                     X = embeddings.cpu()
    #                 else:
    #                     X = np.concatenate((X, embeddings.cpu()), axis=0)

    #     return {self.model_name: X}
    

