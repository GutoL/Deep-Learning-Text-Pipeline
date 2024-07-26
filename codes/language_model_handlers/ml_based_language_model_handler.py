# https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
# https://ddimri.medium.com/bert-for-classification-beyond-the-next-sentence-prediction-task-93acc1412749
# https://towardsdatascience.com/a-beginners-guide-to-text-classification-with-scikit-learn-632357e16f3a#1629
# Getting embeddings from the final BERT layer https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import pipeline 
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import EvalPrediction
import numpy as np
import pickle
from copy import deepcopy

from codes.language_model_handlers.language_model_handler import LanguageModelHandler


class MachineLearningLanguageModelHandler(LanguageModelHandler):
    
    def __init__(self, ml_model_name, llm_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, 
                 batch_size=32, text_size_limit=512, seed=42):
        
        # Initialize the machine learning model
        self.ml_model_mapping = {
            'random forest': RandomForestClassifier(n_estimators=100, random_state=seed),
            'svm': SVC(kernel='linear'),
            'decision tree': DecisionTreeClassifier(),
            'naive bayes': GaussianNB(),
            'logistic regression': LogisticRegression(),
            'gradient boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=seed)
        }

        self.ml_model_name = ml_model_name

        self.ml_model = None

        super().__init__(llm_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states, batch_size, text_size_limit)
        self.handler_type = 'machine_learning'
        

    def data_loader(self, dataframe, column):
        """
        Generator function that yields processed text data from a specified column in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            column (str): The column name containing the text data.

        Yields:
            list or str: A list of words (if exceeds size limit) or the original text.
        """
        for row in dataframe.values:
            text = row[column]  # Getting the text from the specified column

            # Yield a truncated list of words if the text exceeds the size limit
            if len(text.split()) > self.text_size_limit:
                yield text.split()[:self.text_size_limit]
            else:
                yield text  # Yield the original text

            
    def generate_embeddings_pipeline(self, sentences_df, column_index):
        """
        Generates embeddings for sentences in the specified DataFrame column using a pre-trained model.

        Args:
            sentences_df (pd.DataFrame): DataFrame containing the sentences to embed.
            column_index (int): Index of the column in the DataFrame that contains the sentences.

        Returns:
            list: A list of embeddings for each sentence in the DataFrame column.
        """
        
        # Initialize the feature extraction pipeline with the specified tokenizer and model
        fine_tuning_pipeline = pipeline(
            task="feature-extraction",
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            max_length=self.text_size_limit,
            truncation=True,
            padding=True,
            framework='pt'
        )
        
        # Set tokenizer arguments for padding, truncation, and tensor format
        tokenizer_kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': 512,
            'return_tensors': 'pt'
        }
        
        all_embeddings = []  # List to store embeddings for all sentences
        
        # Generate embeddings for each batch of sentences using the fine-tuning pipeline
        for embedding in tqdm(
                fine_tuning_pipeline(self._data_loader(sentences_df, column_index), batch_size=self.batch_size), 
                total=sentences_df.shape[0], **tokenizer_kwargs):
            all_embeddings.append(embedding)  # Append each embedding to the list
        
        return all_embeddings


    def reshape_dataset_for_ml(self, data):
        nsamples, nx, ny = data.shape
        return data.reshape((nsamples, nx*ny))

    def compute_metrics(self, predictions_labels):
        # preds_flat = np.argmax(preds, axis=1).flatten()
        # labels_flat = labels.flatten()
        # return f1_score(labels_flat, preds_flat, average='weighted')
        predictions, labels = predictions_labels

        average_mode = 'weighted'
        
        results = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average_mode),
            'recall': recall_score(labels, predictions, average=average_mode),
            'f1': f1_score(labels, predictions, average=average_mode)
        }
        
        return results
    
    def calculate_embeddings_local_model(self, data):
        """
        Calculates embeddings for the input data using a locally loaded pre-trained model.

        Args:
            data (pd.DataFrame): The input data containing text to be embedded.

        Returns:
            torch.Tensor: A tensor containing the [CLS] embeddings for each sentence in the input data.
        """
        
        # Tokenize the input dataset
        data_tokenized = self.tokenize_dataset(data)

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            # Get the hidden states from the model
            hidden_states = self.model(**data_tokenized)  # Dimension: [batch_size, tokens, emb_dim]

        # Extract the hidden states corresponding to the [CLS] token
        cls = hidden_states.last_hidden_state[:, 0, :]  # Dimension: [batch_size, emb_dim]

        return cls


    def generate_embeddings(self, data, only_last_layer=True):
        """
        Generates embeddings for the input data using the model.

        Args:
            data (pd.DataFrame): The input data containing text to be embedded.
            only_last_layer (bool): If True, use only the last layer's embeddings. Default is True.

        Returns:
            np.ndarray: A numpy array containing the average embeddings.
        """
        
        # Calculate embeddings for the input data
        embeddings = self.calculate_embeddings_model_layers(data, only_last_layer=only_last_layer)
        hidden_states = embeddings['hidden_states']  # Extract hidden states from embeddings
        masks = embeddings['masks']  # Extract attention masks from embeddings

        # Calculate average embeddings from the hidden states and attention masks
        average_embeddings = self.calculate_average_embeddings(hidden_states, masks, layers_ids=[0])

        # Return the average embeddings as a numpy array
        return np.array(list(average_embeddings.values())[0])    
    
    
    def train_ml_model(self, X_train, y_train, convert_to_one_hot_encoding=True):
        """
        Trains a machine learning model using the provided training data.

        Args:
            X_train (np.ndarray or pd.DataFrame): The training input data.
            y_train (np.ndarray or pd.Series): The training labels.
            convert_to_one_hot_encoding (bool): If True, convert the labels to one-hot encoding. Default is True.

        Returns:
            model: The trained machine learning model.
        """
        
        # Convert labels to one-hot encoding if specified
        if convert_to_one_hot_encoding:
            y_train = self.one_hot_encode(y_train)

        # Fit the model with the training data
        self.ml_model.fit(X_train, y_train)

        # Return the trained model
        return self.ml_model


    def evaluate_ml_model(self, X_test, y_test, convert_to_one_hot_encoding=True):
        """
        Evaluates the machine learning model using the provided test data.

        Args:
            X_test (np.ndarray or pd.DataFrame): The test input data.
            y_test (np.ndarray or pd.Series): The test labels.
            convert_to_one_hot_encoding (bool): If True, convert the labels to one-hot encoding. Default is True.

        Returns:
            tuple: A tuple containing performance metrics and the model's predictions.
        """

        # Check if the machine learning model is loaded
        if self.ml_model is None:
            raise ValueError("Machine learning model is not defined, please use the load_ml_model function.")

        # Convert labels to one-hot encoding if specified
        if convert_to_one_hot_encoding:
            y_test = self.one_hot_encode(y_test)

        # Make predictions on the test set
        y_pred = self.ml_model.predict(X_test)

        # Calculate performance metrics
        performance_metrics = self.compute_metrics(EvalPrediction(predictions=y_pred, label_ids=y_test))

        return performance_metrics, y_pred

    def train_evaluate_model(self, training_args):
        """
        Trains and evaluates multiple machine learning models using the provided training arguments.

        Args:
            training_args (dict): A dictionary containing the following keys:
                - 'dataset_train': Training dataset (pd.DataFrame).
                - 'dataset_test': Test dataset (pd.DataFrame).
                - 'ml_models_list': List of machine learning model names to be trained and evaluated (list of str).
                - 'convert_to_one_hot_encoding': Boolean indicating whether to convert labels to one-hot encoding.

        Returns:
            tuple: A tuple containing:
                - performance_metrics (dict): Performance metrics for each model.
                - predictions (dict): Predictions made by each model.
                - models (dict): Trained machine learning models.
        """

        performance_metrics = {}
        predictions = {}
        models = {}

        # Generate embeddings for the training and test datasets
        X_train = self.generate_embeddings(training_args['dataset_train'])
        y_train = training_args['dataset_train'][self.label_column]

        X_test = self.generate_embeddings(training_args['dataset_test'])
        y_test = training_args['dataset_test'][self.label_column]

        # Iterate over each machine learning model specified in the list
        for ml_model_name in training_args['ml_models_list']:
            print('Training model', ml_model_name, '...')

            # Check if the model name is recognized
            if ml_model_name not in self.ml_model_mapping:
                raise ValueError(f"Model '{ml_model_name}' is not recognized.")
            
            # Retrieve the machine learning model from the mapping
            self.ml_model = self.ml_model_mapping.get(ml_model_name)

            # Train the machine learning model
            ml_model = self.train_ml_model(
                X_train=X_train, 
                y_train=y_train, 
                convert_to_one_hot_encoding=training_args['convert_to_one_hot_encoding']
            )

            # Evaluate the trained model on the test set
            metrics, pred = self.evaluate_ml_model(
                X_test=X_test, 
                y_test=y_test, 
                convert_to_one_hot_encoding=training_args['convert_to_one_hot_encoding']
            )
            
            # Store the trained model, its performance metrics, and predictions
            models[ml_model_name] = deepcopy(ml_model)
            performance_metrics[ml_model_name] = metrics
            predictions[ml_model_name] = pred

        return performance_metrics, predictions, models

    def save_ml_model(self, model, path, ml_model_name):
        """
        Saves a machine learning model to a specified file path.

        Args:
            model (sklearn.base.BaseEstimator): The machine learning model to be saved. If None, the current `self.ml_model` will be saved.
            path (str): The directory path where the model should be saved.
            ml_model_name (str): The name of the machine learning model, used to create the filename.

        Returns:
            None
        """
        
        # If no model is provided, use the current ml_model
        if model is None:
            model = self.ml_model

        # Format the file path
        path = path + ml_model_name.replace(' ', '_')

        # Save the model using pickle
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(model, f)

    
    def load_ml_model(self, path, ml_model_name):
        """
        Loads a machine learning model from a specified path using pickle.

        Args:
            path (str): The directory path where the model is saved.
            ml_model_name (str): The name of the machine learning model. This name is used to locate the file to be loaded.

        Returns:
            object: The loaded machine learning model.
        """

        # Load the language model, if needed
        self.load_llm_model(path=path, name_file=self.model_name)

        # Generate the full path for the model file
        path = path + ml_model_name.replace(' ', '_')

        # Load the model from the pickle file
        with open(path + '.pkl', 'rb') as f:
            self.ml_model = pickle.load(f)

        return self.ml_model
