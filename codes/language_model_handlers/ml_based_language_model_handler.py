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
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import EvalPrediction

from codes.language_model_handlers.language_model_handler import LanguageModelHandler


class MachineLearningLanguageModelHandler(LanguageModelHandler):
    
    def __init__(self, model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        super().__init__(model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states, batch_size, text_size_limit)
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
        fine_tuning_pipeline = pipeline(task="feature-extraction", tokenizer=self.tokenizer, model=self.model, 
                                        device=self.device, max_length=self.text_size_limit, truncation=True, 
                                        padding=True, framework='pt')
        
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512,'return_tensors':'pt'}
        
        all_embeddings = []
        
        for embedding in tqdm(fine_tuning_pipeline(self._data_loader(sentences_df, column_index), batch_size=self.batch_size), 
                              total=sentences_df.shape[0], **tokenizer_kwargs):            
            all_embeddings.append(embedding)
            
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
        
        data_tokenized = self.tokenize_dataset(data)

        with torch.no_grad():
            hidden_states = self.model(**data_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]

        #get only the [CLS] hidden states
        cls = hidden_states.last_hidden_state[:,0,:]

        return cls

        
    def _reshape_dataset_for_ml(self, data):
        nsamples, nx, ny = data.shape
        return data.reshape((nsamples, nx*ny))
    
    def train_evaluate_model(self, training_args, iterations):
        """
        Trains and evaluates a specified machine learning model using embeddings from a transformer model.

        Args:
            training_args (dict): Dictionary containing training parameters, dataset, and ML model configuration.
            iterations (int): Number of iterations for training (if applicable).

        Returns:
            dict: Performance metrics of the trained model.
        """
        
        # Calculate embeddings for training and test datasets
        hidden_states, masks, _ = self.calculate_embeddings_model_layers(training_args['dataset_train'], only_last_layer=True)
        X_train = np.array(list(self.calculate_average_embeddings(hidden_states, masks, layers_ids=[0]).values())[0])

        hidden_states, masks, _ = self.calculate_embeddings_model_layers(training_args['dataset_test'], only_last_layer=True)
        X_test = np.array(list(self.calculate_average_embeddings(hidden_states, masks, layers_ids=[0]).values())[0])
        
        print(f"Train embeddings shape: {X_train.shape}, Test embeddings shape: {X_test.shape}")
        
        y_train = training_args['dataset_train'][self.label_column]
        y_test = training_args['dataset_test'][self.label_column]

        # Initialize the machine learning model
        ml_model_mapping = {
            'random forest': RandomForestClassifier(n_estimators=100, random_state=training_args['seed']),
            'svm': SVC(kernel='linear'),
            'decision tree': DecisionTreeClassifier(),
            'naive bayes': GaussianNB(),
            'logistic regression': LogisticRegression(),
            'gradient boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=training_args['seed'])
        }

        ml_model_name = training_args['ml_model'].lower()
        ml_model = ml_model_mapping.get(ml_model_name)

        if ml_model is None:
            raise ValueError(f"Model '{training_args['ml_model']}' is not recognized.")

        print('Training model...')
        ml_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = ml_model.predict(X_test)

        # Calculate performance metrics
        performance_metrics = self.compute_metrics(EvalPrediction(predictions=y_pred, label_ids=y_test))
        
        return performance_metrics
