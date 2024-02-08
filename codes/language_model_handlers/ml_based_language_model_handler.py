# https://towardsdatascience.com/feature-extraction-with-bert-for-text-classification-533dde44dc2f
# https://ddimri.medium.com/bert-for-classification-beyond-the-next-sentence-prediction-task-93acc1412749
# https://towardsdatascience.com/a-beginners-guide-to-text-classification-with-scikit-learn-632357e16f3a#1629
# Getting embeddings from the final BERT layer https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import pipeline 
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from codes.language_model_handlers.language_model_handler import LanguageModelHandler


class MachineLearningLanguageModelHandler(LanguageModelHandler):
    
    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
        
    def data_loader(self, dataframe, column):
        for row in dataframe.values:
            text = row[column] # Getting the text of the tweet
            
            if len(text.split()) > self.text_size_limit:
                yield text.split()[:self.text_size_limit]
            else:
                yield text    
            yield text 
            
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
    
    def prepare_dataset(self, data):

        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list())     
        # y = torch.LongTensor(data[self.label_column].to_list())
        
        #move on device (GPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        # y = y.to(self.device)
        
        # dataset = TensorDataset(input_ids, att_masks, y)
        dataset = TensorDataset(input_ids, att_masks)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return data_loader
    
    def calculate_embeddings_local_model(self, data):
        
        data_tokenized = self.tokenize_dataset(data)

        with torch.no_grad():
            hidden_states = self.model(**data_tokenized) #dim : [batch_size(nr_sentences), tokens, emb_dim]

        #get only the [CLS] hidden states
        cls = hidden_states.last_hidden_state[:,0,:]

        return cls

    def calculate_embeddings_local_model_with_batches(self, original_data):
        
        data_loader = self.prepare_dataset(data=original_data)
        
        X = np.array([])
        
        output_class = 'hidden_states' # 'logits' 
        
        for batch_data in tqdm(data_loader, desc='Data'):

                input_ids, att_mask = [data for data in batch_data] # data.to(self.device)

                with torch.no_grad():
                    model_output = self.model(input_ids=input_ids, attention_mask=att_mask)

                    # Removing the first hidden state
                    # The first state is the input state
                    token_embeddings = model_output[output_class][1:][-1]

                    if X.shape[0] == 0:
                        X = token_embeddings.cpu()
                    else:
                        X = np.concatenate((X, token_embeddings.cpu()), axis=0)
        return X
    
    def _reshape_dataset_for_ml(self, data):
        nsamples, nx, ny = data.shape
        return data.reshape((nsamples, nx*ny))
    
    def train_evaluate_model(self, training_args, iterations):
        
        X_train = self.calculate_embeddings_local_model_with_batches(training_args['dataset_train'])
        X_test = self.calculate_embeddings_local_model_with_batches(training_args['dataset_test'])

        X_train = self._reshape_dataset_for_ml(X_train)
        X_test = self._reshape_dataset_for_ml(X_test)

        print(X_train.shape, X_test.shape)
        
        y_train = training_args['dataset_train'][self.label_column]
        y_test = training_args['dataset_test'][self.label_column]

        if training_args['ml_model'].lower() == 'random forest':
            ml_model = RandomForestClassifier(n_estimators=100, random_state=training_args['seed'])
            
        elif training_args['ml_model'].lower() == 'svm':
            ml_model = SVC(kernel='linear')

        elif training_args['ml_model'].lower() == 'decision tree':
            ml_model = DecisionTreeClassifier()

        elif training_args['ml_model'].lower() == 'naive bayes':
            ml_model = GaussianNB()

        elif training_args['ml_model'].lower() == 'logistic regression':
            ml_model = LogisticRegression()

        print('Training model...')
        ml_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = ml_model.predict(X_test)
        
        # Calculate accuracy
        performance_metrics = self.compute_metrics((y_pred, y_test))
        print('Testing metrics:', performance_metrics)
    
    # def load_model(self, path, name_file):
    #     # Load tokenizer
    #     self.tokenizer = AutoTokenizer.from_pretrained(path+name_file)
    #     # Load Model
    #     self.model = AutoModelForSequenceClassification.from_pretrained(path+name_file)
    #     self.model.to(self.device)