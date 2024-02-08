import os
import torch
from abc import abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification #, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer


class LanguageModelHandler():
    def __init__(self, model_name, new_labels, text_column, label_column, batch_size=32, text_size_limit=512):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.pipeline = None
        self.zero_shot_pipeline = None
        
        self.text_column = text_column
        self.label_column = label_column

        self.num_labels = len(new_labels)
        self.new_labels = new_labels
        
        self.text_size_limit = text_size_limit
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_tokenizer()
        

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

    def create_dl_model(self):
        print('Creating model:', self.model_name)
        print('Device:', self.device)
        print('Number of output classes:', self.num_labels)

        dropout_value = 0.2

        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                            num_labels=self.num_labels,
                                                                            output_attentions=False,
                                                                            output_hidden_states=True,
                                                                            id2label=self.new_labels,
                                                                            hidden_dropout_prob=dropout_value)
        except:
            print('Error to import the model, ignore mismatched sizes')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                            num_labels=self.num_labels,
                                                                            output_attentions=False,
                                                                            output_hidden_states=True,
                                                                            ignore_mismatched_sizes=True,
                                                                            id2label=self.new_labels,
                                                                            hidden_dropout_prob=dropout_value)
        self.model.to(self.device)

        return self.model

    def add_new_tokens_to_tokenizer(self, new_tokens):
        if self.tokenizer is not None:
            number_of_tokens_added = self.tokenizer.add_tokens(new_tokens=new_tokens)

            if self.model is not None:
                print('### Resizing the model embeddings layer...')
                self.model.resize_token_embeddings(len(self.tokenizer))

            return number_of_tokens_added

    def save_model(self, path, name_file):
        # Save tokenizer
        self.tokenizer.save_pretrained(path+name_file)
        # Save model
        model_path = os.path.join(path, name_file, name_file+'.pth')
        torch.save(self.model, model_path)

        

    def load_model(self, path, name_file):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path+name_file)
        # Load Model
        # self.model = AutoModelForSequenceClassification.from_pretrained(path+name_file)
        self.model = torch.load(path+name_file+'/'+name_file+'.pth')
        self.model.to(self.device)

        return self.tokenizer, self.model

    ### EMBEDDINGS METHODS
    def sentences_to_embedding_standard(self, sentences, model_names=None):

        embeddings_results = {}

        if model_names is None:
            model_names = [self.model_name]

        for model_name in model_names:

            model = SentenceTransformer(model_name_or_path=model_name, device='cpu') #  device='gpu'

            embeddings = model.encode(sentences)
            embeddings_results[model_name] = embeddings

        return embeddings_results
    
        
    #### ABSTRACT METHODS

    # Function to tokenize the dataset before training
    @abstractmethod
    def tokenize_dataset(self, data):
        pass

    # Function to train and evaluate the model
    @abstractmethod
    def train_evaluate_model(self, training_parameters):
        pass

    # Function to compute the evaluation metrics
    @abstractmethod
    def compute_metrics(self, predictions_labels):
        pass

    @abstractmethod
    def prepare_dataset(self, data):
        pass
