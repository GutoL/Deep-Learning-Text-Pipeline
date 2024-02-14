import os
import torch
from abc import abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification #, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler #, SequentialSampler
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class LanguageModelHandler():
    def __init__(self, model_name, new_labels, text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        self.handler_type = None
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.pipeline = None
        self.zero_shot_pipeline = None
        self.output_hidden_states = output_hidden_states
        
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
                                                                            output_hidden_states=self.output_hidden_states,
                                                                            id2label=self.new_labels
                                                                            # hidden_dropout_prob=dropout_value
                                                                            )
        except:
            print('Error to import the model, ignore mismatched sizes')
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                            num_labels=self.num_labels,
                                                                            output_attentions=False,
                                                                            output_hidden_states=self.output_hidden_states,
                                                                            ignore_mismatched_sizes=True,
                                                                            id2label=self.new_labels
                                                                            # hidden_dropout_prob=dropout_value
                                                                            )
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
        name_file = name_file.replace('/','-')
        
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

    def compute_metrics(self, eval_pred):
        
        if self.handler_type == 'hugging_face':
            preds = eval_pred.predictions[0].argmax(-1) # you have to extract the logist from the outpout model if it is a tuple
        
        elif self.handler_type == 'pytorch':
            preds = eval_pred.predictions.argmax(-1)
        
        elif self.handler_type == 'machine_learning':
            preds = eval_pred.predictions

        labels = eval_pred.label_ids
        
        average_mode = 'weighted'

        # print('===================================')
        # print(preds)
        # print(labels)
        # print('===================================')
        
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average=average_mode),
            'recall': recall_score(labels, preds, average=average_mode),
            'f1': f1_score(labels, preds, average=average_mode)
        }
    
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
    
    def calculate_embeddings_local_model_with_batches(self, data):
        
        data_loader = self.prepare_dataset(data=data)
        
        X = np.array([])
        
        output_class = 'logits' # 'logits'  hidden_states
        
        for batch_data in tqdm(data_loader, desc='Data'):

                input_ids, att_mask, _ = [data for data in batch_data] # data.to(self.device)

                with torch.no_grad():
                    model_output = self.model(input_ids=input_ids, attention_mask=att_mask)

                    # Removing the first hidden state
                    # The first state is the input state
                    # model_output[output_class][1:][-1]

                    embeddings = model_output[output_class]

                    if X.shape[0] == 0:
                        X = embeddings.cpu()
                    else:
                        X = np.concatenate((X, embeddings.cpu()), axis=0)

        return {self.model_name: X}
    
    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    
    def prepare_dataset(self, data):

        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list())     
        Y = torch.LongTensor(data[self.label_column].to_list())
        
        #move on device (GPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        dataset = TensorDataset(input_ids, att_masks, Y)
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return data_loader

    def reduce_embeddings_dimentionality(self, X, algorithm='PCA'):
        n_components = 2

        if len(X.shape) > 2:
            nsamples, nx, ny = X.shape
            X = X.reshape((nsamples, nx*ny))
     
        if algorithm == 'PCA':
            dim_reduction_obj = PCA(n_components=n_components)

        elif algorithm == 'TSNE':
            dim_reduction_obj = TSNE(n_components=n_components, verbose=0, perplexity=40, n_iter=300)

        elif algorithm == 'MDS':
            dim_reduction_obj = MDS(n_components=n_components, metric=True, random_state=42)
        
        return dim_reduction_obj.fit_transform(X)
    
    
    def get_bert_encoded_data_in_batches(self, df):
        data = [(row.text, row.label,) for _, row in df.iterrows()]
        sampler = torch.utils.data.sampler.SequentialSampler(data)
        batch_sampler = torch.utils.data.BatchSampler(sampler,batch_size=self.batch_size if self.batch_size > 0 else len(data), drop_last=False)
        
        for batch in batch_sampler:
            encoded_batch_data = self.tokenizer.batch_encode_plus([data[i][0] for i in batch], add_special_tokens=True, max_length=128, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')

            # seq = torch.tensor(encoded_batch_data['input_ids'])
            # mask = torch.tensor(encoded_batch_data['attention_mask'])
            seq = encoded_batch_data['input_ids'].clone().detach()
            mask = encoded_batch_data['attention_mask'].clone().detach()

            yield (seq, mask), torch.LongTensor([data[i][1] for i in batch])

    def plot_embeddings_all_layters(self, data, title):
        max_length = 128
        val_masks,val_ys = torch.zeros(0, max_length), torch.zeros(0, 1)

        val_hidden_states = None

        for x, y in self.get_bert_encoded_data_in_batches(data):
            sent_ids, masks = x
            sent_ids = sent_ids.to(self.device)
            masks = masks.to(self.device)
            
            with torch.no_grad():
                model_out = self.model(sent_ids, masks)
            
                hidden_states = model_out.hidden_states[1:]
            
                val_masks = torch.cat([val_masks,masks.cpu()])
                val_ys = torch.cat([val_ys, y.cpu().view(-1,1)])

                if type(val_hidden_states) == type(None):
                    val_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
                else:
                    val_hidden_states = tuple(torch.cat([layer_hidden_state_all,layer_hidden_state_batch.cpu()]) 
                                            for layer_hidden_state_all,layer_hidden_state_batch in zip(val_hidden_states,hidden_states))

        self.visualize_layerwise_embeddings(hidden_states=val_hidden_states, masks=val_masks, ys=val_ys, title=title)

    def visualize_layerwise_embeddings(self, hidden_states, masks, ys, title, layers_to_visualize=[0,1,2,3,8,9,10,11]):
        title = title.replace('/','-')
        filename = f'results/embeddings/{title}.png'
        print('visualize_layerwise_embeddings for', title)

        num_layers = len(layers_to_visualize)
        fig = plt.figure(figsize=(24,(num_layers/4)*6)) #each subplot of size 6x6
        ax = [fig.add_subplot(int(num_layers/4),4,i+1) for i in range(num_layers)]
        ys = ys.numpy().reshape(-1)
        
        for i,layer_i in enumerate(layers_to_visualize):#range(hidden_states):
            layer_hidden_states = hidden_states[layer_i]
            averaged_layer_hidden_states = torch.div(layer_hidden_states.sum(dim=1),masks.sum(dim=1,keepdim=True))
            # layer_dim_reduced_vectors = dim_reducer.fit_transform(averaged_layer_hidden_states.numpy())
            layer_dim_reduced_vectors = self.reduce_embeddings_dimentionality(averaged_layer_hidden_states.numpy(), algorithm='TSNE')
            df = pd.DataFrame.from_dict({'x':layer_dim_reduced_vectors[:,0],'y':layer_dim_reduced_vectors[:,1],'label':ys})
            df.label = df.label.astype(int)
            sns.scatterplot(data=df,x='x',y='y',hue='label',ax=ax[i])
            # fig.suptitle(f"{title}: epoch {epoch}")
            ax[i].set_title(f"layer {layer_i+1}")
        
        plt.savefig(filename, format='png', pad_inches=0)


    #### ABSTRACT METHODS

    # Function to train and evaluate the model
    @abstractmethod
    def train_evaluate_model(self, training_parameters):
        pass


