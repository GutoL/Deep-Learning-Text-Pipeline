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
        
        ## Save tokenizer
        self.tokenizer.save_pretrained(path+name_file)

        ## Save model
        # model_path = os.path.join(path, name_file, name_file+'.pth')
        # torch.save(self.model, model_path)
        self.model.save_pretrained(path+name_file)
        

    def load_model(self, path, name_file):
        name_file = name_file.replace('/','-')
        
        ## Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path+name_file)
        
        ## Load Model
        self.model = AutoModelForSequenceClassification.from_pretrained(path+name_file)
        # self.model = torch.load(path+name_file+'/'+name_file+'.pth')
        self.model.to(self.device)

        return self.tokenizer, self.model

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

    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=self.max_length, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    
    def prepare_dataset(self, data, shuffle=True):

        input_ids, att_masks = self.tokenize_dataset(data[self.text_column].to_list())

        Y = torch.LongTensor(data[self.label_column].to_list())
        
        # move on device (GPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        dataset = TensorDataset(input_ids, att_masks, Y)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

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
        data = [(row[self.text_column], row[self.label_column],) for _, row in df.iterrows()]
        sampler = torch.utils.data.sampler.SequentialSampler(data)
        batch_sampler = torch.utils.data.BatchSampler(sampler,batch_size=self.batch_size if self.batch_size > 0 else len(data), drop_last=False)
        
        for batch in batch_sampler:
            encoded_batch_data = self.tokenizer.batch_encode_plus([data[i][0] for i in batch], add_special_tokens=True, max_length=self.max_length, padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')

            seq = encoded_batch_data['input_ids'].clone().detach()
            mask = encoded_batch_data['attention_mask'].clone().detach()

            yield (seq, mask), torch.LongTensor([data[i][1] for i in batch])

    def calculate_embeddings_model_layers(self, data, only_last_layer):
        test_masks, test_ys = torch.zeros(0, self.max_length), torch.zeros(0, 1)

        test_hidden_states = None

        for x, y in self.get_bert_encoded_data_in_batches(data):
            sent_ids, masks = x
            sent_ids = sent_ids.to(self.device)
            masks = masks.to(self.device)
            
            with torch.no_grad():
                model_out = self.model(sent_ids, masks)
            
                hidden_states = model_out.hidden_states[1:]

                if only_last_layer:
                    hidden_states = tuple([hidden_states[-1]]) # getting only the embeddings from the last layer
                
                test_masks = torch.cat([test_masks, masks.cpu()])
                test_ys = torch.cat([test_ys, y.cpu().view(-1,1)])

                if type(test_hidden_states) == type(None):
                    test_hidden_states = tuple(layer_hidden_states.cpu() for layer_hidden_states in hidden_states)
                else:
                    test_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu()]) 
                                            for layer_hidden_state_all, layer_hidden_state_batch in zip(test_hidden_states, hidden_states))
        
        return {'hidden_states':test_hidden_states, 'masks':test_masks, 'ys':test_ys}
    
    def calculate_average_embeddings(self, hidden_states, masks, layers_ids):
        all_averaged_layer_hidden_states = {}
        
        for layer_i in range(len(hidden_states)):
            if layer_i in layers_ids:
                all_averaged_layer_hidden_states[layer_i] = torch.div(hidden_states[layer_i].sum(dim=1), masks.sum(dim=1, keepdim=True))
        
        return all_averaged_layer_hidden_states

    def plot_embeddings_layers(self, data, results_path, filename, sample_size=None, algorithm='TSNE', labels_to_replace=None, number_of_layers_to_plot=2):
        
        if sample_size:
            reduced_data = pd.DataFrame()

            for label_value in set(data[self.label_column].tolist()):
                
                temp_df = data[data[self.label_column] == label_value]
                
                if sample_size > temp_df.shape[0]:
                    sample_size = temp_df.shape[0]
                
                reduced_data = pd.concat([reduced_data, temp_df.sample(n=sample_size, random_state=self.random_state)])

            data = reduced_data
        
        embeddings = self.calculate_embeddings_model_layers(data, only_last_layer=False)
                
        test_hidden_states = embeddings['hidden_states']
        test_masks = embeddings['masks']
        test_ys = embeddings['ys']

        layers_to_visualize = [] # [0,1,10,11]

        for x in range(number_of_layers_to_plot):
            layers_to_visualize.append(x) # ploting first N layers
            layers_to_visualize.append(len(test_hidden_states)-x-1) # ploting last N layers
        
        layers_to_visualize.sort()

        all_averaged_layer_hidden_states = self.calculate_average_embeddings(test_hidden_states, test_masks, layers_ids=layers_to_visualize)

        if labels_to_replace:
            test_ys = data[self.label_column].map(labels_to_replace).to_list()

        self.visualize_layerwise_embeddings(all_averaged_layer_hidden_states=all_averaged_layer_hidden_states, ys=test_ys, results_path=results_path, filename=filename, algorithm=algorithm)
            

    # Based on: https://medium.com/towards-data-science/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568
    def visualize_layerwise_embeddings(self, all_averaged_layer_hidden_states, ys, results_path, filename, algorithm):
        
        num_layers = len(all_averaged_layer_hidden_states)
        
        plot_size_factor = 5
        
        fig = plt.figure(figsize=((plot_size_factor*num_layers*2), plot_size_factor+2))
        ax = [fig.add_subplot(int(num_layers/2), num_layers, i+1) for i in range(num_layers)]

        if type(ys) != list:
            ys = ys.numpy().reshape(-1)
        
        ys = [y.capitalize().replace('_',' ') for y in ys]

        for i, layer_i in enumerate(all_averaged_layer_hidden_states): #range(hidden_states):
            
            layer_dim_reduced_vectors = self.reduce_embeddings_dimentionality(all_averaged_layer_hidden_states[layer_i].numpy(), algorithm=algorithm)
            
            df = pd.DataFrame.from_dict({'x':layer_dim_reduced_vectors[:,0], 'y':layer_dim_reduced_vectors[:,1], 'label':ys})
            
            # df.label = df.label.astype(int)

            sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[i], palette=sns.color_palette(cc.glasbey, n_colors=self.num_labels))
            # fig.suptitle(f"{title}: epoch {epoch}")
            ax[i].set_title(f"layer {layer_i+1}")
            
            # Removing axis values
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].legend([],[], frameon=False)

        print('visualize_layerwise_embeddings for', results_path+filename)
        
        plt.legend(bbox_to_anchor=(1, 0), loc="lower right", borderaxespad=0, bbox_transform=fig.transFigure, ncol=3)
        # plt.legend(bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=fig.transFigure, ncol=3)
        
        plt.savefig(results_path+filename, format='png', pad_inches=0) # f'results/embeddings/{title}.png'


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
    

