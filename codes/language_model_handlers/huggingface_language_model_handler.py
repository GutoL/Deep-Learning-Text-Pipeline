import numpy as np 

from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import Dataset
import evaluate
from copy import deepcopy
from transformers import Trainer, EarlyStoppingCallback
from transformers import pipeline #, BertModel, BertTokenizer
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import torch
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


from codes.language_model_handlers.language_model_handler import LanguageModelHandler


class HuggingfaceLanguageModelHandler(LanguageModelHandler):
    
    def __init__(self, model_name, new_labels, text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        super().__init__(model_name, new_labels, text_column, label_column, output_hidden_states, batch_size, text_size_limit)
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
        if self.tokenizer is not None:
            number_of_tokens_added = self.tokenizer.add_tokens(new_tokens=new_tokens)

            if self.model is not None:
                print('### Resizing the model embeddings layer...')
                self.model.resize_token_embeddings(len(self.tokenizer))

            return number_of_tokens_added

    def prepare_dataset(self, data):
        
        self.num_labels = len(data[self.label_column].value_counts())
        
        hg_data = Dataset.from_pandas(data)        

        # Tokenize the dataset
        tokenized_dataset = hg_data.map(self.tokenize_dataset)
        
        return tokenized_dataset

    

    # Function to compute the metric
    def compute_metrics(self, predictions_labels): # predictions_labels: EvalPrediction
        predictions, labels = predictions_labels

        if self.output_hidden_states == True:
            predictions = np.argmax(predictions[0], axis=1)
        else:
            # probabilities = tf.nn.softmax(logits)
            predictions = np.argmax(predictions, axis=1) # the first element of EvalPrediction is the logists

        results = {
            'accuracy': self.metric_accuracy.compute(predictions=predictions, references=labels),
            'precision': self.metric_precision.compute(predictions=predictions, references=labels, average='weighted'),
            'recall': self.metric_recall.compute(predictions=predictions, references=labels, average='weighted'),
            'f1': self.metric_f1.compute(predictions=predictions, references=labels, average='weighted')
        }

        return results

    def train_evaluate_model(self, training_parameters):
        
        results_summary = {}
        detailed_metrics = ['eval_accuracy', 'eval_precision', 'eval_recall',  'eval_f1']

        self.tokenized_dataset_train = self.prepare_dataset(training_parameters['dataset_train'])
        self.tokenized_dataset_test = self.prepare_dataset(training_parameters['dataset_test'])

        self.create_dl_model()

        model = deepcopy(self.model)

        # Create adamw_torch optimizer manually
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
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
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=training_parameters['training_args'].learning_rate,
            betas=(training_parameters['training_args'].adam_beta1, training_parameters['training_args'].adam_beta2),
            eps=training_parameters['training_args'].adam_epsilon
        )

        
        self.trainer = MyTrainer(
            model=model, 
            args=training_parameters['training_args'], 
            train_dataset=self.tokenized_dataset_train, 
            eval_dataset=self.tokenized_dataset_test, 
            compute_metrics=self.compute_metrics, 
            loss_function=training_parameters['loss_function'],
            optimizers=(optimizer, None),
            tokenizer=self.tokenizer
        )


        # self.trainer = MyTrainer( # Trainer(
        #     model = model,
        #     args = training_parameters['training_args'],
        #     train_dataset = self.tokenized_dataset_train,
        #     eval_dataset = self.tokenized_dataset_test,
        #     compute_metrics = self.compute_metrics,
        #     loss_function = training_parameters['loss_function']            
        # )

        if training_parameters['early_stopping_patience']:
            self.trainer.callbacks = [EarlyStoppingCallback(early_stopping_patience=training_parameters['early_stopping_patience'])]

        self.trainer.train()

        results = self.trainer.evaluate(self.tokenized_dataset_test)

        for metric in results:
            if metric not in results_summary:
                if metric in detailed_metrics:
                    results_summary[metric] = [results[metric]["".join(metric.split('eval_'))]]
                else:
                    results_summary[metric] = [results[metric]]
            else:
                if metric in detailed_metrics:
                    results_summary[metric].append(results[metric]["".join(metric.split('eval_'))])
                else:
                    results_summary[metric].append(results[metric])

        # torch.cuda.empty_cache()
        
        self.model = self.trainer.model
        
        return results_summary, self.trainer
    
    def tokenize_dataset_for_embeddings(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']
    
    def prepare_dataset_for_embeddings(self, data):

        input_ids, att_masks = self.tokenize_dataset_for_embeddings(data[self.text_column].to_list())     
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
    
    def calculate_embeddings_local_model_with_batches(self, data: pd.DataFrame):
        
        data_loader = self.prepare_dataset_for_embeddings(data=data)
        
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
        return {self.model_name: X}

    def data_loader(self, dataframe, column):
        for row in dataframe.values:
            text = row[column] # Getting the text of the tweet

            print(text)
            
            if len(text.split()) > self.text_size_limit:
                yield text.split()[:self.text_size_limit]
            else:
                yield text    
            yield text 

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

    def plot_embeddings(self, embeddings_results, labels, algorithm='PCA', all_together=False):

        embeddings_df = pd.DataFrame()

        for model_name in embeddings_results:
            X = embeddings_results[model_name]

            reduced_data =  self.reduce_embeddings_dimentionality(X, algorithm)

            df = pd.DataFrame()
            df['model'] = [model_name]*X.shape[0]
            df['model'] = df['model'].apply(lambda i: str(i))

            df['label'] = labels

            df['first_dimension'] = reduced_data[:,0]
            df['second_dimension'] = reduced_data[:,1]

            embeddings_df = pd.concat([embeddings_df, df])

        if all_together:
            self.plot_embbedings_together(embeddings_df)
        else:
            self.plot_embbedings_separated(embeddings_df)

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


    def plot_embbedings_separated(self, embeddings_df):
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
        plt.savefig('embeddings.png')


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