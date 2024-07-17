import sys
sys.path.append("../../")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import plot_text_size_distribution, generate_word_cloud
from codes.language_model_handlers.huggingface_language_model_handler import HuggingfaceLanguageModelHandler
from codes.explainable_ai_llm import ExplainableTransformerPipeline

import torch 
import gc 
from torch import nn
from transformers import TrainingArguments

original_text_column = 'text'
label_column = 'classification'
dataset_type = 'consistency' #'accuracy' 'healthy' 'consistency'
extension = '.csv'

# Colab
path = 'drive/My Drive/TikTok/'
results_path = path+'results/chatGPT/jul_dec_2023/'+dataset_type+'_analysis_gpt_45/'
results_file_name = path+'codes/classification/datasets/'+dataset_type+'_analysis_jul_dec_2023.csv'

# local
path = '../' 
results_path = path+'datasets/'
results_file_name = path+'datasets/'+dataset_type+'_analysis_jul_dec_2023.csv'

replace_values={
    'accuracy': {
        'inaccurate': 0,
        'accurate': 1,
        'partially accurate': 2,
        'uncertain': 3,
    },
    'healthy': {
        'unhealthy':0,
        'healthy': 1,
        'partially healthy': 2,
        'uncertain': 3,
    },
    'consistency': {
        'not consistent': 0,
        'consistent': 1,
        'partially consistent': 2,
        'uncertain': 3,
    }
}

new_labels = {v: k for k, v in replace_values[dataset_type].items() if v in [0, 1]}

# ---------------------------------------------------------------------------
print(results_file_name)
df = pd.read_csv(results_file_name)

df.replace(replace_values[dataset_type], inplace=True)

df[label_column].value_counts()
# df.to_csv('consistency_analysis.csv', index=True)

# ---------------------------------------------------------------------------
# df.dropna(subset=[original_text_column], inplace=True)
df.dropna(subset=[label_column, original_text_column], inplace=True)
df.drop_duplicates(subset=[original_text_column], inplace=True)

df.rename(columns={original_text_column:'text', label_column: 'labels'}, inplace=True)

original_text_column = 'text'
label_column = 'labels'

df[label_column] = df[label_column].replace({2:1}) # converting partially 

df = df[df[label_column].isin([0, 1])]

df[label_column] = df[label_column].astype(int)

print(df[label_column].value_counts())

# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {
    'lower_case': False,
    'remove_emojis': True,
    'replace_emojis_by_text': False,
    'remove_stop_words': False, #True,
    'remove_numbers': False,
    'remove_users': True,
    'remove_urls': True,
    'remove_non_text_characters': True,
    'lemmatize': False,
    'expand_contractions': False,
    'remove_hashtags': False,
    'remove_money_values': False,
    'remove_apostrophe_contractions': False,
    'symbols_to_remove': ['*', '@'],
    'remove_between_substrings': None, # [('_x0','d_')]
    'remove_terms_hashtags': None # ['euros', 'the euros', 'euro']
} 


data_handler = DataHandler(df=df, text_column=original_text_column, label_column=label_column)

data_handler.preprocess(setup=preprocessing_setup)

# plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

train_data, test_data = data_handler.split_train_test_dataset()

print('train_data', train_data.shape)
print('test_data', test_data.shape)
print('total of data to train and test:', train_data.shape[0]+test_data.shape[0], (train_data.shape[0]+test_data.shape[0])/2, 'per class')

# Language Model (Huggingface based) Handler
# --------------------------------------------------------------------------------


# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'bert-base-uncased'

language_model_manager = HuggingfaceLanguageModelHandler(model_name=model_name, # 'roberta-base',
                                              text_column=data_handler.get_text_column_name(),
                                              label_column=data_handler.label_column,
                                              new_labels=new_labels,
                                              output_hidden_states=True)


# sentences = test_data[data_handler.get_text_column_name()].to_list()

# embeddings = language_model_manager.sentences_to_embedding(sentences=sentences, model_name=None, pre_trained_model=None)

# print(embeddings)


# Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./sentiment_transfer_learning_transformer/",
#     logging_dir='./sentiment_transfer_learning_transformer/logs',
#     logging_strategy='epoch',
#     logging_steps=100,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=0.00005, # 0.000005
#     save_strategy='epoch',
#     save_steps=100,
#     evaluation_strategy='epoch',
#     eval_steps=100,
#     load_best_model_at_end=True,
#     num_train_epochs=15,
#     # seed=42
# )


training_args = TrainingArguments(num_train_epochs=10, 
                               learning_rate=2e-5,
                               seed=42,
                               optim='adamw_torch',
                               output_dir='output',
                               overwrite_output_dir=True,
                               evaluation_strategy='epoch',
                               do_eval=True,
                               full_determinism=True)


training_parameters = {
    'training_args' : training_args,
    'dataset_train':train_data,
    'dataset_test':test_data,
    'loss_function': nn.CrossEntropyLoss(),
    'early_stopping_patience':2,
    'iterations':1    
}


# results, trainer = language_model_manager.train_evaluate_model(training_parameters)

# print('Testing results:', results)

# language_model_manager.save_model(path=path+'saved_models/', name_file=model_name)# '''

language_model_manager.load_model(path=path+'saved_models/', name_file=model_name)




#### Generating the Embeddings

#### Generating the Embeddings
language_model_manager.calculate_embeddings_model_layers(test_data, only_last_layer=False)

language_model_manager.plot_embeddings_layers(data=test_data, results_path=path+'results/embeddings/'+dataset_type+'/', filename=model_name+'_'+dataset_type+'.png',
                                                  labels_to_replace={0:'non '+dataset_type, 1: dataset_type}, algorithm='TSNE', sample_size=100, number_of_layers_to_plot=1)


####### INTEGRATED GRADIENTS

exp_model = ExplainableTransformerPipeline(model=language_model_manager.model, tokenizer=language_model_manager.tokenizer,
                                           device=language_model_manager.device)

# Using integrated gradients to plot the word importance for few samples
samples = test_data[test_data[label_column]==1].sample(n=3, random_state=10)

print(data_handler.df.iloc[samples.index][data_handler.text_column].values)
samples = samples[data_handler.get_text_column_name()]


for i, sample in enumerate(samples):
    exp_model.explain(sample, file_name=path+'results/xai/'+dataset_type+'/'+model_name+'_'+str(i)+'.png')


# results = exp_model.get_most_impactful_words_for_dataset(dataset=test_data,
#                                                column_text=data_handler.get_text_column_name(),
#                                                threshold=0.1,
#                                                keyword=dataset_type,
#                                                method='integrated_gradients',
#                                                n=50)
# results
print('Done!')