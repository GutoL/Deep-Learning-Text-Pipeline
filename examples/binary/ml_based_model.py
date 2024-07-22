import sys
sys.path.append("../../")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import plot_text_size_distribution, generate_word_cloud
from codes.language_model_handlers.ml_based_language_model_handler import MachineLearningLanguageModelHandler

import torch 
import gc 
from torch import nn
from transformers import TrainingArguments

original_text_column = 'text'
label_column = 'classification'
dataset_type = 'accuracy' #'accuracy' 'healthy' 'consistency'
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

id_column = 'id'
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
  


data_handler = DataHandler(df=df, text_column=original_text_column, label_column=label_column, extra_columns=[id_column])

data_handler.preprocess(setup=preprocessing_setup)

# plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

train_data, test_data = data_handler.split_train_test_dataset()

# Language Model (ML models) Handler
# -----------------------------------------------------------------------
# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'bert-base-uncased'

ml_language_model_handler = MachineLearningLanguageModelHandler(model_name=model_name,
                                              text_column = data_handler.text_column,
                                              processed_text_column=data_handler.get_text_column_name(),
                                              label_column=data_handler.label_column,
                                              batch_size=64,
                                              new_labels=new_labels,
                                              output_hidden_states=True)

tokenizer, model = ml_language_model_handler.load_model(path=path+'saved_models/'+dataset_type+'/', name_file=model_name)

# Set up training arguments
training_args = {
    'dataset_train':train_data,
    'dataset_test':test_data,
    'ml_model': 'gradient boosting', # 'random forest', 'svm', 'decision tree', 'naive bayes', 'logistic regression' 'gradient boosting'
    'seed':42
}


results = ml_language_model_handler.train_evaluate_model(training_args=training_args, iterations=1)
print('Testing results:', results) # '''

