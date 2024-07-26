import sys
sys.path.append("../../")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import plot_text_size_distribution, generate_word_cloud
from codes.language_model_handlers.ml_based_language_model_handler import MachineLearningLanguageModelHandler

import torch 
import gc 
from torch import nn

from icecream import ic
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
parser.add_argument('--model_name')
args = parser.parse_args()

model_name = args.model_name

# ---------------------------------------------------------------------------
##### Reading the data set

## Local
path = '../../../../' # '/home/guto/Documents/Projects/hate_speech/' 
datasets_path = path+'datasets/'

# ## Colab
# path = 'drive/My Drive/hate_speech/'
# datasets_path = path+'datasets/'

random_state = 42

hate_speech_file_names = {
    'racism': {'manual':'Manually_racism_EUROS.csv', 'GPT':'GPT_Manually_racism_EUROS.xlsx', 'true_positive_tab':'TP-RC', 'true_negative_tab':'ABT-RC', 'false_positive_tab':'FP-RC'},
    'sexism': {'manual':'Manually_sexism_EUROS.csv', 'GPT':'GPT_Manually_sexism_EUROS.xlsx', 'true_positive_tab':'TP-FS-SW', 'true_negative_tab':'ABT-FS-SW', 'false_positive_tab':'FP-FS-SW'},
    'ableism': {'manual':'Manually_ableism_EUROS.csv', 'GPT':'GPT_Manually_ableism_EUROS.csv'}
}

id_column = 'id'
text_column = 'text'
label_column = 'label'


hate_speech_df = pd.DataFrame()
non_hate_speech = pd.DataFrame()

new_labels = {0: 'non_hate_speech'} # {0: 'non-'+hate_speech_type, 1: hate_speech_type}

for i, hate_speech_type in enumerate(hate_speech_file_names):

    i += 1
    
    new_labels[i] = hate_speech_type

    # Reading only manually labelled datasets
    
    manual_file_name  = datasets_path+hate_speech_type+'/'+hate_speech_file_names[hate_speech_type]['manual']

    manual_hate_speech_df  = pd.read_csv(manual_file_name)
    manual_hate_speech_df[label_column] = [i]*manual_hate_speech_df.shape[0]

    # Reading GPT and manually labelled datasets

    GPT_file_name  = datasets_path+hate_speech_type+'/'+hate_speech_file_names[hate_speech_type]['GPT']
    
    if '.csv' in GPT_file_name:
        gpt_hate_speech_df = pd.read_csv(GPT_file_name)
        gpt_hate_speech_df[label_column] = [i]*gpt_hate_speech_df.shape[0]

    else:
        tab_true_positive = hate_speech_file_names[hate_speech_type]['true_positive_tab']
        tab_false_positive = hate_speech_file_names[hate_speech_type]['false_positive_tab']
        tab_true_negative = hate_speech_file_names[hate_speech_type]['true_negative_tab']

        gpt_dataframes = pd.read_excel(GPT_file_name, sheet_name=[tab_true_positive, tab_false_positive, tab_true_negative])

        true_positive_gpt = gpt_dataframes[tab_true_positive]
        false_positive_gpt = gpt_dataframes[tab_false_positive]
        true_negative_gpt = gpt_dataframes[tab_true_negative]

        true_positive_gpt[label_column] = [i]*true_positive_gpt.shape[0] # Hate speech
        true_negative_gpt[label_column] = [0]*true_negative_gpt.shape[0] # non-hate speech
        false_positive_gpt[label_column] = [0]*false_positive_gpt.shape[0] # non-hate speech

        gpt_hate_speech_df = pd.concat([true_positive_gpt[[text_column, label_column]]], axis=0)

        non_hate_speech = pd.concat([non_hate_speech, true_negative_gpt[[text_column, label_column]], false_positive_gpt[[text_column, label_column]]], axis=0)

    # Concatenating hate speech datasets
    hate_speech_df = pd.concat([hate_speech_df, manual_hate_speech_df, gpt_hate_speech_df], axis=0)

# Reading non-hate speech dataset
non_hate_speech_df = pd.read_csv(datasets_path+'non_hate_speech/GPT_non_hate_speech_EUROS.csv', low_memory=False)

non_hate_speech_df = non_hate_speech_df.sample(frac=1, random_state=random_state) # shuffling
non_hate_speech_df[label_column] = [0]*non_hate_speech_df.shape[0]


# if we have false positives and true negatives from GPT/manually classifications, let's concatenate with our data set
if non_hate_speech.shape[0] > 0:
    non_hate_speech_df = pd.concat([non_hate_speech_df, non_hate_speech], axis=0)

hate_speech_df = pd.concat([hate_speech_df, non_hate_speech_df], axis=0)

hate_speech_df = hate_speech_df.drop_duplicates(subset=[text_column]) # droping duplicates samples

hate_speech_df = hate_speech_df.sample(frac=1, random_state=random_state) # shuffling

ic(hate_speech_df[label_column].value_counts())

fp = open('/home/guto/Documents/Projects/hate_speech/euros_hashtags_terms.txt', 'r')
terms_hashtags_euros = [line.replace('\n', '') for line in fp.readlines()]
fp.close()

# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {

    'lower_case': True,
    'remove_emojis': False,
    'replace_emojis_by_text': True,
    'remove_stop_words': False, #True,
    'remove_numbers': False,
    'remove_hashtags': True,
    'remove_users': True,
    'remove_urls': True,
    'remove_non_text_characters': True,
    'lemmatize': False,
    'expand_contractions': False,
    'remove_money_values': False,
    'remove_apostrophe_contractions': False,
    'symbols_to_remove': ['*', '@', '<url>'],
    'remove_between_substrings': None, # [('_x0','d_')]
    'remove_terms_hashtags': terms_hashtags_euros+['\n']#+['euros', 'the euros', 'euro']
}

data_handler = DataHandler(df=hate_speech_df, text_column=text_column, label_column=label_column, extra_columns=[id_column], random_state=random_state)

data_handler.unsample()

train_data = data_handler.df.sample(frac=0.8, random_state=random_state)
test_data = data_handler.df.drop(train_data.index)

# Adding syntetic data to the training -----------------------------------------------------------------------------------------------------

# non_hate_speech_syntetic_df = pd.read_csv(datasets_path+'non_hate_speech/gpt_synthetic_data.csv', low_memory=False, sep=';')

# non_hate_temp = pd.DataFrame(columns=['text'])

# for column in non_hate_speech_syntetic_df:
    
#     non_hate_temp = pd.concat([non_hate_temp, non_hate_speech_syntetic_df[column].rename("text")], axis=0)

# non_hate_temp[label_column] = [0]*non_hate_temp.shape[0]

# train_data = pd.concat([train_data, non_hate_temp], axis=0)

# train_data = train_data.sample(frac=1, random_state=random_state) # shuffling

# ic(train_data.shape)

# ------------------------------------------------------------------------------------------------------------


for hate_speech_type in hate_speech_file_names:
    new_test_instances = pd.read_excel(datasets_path+'non_hate_speech/new_test_instances.xlsx', sheet_name=hate_speech_type)

    new_test_instances.rename(columns={'classification': label_column}, inplace=True)

    test_data = pd.concat([test_data, new_test_instances[[text_column, label_column]]])

train_data_handler = DataHandler(df=train_data, text_column=text_column, label_column=label_column, extra_columns=[id_column], random_state=random_state)
test_data_handler = DataHandler(df=test_data, text_column=text_column, label_column=label_column, extra_columns=[id_column], random_state=random_state)

train_data = train_data_handler.preprocess(setup=preprocessing_setup)
test_data = test_data_handler.preprocess(setup=preprocessing_setup)


# Language Model (ML models) Handler
# -----------------------------------------------------------------------
# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'FacebookAI/roberta-base'.replace('/','_')

# 'random forest', 'svm', 'decision tree', 'naive bayes', 'logistic regression'

ml_language_model_handler = MachineLearningLanguageModelHandler(
                                              ml_model_name=None, 
                                              llm_name=model_name,
                                              text_column = data_handler.text_column,
                                              processed_text_column=data_handler.get_text_column_name(),
                                              label_column=data_handler.label_column,
                                              batch_size=64,
                                              new_labels=new_labels,
                                              output_hidden_states=True)

ml_language_model_handler.load_llm_model(path='saved_models/', name_file=model_name)
                                                        
# Set up training arguments
training_args = {
    'ml_models_list':['random forest', 'decision tree'], # 'gradient boosting''logistic regression',
    'dataset_train':train_data,
    'dataset_test':test_data,
    'convert_to_one_hot_encoding': True
}

results, _, ml_models = ml_language_model_handler.train_evaluate_model(training_args=training_args)

for ml_model_name in results:
    print('Restuls', ml_model_name, results[ml_model_name])
    ml_language_model_handler.save_ml_model(ml_models[ml_model_name], path='saved_models/', ml_model_name=ml_model_name) # '''

