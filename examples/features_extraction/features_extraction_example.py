import sys
sys.path.append("/home/guto/Documents/GitHub/Deep-Learning-Text-Pipeline/")

import pandas as pd
import argparse
from icecream import ic 
ic.configureOutput(includeContext=True) 

from codes.data_handler import DataHandler
from codes.features_extraction.features_extractor import FeaturesExtractor
from codes.features_extraction.machine_learning_manager import MachineLearningManager
from codes.features_extraction.LIWC.my_liwc import CustomLiwc

from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler

from codes.features_extraction.enssemble_features_llm import EnsembleFeaturesLlm

import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np



def plot_confusion_matrix(predictions, actuals, class_names, file_name):
    """
    Plots a confusion matrix for a multiclass classification problem.
    
    Parameters:
    predictions (list or array-like): Predicted labels.
    actuals (list or array-like): Actual labels.
    class_names (list): List of class names corresponding to the labels.
    """
    # Compute the confusion matrix
    cm = confusion_matrix(actuals, predictions)
    
    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    # Add labels and titles
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Show the plot
    # plt.show()
    plt.savefig(file_name)

## Create an ArgumentParser object
# parser = argparse.ArgumentParser()
# parser.add_argument('--hate_speech_type', required=True)
# args = parser.parse_args()

# ---------------------------------------------------------------------------
##### Reading the data set

## Local
path = '' # '/home/guto/Documents/Projects/hate_speech/' 
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

non_hate_speech_syntetic_df = pd.read_csv(datasets_path+'non_hate_speech/gpt_synthetic_data.csv', low_memory=False, sep=';')

non_hate_temp = pd.DataFrame()

for column in non_hate_speech_syntetic_df:
    non_hate_temp = pd.concat([non_hate_temp, non_hate_speech_syntetic_df[column]], axis=0)

non_hate_temp[label_column] = [0]*non_hate_temp.shape[0]

non_hate_speech_df = pd.concat([non_hate_speech_df, non_hate_temp], axis=0)

# if we have false positives and true negatives from GPT/manually classifications, let's concatenate with our data set
if non_hate_speech.shape[0] > 0:
    non_hate_speech_df = pd.concat([non_hate_speech_df, non_hate_speech], axis=0)

hate_speech_df = pd.concat([hate_speech_df, non_hate_speech_df], axis=0)

hate_speech_df = hate_speech_df.drop_duplicates(subset=[text_column]) # droping duplicates samples

hate_speech_df = hate_speech_df.sample(frac=1, random_state=random_state) # shuffling

fp = open(path+'/home/guto/Documents/Projects/hate_speech/euros_hashtags_terms.txt', 'r')
terms_hashtags_euros = [line.replace('\n', '') for line in fp.readlines()]
fp.close()


# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {

    'lower_case': True,
    'remove_emojis': True,
    'replace_emojis_by_text': False,
    'remove_stop_words': False, #True,
    'remove_numbers': False,
    'remove_users': True,
    'remove_urls': True,
    'remove_non_text_characters': True,
    'lemmatize': False,
    'expand_contractions': False,
    'remove_hashtags': True,
    'remove_money_values': False,
    'remove_apostrophe_contractions': False,
    'symbols_to_remove': ['*', '@', '<url>'],
    'remove_between_substrings': None, # [('_x0','d_')]
    'remove_terms_hashtags': terms_hashtags_euros+['\n']#+['euros', 'the euros', 'euro']

}


data_handler = DataHandler(df=hate_speech_df, text_column=text_column, label_column=label_column, random_state=random_state)

data_handler.unsample()

train_data = data_handler.df.sample(frac=0.8, random_state=random_state)
test_data = data_handler.df.drop(train_data.index)

for hate_speech_type in hate_speech_file_names:
    new_test_instances = pd.read_excel(datasets_path+'non_hate_speech/guto_vitor_new_test_instances.xlsx', sheet_name=hate_speech_type)

    new_test_instances.rename(columns={'classification': label_column}, inplace=True)

    test_data = pd.concat([test_data, new_test_instances[[text_column, label_column]]])

train_data_handler = DataHandler(df=train_data, text_column=text_column, label_column=label_column, random_state=random_state)
test_data_handler = DataHandler(df=test_data, text_column=text_column, label_column=label_column, random_state=random_state)

train_data = train_data_handler.preprocess(setup=preprocessing_setup)
test_data = test_data_handler.preprocess(setup=preprocessing_setup)


# ---------------------------------------------------------------------------
## Creating the text features
hate_speech_terms = pd.read_excel('/home/guto/Documents/Projects/hate_speech/keyword_search/key_words/ableism/ableist_terms.xlsx', sheet_name='Our list').values
hate_speech_terms = [term[0].lower() for term in hate_speech_terms]

features_columns=['list_word_count', 'list_word_proportion', 'subjectivity', 'negative', 'neutral', 'positive']

feature_extractor = FeaturesExtractor(features_columns)

train_data = feature_extractor.extract_all_features(df=train_data, text_column=data_handler.get_text_column_name(),
                                       terms=hate_speech_terms, liwc_dict_path='../../codes/features_extraction/LIWC/dictionaries/dehumanization-dictionary.dicx')

test_data = feature_extractor.extract_all_features(df=test_data, text_column=data_handler.get_text_column_name(),
                                       terms=hate_speech_terms, liwc_dict_path='../../codes/features_extraction/LIWC/dictionaries/dehumanization-dictionary.dicx')

# ---------------------------------------------------------------------------
# Creating Machine Learning model

ml_manager = MachineLearningManager(features_names=feature_extractor.features_columns, label_column=label_column)

# training_args = {
#     'dataset_train':train_data,
#     'dataset_test':test_data,
#     'ml_model': 'gradient boosting', # 'random forest', 'svm', 'decision tree', 'naive bayes', 'logistic regression' 'gradient boosting'
#     'seed':random_state
# }

# metrics, ml_predictions_df = ml_manager.train_evaluate_ml_model(training_args)

# ic(metrics)

# plot_confusion_matrix(predictions=ml_predictions_df['predictions'].to_list(), 
#                       actuals=ml_predictions_df['labels'].to_list(), 
#                       class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
#                       file_name=training_args['ml_model'].replace(' ', '_')+'_confusion_matrix.png')

training_args = {
    'dataset_train':train_data,
    'dataset_test':test_data,
    'hidden_size': [64, 32, 16],
    'num_classes': len(hate_speech_df[label_column].value_counts()),
    'learning_rate': 0.0001,
    'epochs': 100,
    'batch_size': 32,
    'ml_model': 'fully_connected_features_extraction'
}

metrics, ml_predictions_df = ml_manager.train_evaluate_nn_model(training_args)

plot_confusion_matrix(predictions=ml_predictions_df['predictions'].to_list(), 
                      actuals=ml_predictions_df['labels'].to_list(), 
                      class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
                      file_name='confusion_matrix/'+training_args['ml_model'].replace(' ', '_')+'_confusion_matrix.png')

# predictions_df.to_csv(hate_speech_type+'_predictions.csv', index=False) #'''



# ----------------------------------------------------------------------------------
## Creating BERT-based model

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'vinai/bertweet-base' #'cardiffnlp/twitter-roberta-base-offensive' #'roberta-base' #'bert-base-uncased' FacebookAI/roberta-base vinai/bertweet-base

print('*** Model:', model_name)


language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                                    text_column = data_handler.text_column,
                                                    processed_text_column=data_handler.get_text_column_name(),
                                                    label_column=data_handler.label_column,
                                                    new_labels=new_labels,
                                                    output_hidden_states=True)


model_name = model_name.replace('/', '_')

# Set up training arguments
training_parameters = {
    'learning_rate':0.000001,
    # 'eps':1e-8,
    'betas':(0.9, 0.999),
    'weight_decay':0.01,
    'loss_function':torch.nn.CrossEntropyLoss(),
    'dataset_train':train_data,
    'dataset_test':test_data,
    'epochs':20,
    'seed':42,
    'repetitions':1,
    'model_file_name': 'saved_models/'+'multi_class_'+model_name,
    #Early stop mechanism
    'patience': 2,
    'min_delta': 0.1
}

metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

# language_model_manager.save_model(path=path+'saved_models/', name_file='multi_class_'+model_name)

# language_model_manager.load_model(path=path+'saved_models/', name_file='multi_class_'+model_name)


_, metrics, llm_classifications_df = language_model_manager.evaluate_model(test_data)

plot_confusion_matrix(predictions=llm_classifications_df['predictions'].to_list(), 
                      actuals=llm_classifications_df['labels'].to_list(), 
                      class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
                      file_name='confusion_matrix/'+model_name+'_confusion_matrix.png')

ensemble = EnsembleFeaturesLlm(features_names=feature_extractor.features_columns)

ensemble_classification = ensemble.perform_ensemble(data=test_data, llm_tokenizer=language_model_manager.tokenizer, llm=language_model_manager.model,
                          feature_ml=ml_manager.tabular_model)

final_classification = pd.DataFrame()

final_classification['text'] = ml_predictions_df['text']
final_classification['Neural Network'] = ml_predictions_df['predictions']
final_classification['LLM'] = llm_classifications_df['predictions']
final_classification['ensemble'] = np.argmax(ensemble_classification, axis=1).tolist()
final_classification['labels'] = llm_classifications_df['labels']

ic(final_classification)

final_classification.to_csv('ensemble.csv', index=False)

plot_confusion_matrix(predictions=final_classification['ensemble'].to_list(), 
                      actuals=final_classification['labels'].to_list(), 
                      class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
                      file_name='confusion_matrix/ensemble_confusion_matrix.png')