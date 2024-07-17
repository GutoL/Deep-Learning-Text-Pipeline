import sys
sys.path.append("/home/guto/Documents/GitHub/Deep-Learning-Text-Pipeline/")

import pandas as pd
import argparse
from icecream import ic 
ic.configureOutput(includeContext=True) 

from codes.data_handler import DataHandler

from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler

from codes.esemble.ensemble_features_llm import EnsembleFeaturesLlm

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

train_data_handler = DataHandler(df=train_data, text_column=text_column, label_column=label_column, random_state=random_state)
test_data_handler = DataHandler(df=test_data, text_column=text_column, label_column=label_column, random_state=random_state)

train_data = train_data_handler.preprocess(setup=preprocessing_setup)
test_data = test_data_handler.preprocess(setup=preprocessing_setup)



# ----------------------------------------------------------------------------------
## Creating BERT-based model

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

# model_name = 'vinai/bertweet-base' #'cardiffnlp/twitter-roberta-base-offensive' #'roberta-base' #'bert-base-uncased' FacebookAI/roberta-base vinai/bertweet-base

print('*** Model:', model_name)


# language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
#                                                     text_column = data_handler.text_column,
#                                                     processed_text_column=data_handler.get_text_column_name(),
#                                                     label_column=data_handler.label_column,
#                                                     new_labels=new_labels,
#                                                     output_hidden_states=True)


# model_name = model_name.replace('/', '_')

# # Set up training arguments
# training_parameters = {
#     'learning_rate':0.000001,
#     # 'eps':1e-8,
#     'betas':(0.9, 0.999),
#     'weight_decay':0.01,
#     'loss_function':torch.nn.CrossEntropyLoss(),
#     'dataset_train':train_data,
#     'dataset_test':test_data,
#     'epochs':20,
#     'seed':42,
#     'repetitions':1,
#     'model_file_name': 'saved_models/'+'multi_class_'+model_name,
#     #Early stop mechanism
#     'patience': 2,
#     'min_delta': 0.1
# }

# metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

# language_model_manager.save_model(path=path+'saved_models/', name_file='multi_class_'+model_name)

# language_model_manager.load_model(path=path+'saved_models/', name_file='multi_class_'+model_name)


# _, metrics, llm_classifications_df = language_model_manager.evaluate_model(test_data)

# plot_confusion_matrix(predictions=llm_classifications_df['predictions'].to_list(), 
#                       actuals=llm_classifications_df['labels'].to_list(), 
#                       class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
#                       file_name='confusion_matrix/'+model_name+'_confusion_matrix.png')

'''
esemble = EnsembleFeaturesLlm()

dynamic_weights = False

ensemble_classification, llms_predictions = esemble.perform_ensemble_llms(data=test_data, 
                                                                          models_names=['bert-base-uncased', 'FacebookAI/roberta-base', 'vinai/bertweet-base'],
                                                                          dynamic_weights=dynamic_weights)

final_classification = pd.DataFrame()

final_classification['text'] = test_data['text']
final_classification[label_column] = test_data[label_column]
final_classification['ensemble'] = np.argmax(ensemble_classification, axis=1).tolist()

for llm_name in llms_predictions:
    
    llm_pred = np.argmax(llms_predictions[llm_name], axis=1).tolist()

    llm_name = llm_name.replace('/', '_')

    final_classification[llm_name] = llm_pred

    plot_confusion_matrix(predictions=llm_pred, 
                      actuals=final_classification[label_column].to_list(), 
                      class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
                      file_name='confusion_matrix/'+llm_name+'_confusion_matrix.png')

ic(final_classification)

final_classification.to_csv('ensemble.csv', index=False)

if dynamic_weights:
    confusion_matrix_name = 'confusion_matrix/dynamic_ensemble_confusion_matrix.png'
else:
    confusion_matrix_name = 'confusion_matrix/static_ensemble_confusion_matrix.png'

plot_confusion_matrix(predictions=final_classification['ensemble'].to_list(), 
                      actuals=final_classification[label_column].to_list(), 
                      class_names=[class_name.capitalize().replace('_', ' ') for class_name in list(new_labels.values())],
                      file_name=confusion_matrix_name) # '''