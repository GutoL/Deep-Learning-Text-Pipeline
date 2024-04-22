import sys
sys.path.append("../../../GitHub/Deep-Learning-Text-Pipeline/")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import plot_text_size_distribution, generate_word_cloud
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
from codes.explainable_ai_llm import ExplainableTransformerPipeline


import torch 
import gc 
from torch import nn
from transformers import TrainingArguments
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------
##### Reading the data set

## Local
path = '' 
datasets_path = path+'../datasets/'

# ## Colab
# path = 'drive/My Drive/hate_speech/'
# datasets_path = path+'datasets/'

random_state = 42

dataset_type = 'sexism'

text_column = 'text'
label_column = 'label'

# ----------------------------------------------------------------------------------------------------------------------------------------
if dataset_type == 'sexism':
    GPT_file_name = datasets_path+'GPT_and_manually_coded/Labelled_Dataset_Sexism_Against_Women_and_Feminine_Slurs_Euros.xlsx'
    tab_true_positive = 'TP-FS-SW'
    tab_false_positive = 'FP-FS-SW'
    tab_true_negative = 'ABT-FS-SW'
    
    hate_speech_df  = pd.read_excel(datasets_path+'manually_coded/sexism.xlsx')
    

elif dataset_type == 'homophobia':
    GPT_file_name = datasets_path+'GPT_and_manually_coded/Labelled-Dataset-Homophobia-Homophobic_Slurs.xlsx'
    tab_true_positive = 'TP-HP'
    tab_false_positive = 'FP-HP'
    tab_true_negative = 'About'

    hate_speech_df  = pd.read_excel(datasets_path+'manually_coded/homophobia.xlsx')


elif dataset_type == 'ableism':
    GPT_file_name = datasets_path+'GPT_and_manually_coded/Labelled_Dataset_Ableism_EUROS.xlsx'
    tab_true_positive = 'TP-AB-AS'
    tab_false_positive = 'FP-AB-AS'
    tab_true_negative = 'About'

    hate_speech_df  = pd.read_excel(datasets_path+'manually_coded/ableism.xlsx')

# ----------------------------------------------------------------------------------------------------------------------------------------

# Reading only manually labelled datasets
hate_speech_df[label_column] = [1]*hate_speech_df.shape[0]

non_hate_speech_df = pd.read_csv(datasets_path+'manually_coded/non_hate_speech.csv')
non_hate_speech_df[label_column] = [0]*non_hate_speech_df.shape[0]

# Reading GPT + manually labelled datasets
gpt_dataframes = pd.read_excel(GPT_file_name, sheet_name=[tab_true_positive, tab_false_positive, tab_true_negative])

true_positive_gpt = gpt_dataframes[tab_true_positive]
false_positive_gpt = gpt_dataframes[tab_false_positive]
true_negative_gpt = gpt_dataframes[tab_true_negative]

true_positive_gpt[label_column] = [1]*true_positive_gpt.shape[0]
true_negative_gpt[label_column] = [0]*true_negative_gpt.shape[0]
false_positive_gpt[label_column] = [0]*false_positive_gpt.shape[0]

# Putting all toghether
hate_speech_df = pd.concat([hate_speech_df[[text_column, label_column]], true_positive_gpt[[text_column, label_column]]], axis=0)

non_hate_speech_df = pd.concat([non_hate_speech_df[[text_column, label_column]], true_negative_gpt[[text_column, label_column]], false_positive_gpt[[text_column, label_column]]], axis=0)

hate_speech_df = pd.concat([hate_speech_df, non_hate_speech_df], axis=0)

hate_speech_df = hate_speech_df.sample(frac=1, random_state=random_state) # shuffling

print(hate_speech_df[label_column].value_counts())

# Defining the output labels to the model
new_labels = {0: 'non-'+dataset_type, 1: dataset_type}

# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {
    'lower_case': True,
    'remove_emojis': True,
    'replace_emojis_by_text': False,
    'remove_stop_words': True,
    'remove_numbers': False,
    'remove_users': True,
    'remove_urls': True,
    'remove_non_text_characters': True,
    'lemmatize': False,
    'expand_contractions': False,
    'remove_hashtags': True,
    'remove_money_values': False,
    'remove_apostrophe_contractions': False,
    'symbols_to_remove': False, # ['&', '$', '*']
    'remove_between_substrings': None # [('_x0','d_')]
}


data_handler = DataHandler(df=hate_speech_df, text_column=text_column, label_column=label_column)

data_handler.preprocess(setup=preprocessing_setup)

# plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

print(data_handler.df[label_column].value_counts())

train_data, test_data = data_handler.split_train_test_dataset()

print('train_data', train_data.shape)
print('test_data', test_data.shape)
print('total of data to train and test:', train_data.shape[0]+test_data.shape[0],'//', (train_data.shape[0]+test_data.shape[0])/2, 'per class')

# Language Model Handler
# --------------------------------------------------------------------------------

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'FacebookAI/roberta-base' #'cardiffnlp/twitter-roberta-base-offensive' #'roberta-base' #'bert-base-uncased'

print('*** Model:', model_name)


language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                                    text_column = data_handler.text_column,
                                                    processed_text_column=data_handler.get_text_column_name(),
                                                    label_column=data_handler.label_column,
                                                    new_labels=new_labels,
                                                    output_hidden_states=True)

# Set up training arguments
training_parameters = {
    'learning_rate':0.000001,
    # 'eps':1e-8,
    'betas':(0.9, 0.999),
    'weight_decay':0.01,
    'loss_function':nn.CrossEntropyLoss(),
    'dataset_train':train_data,
    'dataset_test':test_data,
    'epochs':10,
    'seed':random_state,
    'repetitions':1,
    'model_file_name': path+'saved_models/'+model_name
}

metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

# language_model_manager.save_model(path=path+'saved_models/'+dataset_type+'/', name_file=model_name)

_, metrics, classifications_df = language_model_manager.evaluate_model(test_data)

print(metrics)# '''

# classifications_df.to_csv(path+'results/predictions/'+dataset_type+'/'+model_name.replace('/','-')+'.csv', index=False)

#### Generating the Embeddings

# embeddings = language_model_manager.calculate_embeddings_model_layers(data=test_data, only_last_layer=True)
# print(embeddings['hidden_states'][0].shape)

# language_model_manager.plot_embeddings_layers(data=test_data, results_path=path+'results/embeddings/'+dataset_type+'/', filename=model_name.replace('/','-')+'_'+dataset_type+'.png',
#                                                   labels_to_replace=new_labels, algorithm='TSNE', sample_size=100, number_of_layers_to_plot=1)

####### INTEGRATED GRADIENTS

# exp_model = ExplainableTransformerPipeline(model=language_model_manager.model, tokenizer=language_model_manager.tokenizer,
#                                            device=language_model_manager.device)

# ## Using integrated gradients to plot the word importance for few samples
# # samples = test_data[test_data[label_column]==1].sample(n=3, random_state=10)

# # print(data_handler.df.iloc[samples.index][data_handler.text_column].values)
# # samples = samples[data_handler.get_text_column_name()]

# samples = [
#             "Mexican Waves are the gayest thing ever. #Wimbledon #Euro2012",
#             "What a brilliant tackle. Football now a game for pansies! 🤬🤬🤬🤬 #SWISPA #EURO2020",
#             "Ronaldo might be a homo but his free kicks are #phenomenal #Euro2012",
#             # "Sodomy England team have their ref on side #eng #ENGvsSEN",            
#             "Since Greizmann grew his hair like a Pansy, he hasn't been the same player #Fra #EURO2020"
#            ]

# samples = ["Watching Euro 2012 and confused by the lack of helmets and shoulder pads."]

# for i, sample in enumerate(samples):
#     print('----------------------------------------------')
#     print(sample)
#     exp_model.explain(sample, file_name=path+'results/xai/'+dataset_type+'/'+model_name.replace('/','-')+'_'+str(i)+'.png')
#     # exp_model.plot_word_importance(sample, clean_text=False, bar=True)

    


# results = exp_model.get_most_impactful_words_for_dataset(dataset=test_data,
#                                                column_text=data_handler.get_text_column_name(),
#                                                threshold=0.1,
#                                                keyword=dataset_type,
#                                                method='integrated_gradients',
#                                                n=50) #'''

print('Done!')