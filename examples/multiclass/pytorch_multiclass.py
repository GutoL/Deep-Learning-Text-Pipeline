import sys
sys.path.append("../../../GitHub/Deep-Learning-Text-Pipeline/")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import ExploratoryDataAnalysis
from codes.language_model_handlers.huggingface_language_model_handler import HuggingfaceLanguageModelHandler
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
from codes.language_model_handlers.ml_based_language_model_handler import MachineLearningLanguageModelHandler
from codes.explainable_ai_llm import ExplainableTransformerPipeline


import torch 
import gc 
from torch import nn
from transformers import TrainingArguments
from sklearn.utils import shuffle

# ---------------------------------------------------------------------------
##### Reading the data set


path = '' # Local
# path = 'drive/My Drive/hate_speech/datasets/manually_coded/' # Colab


dataset_types = ['homophobic', 'racism', 'sexism']

files_names = {
    'homophobic': 'homophobia.xlsx',
    'racism': 'Racism.xlsx',
    'sexism' : 'sexism.xlsx'
}

original_text_column = 'data_text'
label_column = 'label'

df_hate_speech = pd.DataFrame()

new_labels = {0: 'non-hate-speech'}

for i, hate_speech_type in enumerate(files_names):
    i += 1
    if hate_speech_type in dataset_types:
        hate_df = pd.read_excel(path+'../datasets/'+files_names[hate_speech_type])
        hate_df[label_column] = [i]*hate_df.shape[0]
        df_hate_speech = pd.concat([df_hate_speech, hate_df])
        new_labels[i] = hate_speech_type

df_non_hate_speech = pd.read_csv(path+'../datasets/'+'non_hate_speech.csv')[original_text_column].to_frame()

df_non_hate_speech = df_non_hate_speech.head(df_hate_speech.shape[0])

df_non_hate_speech[label_column] = [0]*df_non_hate_speech.shape[0]

df_hate_speech = pd.concat([df_hate_speech, df_non_hate_speech])

text_column = original_text_column

df_hate_speech.dropna(subset=[original_text_column], inplace=True)

df_hate_speech = df_hate_speech.sample(frac=1, random_state=42)

df_hate_speech.reset_index(drop=True, inplace=True)

print(df_hate_speech[label_column].value_counts())

 
# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {
    'lower_case': True,
    'remove_emojis': False,
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
    'symbols_to_remove': False # ['&', '$', '*']
}

data_handler = DataHandler(df=df_hate_speech, text_column=original_text_column, label_column=label_column)

data_handler.preprocess(setup=preprocessing_setup)

# exploratory_data_analysis = ExploratoryDataAnalysis()
# exploratory_data_analysis.plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# exploratory_data_analysis.generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

train_data, test_data = data_handler.split_train_test_dataset()

print('train_data', train_data.shape)
print('test_data', test_data.shape)
print('total of data to train and test:', train_data.shape[0]+test_data.shape[0],'//', (train_data.shape[0]+test_data.shape[0])/2, 'per class')

# Language Model Handler
# --------------------------------------------------------------------------------

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'cardiffnlp/twitter-roberta-base-offensive' #'cardiffnlp/twitter-roberta-base-offensive' #'roberta-base' #'bert-base-uncased'

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
    'epochs':20,
    'seed':42,
    'repetitions':1
}

metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

language_model_manager.save_model(path=path+'saved_models/', name_file=model_name)

# tokenizer, model = language_model_manager.load_model(path=path+'saved_models/'+dataset_type+'/', name_file=model_name)

_, metrics, classifications_df = language_model_manager.evaluate_model(test_data)

print(metrics)# '''

# classifications_df.to_csv(path+'results/predictions/'+dataset_type+'/'+model_name.replace('/','-')+'.csv', index=False)

#### Generating the Embeddings

# embeddings = language_model_manager.calculate_embeddings_local_model_with_batches(data=test_data)
# language_model_manager.plot_embeddings(file_name=path+'results/embeddings/'+dataset_type+'_'+model_name+'.png',
#                                         embeddings_results=embeddings, labels=test_data[data_handler.label_column].to_list(), 
#                                         algorithm='PCA')

# language_model_manager.plot_embeddings_layers(data=test_data, results_path=path+'results/embeddings/'+dataset_type+'/', filename=model_name.replace('/','-')+'_'+dataset_type+'.png',
#                                                   labels_to_replace={0:'non '+dataset_type, 1: dataset_type}, algorithm='TSNE', sample_size=100, number_of_layers_to_plot=1)

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