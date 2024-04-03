import sys
sys.path.append("..")

import pandas as pd
from codes.data_handler import DataHandler
from codes.exploratory_data_analysis import ExploratoryDataAnalysis
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
from codes.explainable_ai_llm import ExplainableTransformerPipeline

import torch 
import gc 
from torch import nn

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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


data_handler = DataHandler(df=df, text_column=original_text_column, label_column=label_column)

data_handler.preprocess(setup=preprocessing_setup)

# exploratory_data_analysis = ExploratoryDataAnalysis()
# exploratory_data_analysis.plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# exploratory_data_analysis.generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

train_data, test_data = data_handler.split_train_test_dataset()

# Language Model (manual loop) Handler
# --------------------------------------------------------------------------------

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name='bert-base-uncased'

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
    'seed':42,
    'repetitions':1
}

# metrics, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

# language_model_manager.save_model(path=path+'saved_models/'+dataset_type+'/', name_file=model_name)

tokenizer, model = language_model_manager.load_model(path=path+'saved_models/'+dataset_type+'/', name_file=model_name)

_, metrics, classifications_df = language_model_manager.evaluate_model(test_data)

y_real = np.array(classifications_df['labels'].to_list())
y_pred = np.array(classifications_df['predictions'].to_list())

labels = list(new_labels.values())
disp = ConfusionMatrixDisplay.from_predictions(y_real, y_pred, display_labels=labels, xticks_rotation='vertical')
# disp.plot()
plt.show()

print(metrics)# '''

# classifications_df.to_csv(path+'results/predictions/'+dataset_type+'/'+model_name+'.csv', index=False)


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