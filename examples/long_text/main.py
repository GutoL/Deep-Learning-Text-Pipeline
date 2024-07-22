import sys
sys.path.append("../../")

from codes.language_model_handlers.long_texts.bert_with_pooling import BertClassifierWithPooling
from codes.data_handler import DataHandler
import pandas as pd

import torch
import gc

# ---------------------------------------------------------------------------
##### Reading the data set

## Local
path = '' 
datasets_path = path+'datasets/splitted/'

# ## Colab
# path = 'drive/My Drive/hate_speech/'
# datasets_path = path+'datasets/'

random_state = 42

dataset_type = 'accuracy'

id_column = 'id'
text_column = 'text'
label_column = 'classification'

train_file_name = datasets_path+'train_bright_data_accuracy.xlsx'
test_file_name = datasets_path+'test_bright_data_accuracy.xlsx'


train_data = pd.read_excel(train_file_name)
test_data = pd.read_excel(test_file_name)

train_data['is_training'] = [True] * train_data.shape[0]
test_data['is_training'] = [False] * test_data.shape[0]

df = pd.concat([train_data, test_data], axis=0)

df[label_column] = df[label_column].str.strip()

df = df[df[label_column].isin(['accurate', 'inaccurate'])]

# Defining the output labels to the model
new_labels = {i:key for i, key in enumerate(df[label_column].value_counts().to_dict())}

# Replacing string to number in the label columns
df.replace({label_column:{value: key for key, value in new_labels.items()}}, inplace=True)


# Data Handler
# ---------------------------------------------------------------------------

preprocessing_setup = {
    'lower_case': True,
    'remove_emojis': True,
    'replace_emojis_by_text': False,
    'remove_stop_words': False,
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
    'remove_between_substrings': [('_x0','d_')]
}


data_handler = DataHandler(df=df, text_column=text_column, label_column=label_column, extra_columns=[id_column])

data_handler.preprocess(setup=preprocessing_setup)

# exploratory_data_analysis = ExploratoryDataAnalysis()
# exploratory_data_analysis.plot_text_size_distribution(data_handler.df, data_handler.get_text_column_name())
# exploratory_data_analysis.generate_word_cloud(data_handler.df, data_handler.get_text_column_name())

# data_handler.unsample()

# print(data_handler.get_top_words(100))
# print(data_handler.get_top_words_tfidf(100))

# train_data, test_data = data_handler.split_train_test_dataset()

train_data = data_handler.df[data_handler.df['is_training'] == True]
test_data = data_handler.df[data_handler.df['is_training'] == False]

print('train_data', train_data.shape)
print('test_data', test_data.shape)
print('total of data to train and test:', train_data.shape[0]+test_data.shape[0],'//', (train_data.shape[0]+test_data.shape[0])/2, 'per class')

# Language Model Handler
# --------------------------------------------------------------------------------

# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

model_name = 'bert-base-uncased' #'cardiffnlp/twitter-roberta-base-offensive' #'roberta-base' #'bert-base-uncased'

print('*** Model:', model_name)

MODEL_PARAMS = {
    "batch_size": 8,
    "learning_rate": 5e-5,
    "epochs": 3,
    "chunk_size": 510,
    "stride": 510,
    "minimal_chunk_length": 510,
    "pooling_strategy": "mean",
    'pretrained_model_name_or_path': model_name
}
model = BertClassifierWithPooling(**MODEL_PARAMS)