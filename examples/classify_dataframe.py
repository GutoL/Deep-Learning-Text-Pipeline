import sys
sys.path.append("../../GitHub/Deep-Learning-Text-Pipeline/")

from codes.language_model_handlers.data_frame_model_handler import DataFrameModelHandler
from codes.data_handler import DataHandler

import pandas as pd
import torch 
import gc

df = pd.read_csv('datasets/Euros/men/euro_2012_no_rt.csv')

model_name = 'bert-base-uncased' #'bert-base-uncased'
text_column = 'text'
dataset_type = 'homophobic'
# Cleaning cache from GPU memory
torch.cuda.empty_cache()
gc.collect()

df[text_column] = df[text_column].str.replace('\\','')

preprocessing_setup = {
    'lower_case': False,
    'remove_emojis': False,
    'remove_hashtags': False,
    'remove_stop_words': True,
    'remove_numbers': True,
    'remove_users': True,
    'remove_urls': True,
    'remove_non_text_characters': True,
    'lemmatize': False,
    'expand_contractions': False
}


data_handler = DataHandler(df=df, text_column=text_column, label_column=None)

# data_handler.preprocess(setup=preprocessing_setup)

dataframe_classifier = DataFrameModelHandler(model_name=model_name, path='saved_models/'+dataset_type+'/', negative_class_suffixe='no')

dataframe_classifier.classify_dataframe(df=data_handler.df, original_text_column=text_column, text_column=data_handler.get_text_column_name(),
                                        extra_columns_to_save=['id'],
                                        result_file_name='results/'+dataset_type+'.csv', batch_size=32, 
                                        batch_size_to_save=1000, class_name=dataset_type, threshold=0.85)