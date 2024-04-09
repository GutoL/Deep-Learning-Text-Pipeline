import sys
sys.path.append("../../../GitHub/Deep-Learning-Text-Pipeline/")

from codes.language_model_handlers.data_frame_model_handler import DataFrameModelHandler
from codes.data_handler import DataHandler

import pandas as pd
import torch 
import gc
from glob import glob
import os

year = '2012'

model_name = 'FacebookAI/roberta-base' #'bert-base-uncased'
text_column = 'text'
classification_type = 'multiclass'
dataset_type = 'sexism'

results_file_name = year+'*.csv'
results_path = 'results/'+dataset_type+'/'+year+'/'

if year in ['2008','2012', '2016', '2020']:
    gender = 'men'
else:
    gender = 'women'


for tweets_type in ['original_tweets', 'replies', 'retweets']:

    data_path = '../datasets/Euros/'+gender+'/'+tweets_type+'/'    

    print(data_path+results_file_name)
    print(glob(data_path+results_file_name))

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for file_name in glob(data_path+results_file_name):
        print('Classifying:', file_name)
        
        df = pd.read_csv(file_name)

        year_file = file_name.split('\\')[-1]

        model_name = model_name.replace('/', '-')

        # Cleaning cache from GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        df[text_column] = df[text_column].str.replace('\\','')

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


        data_handler = DataHandler(df=df, text_column=text_column, label_column=None)

        data_handler.preprocess(setup=preprocessing_setup)

        dataframe_classifier = DataFrameModelHandler(model_name=model_name, path='../'+classification_type+'/saved_models/'+dataset_type+'/', negative_class_prefix='non')

        dataframe_classifier.classify_dataframe(df=data_handler.df, 
                                                original_text_column=text_column, 
                                                text_column=data_handler.get_text_column_name(),
                                                extra_columns_to_save=['id'],
                                                result_file_name=results_path+model_name+'_'+classification_type+'_'+tweets_type+'_'+year_file,
                                                batch_size=32, 
                                                batch_size_to_save=1000, 
                                                threshold=0.95,
                                                include_text=False) # '''