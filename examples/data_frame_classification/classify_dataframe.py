import sys
sys.path.append("../../../GitHub/Deep-Learning-Text-Pipeline/")

from codes.language_model_handlers.data_frame_model_handler import DataFrameModelHandler
from codes.data_handler import DataHandler

import pandas as pd
import torch 
import gc
from glob import glob
import os
import sys


def split_and_save_dataframe(df, path, chunk_size, sep='|'):
    
    # Ensure the output directory exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Calculate the number of chunks
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
    
    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min(start_index + chunk_size, len(df))
        
        chunk_df = df.iloc[start_index:end_index]
        
        # Construct file name for each chunk
        file_name = f"chunk_{i+1}.csv"
        file_path = os.path.join(path, file_name)
        
        # Save the chunk to a CSV file
        chunk_df.to_csv(file_path, index=False, sep=sep)
        
        print(f"Saved chunk {i+1}/{num_chunks} to {file_path}")

def classify_dataframe(model_name, temp_files_path, results_path, text_column, classification_type, 
                       tweets_type, hate_speech_type, year, preprocessing_setup, sep='|'):
    
    threshold_text_size = 512

    model_name = model_name.replace('/', '-')

    for filename in glob(temp_files_path+'*.csv'):
        
        # Cleaning cache from GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        print(filename)
        df = pd.read_csv(filename, sep=sep, low_memory=False, lineterminator='\n')

        df[text_column] = df[text_column].str.replace('\\','')
        df[text_column] = df[text_column].apply(lambda x: x[:threshold_text_size] if len(x) > threshold_text_size else x)

        data_handler = DataHandler(df=df, text_column=text_column, label_column=None, extra_columns=[])

        data_handler.preprocess(setup=preprocessing_setup)

        dataframe_classifier = DataFrameModelHandler(model_name=model_name, 
                                                     trainel_model_path='../'+classification_type+'/saved_models/'+hate_speech_type+'/', 
                                                     negative_class_prefix='non')

        results_file_name = results_path+model_name+'_'+classification_type+'_'+year+'_'+tweets_type+'_'+os.path.basename(filename)
        
        dataframe_classifier.classify_dataframe(df=data_handler.df, 
                                                original_text_column=text_column, 
                                                text_column=data_handler.get_text_column_name(),
                                                extra_columns_to_save=['id'],
                                                results_file_name=results_file_name,
                                                batch_size=64, 
                                                batch_size_to_save=1000, 
                                                threshold=0.95,
                                                include_text=True,
                                                id_column='id',
                                                sep_to_save=sep) # '''

# --------------------------------------------------------------------------------------------------

year = sys.argv[1] #'2022'

hate_speech_type = sys.argv[2]

model_name = sys.argv[3] # 'FacebookAI/roberta-base' #'bert-base-uncased'

if year == '2020':
    text_column = 'data_text'
else:
    text_column = 'text'

id_column = 'id'    

classification_type = 'multi_datasets'

threshold_text_size = 512

results_file_name = year+'*.csv'
results_path = 'results/'+hate_speech_type+'/'+year+'/'


fp = open('../euros_hashtags_terms.txt', 'r')
terms_hashtags_euros = [line.replace('\n', '') for line in fp.readlines()]
fp.close()

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
    'remove_hashtags': True,
    'remove_money_values': False,
    'remove_apostrophe_contractions': False,
    'symbols_to_remove': ['*', '@', '<url>'],
    'remove_between_substrings': None, # [('_x0','d_')]
    'remove_terms_hashtags': terms_hashtags_euros+['\n']+['euros', 'the euros', 'euro']
}


for tweets_type in ['original_tweets', 'replies']: # 'retweets'

    data_path = '../datasets/Euros/FINAL/'+tweets_type+'/'    

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    df = pd.DataFrame()

    for file_name in glob(data_path+results_file_name):
        print('Classifying:', file_name)

        temp_df = pd.read_csv(file_name, sep=',', low_memory=False)

        df = pd.concat([df, temp_df], axis=0)
        
    df.drop_duplicates(subset=[id_column], inplace=True)

    print('Classifying', df.shape[0], 'texts...')

    temp_files_path = results_path+tweets_type+'_temp/'

    split_and_save_dataframe(df=df, path=temp_files_path, chunk_size=100000, sep='|')

    classify_dataframe(model_name, temp_files_path, results_path, text_column, classification_type, 
                       tweets_type, hate_speech_type, year, preprocessing_setup, sep='|')
    
    