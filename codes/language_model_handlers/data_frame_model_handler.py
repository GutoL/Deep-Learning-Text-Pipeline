
import os 
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import gc

class DataFrameModelHandler():
    def __init__(self, model_name, trainel_model_path=None, negative_class_prefix='no') -> None:
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(trainel_model_path+model_name)
        # Load Model
        # self.model = torch.load(path+model_name+'/'+model_name+'.pth')
        self.model = AutoModelForSequenceClassification.from_pretrained(trainel_model_path+model_name)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.pipeline = pipeline('text-classification', model=self.model,
                                tokenizer=self.tokenizer, device=self.device)

        self.negative_class_suffixe = negative_class_prefix
        
    # def __create_classification_column(self, df, class_name, classification_column='classification'):
    #     # Add a new column 'classification' with 0 if 'non-class' has lower probability, else 1
    #     df[classification_column] = df.apply(lambda row: 0 if row[self.negative_class_suffixe+' '+class_name] > row[class_name] else 1, axis=1)
    #     return df
    
    def __data_loader(self, df, column, text_size_limit):
        
        column_index = list(df.columns).index(column)
        
        for row in df.values:
            text = row[column_index] # Getting the text of the tweet

            if len(text.split()) > text_size_limit:
                yield text.split()[:text_size_limit]
            else:
                yield text
            
                
    def classify_dataframe(self, df, original_text_column, text_column, extra_columns_to_save, results_file_name, batch_size, batch_size_to_save, 
                           text_size_limit=512, threshold=None, include_text=True, id_column='id', sep_to_save='|', include_threshold_based_classification=False):
        
        if os.path.isfile(results_file_name): # if the results file exists
            df_results = pd.read_csv(results_file_name)
            df = df.tail(df.shape[0] - df_results.shape[0])

        else:
            df_results = pd.DataFrame(columns=list(self.model.config.id2label.values()))
        

        if df.shape[0] > 0:
            i = 0

            for prediction in tqdm(self.pipeline(self.__data_loader(df, text_column, text_size_limit), batch_size=batch_size, return_all_scores=True), total=df.shape[0]):
                
                result = {column:[df.iloc[i][column]] for column in extra_columns_to_save}

                if include_text:
                    result[original_text_column] = [df.iloc[i][original_text_column]]
                
                highest_value = 0

                if threshold:
                    result['classification_threshold'] = None
                
                for classes_pred in prediction:
                    result[classes_pred['label']] = classes_pred['score']

                    if include_threshold_based_classification:
                        if classes_pred['score'] > highest_value:
                            highest_value = classes_pred['score']
                            result['classification'] = classes_pred['label']
                    
                            # Check whether the positive class is greater than a value or not, if yes, set it to 1, otherwise set it to 0
                            if threshold and (highest_value >= threshold):
                                result['classification_threshold'] = classes_pred['label']

                result_df = pd.DataFrame.from_dict(result)

                if id_column:
                    result_df[id_column] = result_df[id_column].apply(lambda x: str(x))

                df_results = pd.concat([df_results, result_df])

                if i % batch_size_to_save == 0 and i > 0:
                    # self.__create_classification_column(df_results, class_name).to_csv(result_file_name, index=False)
                    df_results.to_csv(results_file_name, index=False, sep=sep_to_save)
                                    
                i += 1

            # self.__create_classification_column(df_results, class_name).to_csv(result_file_name, index=False)#['label_match'].value_counts()
            df_results.to_csv(results_file_name, index=False, sep=sep_to_save)
        
        print('Finished:', results_file_name)
            