
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
        """
        Generator function that yields processed text from a specified column of a dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing the text data.
            column (str): The name of the column from which to extract text.
            text_size_limit (int): The maximum number of words to yield for each text.

        Yields:
            list or str: A list of words if the text exceeds the size limit, otherwise the original text.
        """
        
        # Get the index of the specified column in the dataframe
        column_index = list(df.columns).index(column)
        
        # Iterate over each row in the dataframe
        for row in df.values:
            text = row[column_index]  # Extract the text from the current row

            # Check if the text exceeds the specified word limit
            if len(text.split()) > text_size_limit:
                # Yield the first 'text_size_limit' words as a list
                yield text.split()[:text_size_limit]
            else:
                # Yield the original text if it doesn't exceed the limit
                yield text
            
                
    def classify_dataframe(self, df, original_text_column, text_column, extra_columns_to_save, results_file_name, batch_size, batch_size_to_save, 
                       text_size_limit=512, threshold=None, include_text=True, id_column='id', sep_to_save='|', include_threshold_based_classification=False):
        """
        Classifies the input dataframe using a pipeline and saves the results to a CSV file.

        Args:
            df (pd.DataFrame): The input dataframe containing text data to classify.
            original_text_column (str): The name of the column with the original text.
            text_column (str): The name of the column with the processed text.
            extra_columns_to_save (list): List of additional columns to include in the results.
            results_file_name (str): The filename to save the classification results.
            batch_size (int): The number of samples to process in each batch.
            batch_size_to_save (int): The number of samples to save after processing.
            text_size_limit (int): The maximum length of text to consider for classification. Default is 512.
            threshold (float, optional): A threshold for classifying based on score. Default is None.
            include_text (bool): Whether to include the original text in the results. Default is True.
            id_column (str): The name of the ID column. Default is 'id'.
            sep_to_save (str): The separator to use when saving the results CSV. Default is '|'.
            include_threshold_based_classification (bool): Whether to include classification based on threshold. Default is False.

        Returns:
            None: Saves the classification results to a CSV file.
        """
        
        # Check if the results file already exists
        if os.path.isfile(results_file_name):
            # Load existing results
            df_results = pd.read_csv(results_file_name)
            # Filter out already processed rows from the original dataframe
            df = df.tail(df.shape[0] - df_results.shape[0])
        else:
            # Initialize an empty dataframe for results with column names from the model
            df_results = pd.DataFrame(columns=list(self.model.config.id2label.values()))
        
        # Process the dataframe if it contains rows
        if df.shape[0] > 0:
            i = 0

            # Iterate over predictions from the pipeline
            for prediction in tqdm(self.pipeline(self.__data_loader(df, text_column, text_size_limit), batch_size=batch_size, return_all_scores=True), total=df.shape[0]):
                # Create a dictionary to hold results for the current row
                result = {column: [df.iloc[i][column]] for column in extra_columns_to_save}

                if include_text:
                    result[original_text_column] = [df.iloc[i][original_text_column]]
                
                highest_value = 0

                # Initialize classification threshold if specified
                if threshold:
                    result['classification_threshold'] = None
                
                # Process each class prediction
                for classes_pred in prediction:
                    result[classes_pred['label']] = classes_pred['score']

                    if include_threshold_based_classification:
                        if classes_pred['score'] > highest_value:
                            highest_value = classes_pred['score']
                            result['classification'] = classes_pred['label']
                            
                            # Set classification threshold if score exceeds specified threshold
                            if threshold and (highest_value >= threshold):
                                result['classification_threshold'] = classes_pred['label']

                # Convert the result dictionary to a DataFrame
                result_df = pd.DataFrame.from_dict(result)

                # Ensure ID column values are strings
                if id_column:
                    result_df[id_column] = result_df[id_column].apply(lambda x: str(x))

                # Concatenate the current results with the overall results
                df_results = pd.concat([df_results, result_df])

                # Save results to CSV periodically based on the batch size
                if i % batch_size_to_save == 0 and i > 0:
                    df_results.to_csv(results_file_name, index=False, sep=sep_to_save)
                                    
                i += 1

            # Final save of results to CSV
            df_results.to_csv(results_file_name, index=False, sep=sep_to_save)
        
        print('Finished:', results_file_name)
