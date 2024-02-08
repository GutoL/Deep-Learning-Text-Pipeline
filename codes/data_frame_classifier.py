import pandas as pd
from transformers import pipeline 
import os
from tqdm import tqdm
import numpy as np

from language_model_handler import LanguageModelHandler


class DataFrameClassifier(LanguageModelHandler):
    
    def __create_classification_column(self, df, classification_column='classification'):
        # Add a new column 'classification' with 0 if 'non-sexist' has higher probability, else 1
        df[classification_column] = df.apply(lambda row: 0 if row['non-'+self.dataset_type] > row[self.dataset_type] else 1, axis=1)
        return df

    def __data_loader(self, dataframe, column=1):
        for row in dataframe.values:
            text = row[column] # Getting the text of the tweet

            if len(text.split()) > self.text_size_limit:
                yield text.split()[:self.text_size_limit]
            else:
                yield text

    def classify_unlabaled_datasets(self, dataset_name_file, result_file_name, batch_size_to_save, column_index=1):

        if self.pipeline is None:
            self.pipeline = pipeline('text-classification', model=self.model,
                                     tokenizer=self.tokenizer, device=self.device)

        df = pd.read_csv(dataset_name_file)#.head(4000)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        if os.path.isfile(result_file_name): # if the results file exists
            df_results = pd.read_csv(result_file_name)
            df = df.tail(df.shape[0] - df_results.shape[0])
        else:
            df_results = pd.DataFrame(columns=list(self.model.config.id2label.values()))

        i = 0

        for prediction in tqdm(self.pipeline(self.__data_loader(df, column_index), batch_size=32, return_all_scores=True), total=df.shape[0]):
            result = {
                self.text_column: [df.iloc[i]['text']],
                prediction[0]['label']: [prediction[0]['score']],
                prediction[1]['label']: [prediction[1]['score']]
            }

            df_results = pd.concat([df_results, pd.DataFrame.from_dict(result)])

            if i % batch_size_to_save == 0 and i > 0:
                self.__create_classification_column(df_results, self.dataset_type).to_csv(result_file_name, index=False)
            i += 1

        self.__create_classification_column(df_results, self.dataset_type).to_csv(result_file_name, index=False)#['label_match'].value_counts()

    def predict_probability(self, texts_array):

        if self.pipeline is None:
            self.pipeline = pipeline('text-classification', model=self.model,
                                     tokenizer=self.tokenizer, device=self.device)

        all_results = []

        for predictions in tqdm(self.pipeline(self.__data_loader(pd.DataFrame(texts_array), column=0),
                                              batch_size=32, return_all_scores=True),
                                total=len(texts_array)):
        #for predictions in [{'label': 'non-racism', 'score': 0.44055721163749695}, {'label': 'racism', 'score': 0.5594428181648254}]:
            all_results.append([prediction['score'] for prediction in predictions])

        return np.array(all_results)

    def zero_shot_classification(self, sentence, labels, model_name=None):
        if self.zero_shot_pipeline is None:

            if model_name is None:
                model_name = self.model_name

            self.zero_shot_pipeline = pipeline("zero-shot-classification", model=model_name, device=self.device)

        return self.zero_shot_pipeline(sentence, labels)

    def zero_shot_classification_dataframe(self, dataframe, labels, model_name=None, results_file_name=None,
                                           batch_size=1000, column_index=1):

        if results_file_name and os.path.isfile(results_file_name):
            results_df = pd.read_csv(results_file_name)
            dataframe = dataframe.tail(dataframe.shape[0]-results_df.shape[0]) # getting the last rows that were not collected yet

        else:
            results_df = pd.DataFrame(columns=['text']+labels)

        print('Classifying', dataframe.shape[0], 'tweets')

        if self.zero_shot_pipeline is None:
            if model_name is None:
                model_name = self.model_name
            self.zero_shot_pipeline = pipeline("zero-shot-classification", model=model_name, device=self.device)

        i = results_df.shape[0]

        for prediction in tqdm(self.zero_shot_pipeline(self.__data_loader(dataframe, column_index), labels,
                                                       batch_size=32, return_all_scores=True),
                                                       total=dataframe.shape[0]):
            pred = {} # {'text': prediction['sequence']}

            for j in range(len(prediction['labels'])):
                pred[prediction['labels'][j]] = prediction['scores'][j]

            results_df = pd.concat([results_df, pd.DataFrame([pred])], ignore_index=True)

            if i % batch_size == 0 and results_file_name is not None:
                results_df.to_csv(results_file_name, index=False)

            i += 1

        results_df['text'] = dataframe.iloc[:, column_index]

        if results_file_name:
            results_df.to_csv(results_file_name, index=False)

        return results_df