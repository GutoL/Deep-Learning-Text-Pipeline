from codes.features_extraction.LIWC.my_liwc import CustomLiwc

import pandas as pd
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

class FeaturesExtractor():
    
    def __init__(self, features_columns=None) -> None:
        
        if features_columns is None:
            self.features_columns = ['word_count', 'list_word_count', 'list_word_proportion', 'subjectivity', 'negative', 'neutral', 'positive'] # 'polarity'

        else:
            self.features_columns = features_columns
    
    def get_features(self, df, extra_columns=None):

        if extra_columns:
            return df[self.features_columns+extra_columns]
        else: 
            return df[self.features_columns]
            

    def analyze_word_presence(self, df, column_name, word_list):
        # Function to count words in a string
        def count_words(text):
            return len(text.split())
        
        # Function to count how many words from the list are present in the string
        def count_words_from_list(text, word_list):
            words = text.split()
            return sum(1 for word in words if word in word_list)
        
        # Apply the count_words function to each row in the specified column
        df['word_count'] = df[column_name].apply(count_words)
        
        # Apply the count_words_from_list function to each row in the specified column
        df['list_word_count'] = df[column_name].apply(lambda text: count_words_from_list(text, word_list))
        
        # Calculate the proportion of words from the list in the row content
        df['list_word_proportion'] = df['list_word_count'] / df['word_count']
        
        return df

    def analyze_text(self, text):
        # Create a TextBlob object
        blob = TextBlob(text)
        sia = SentimentIntensityAnalyzer()
        
        # Get the subjectivity score
        subjectivity = blob.sentiment.subjectivity
        
        # # Get the emotion score (polarity)
        # polarity = blob.sentiment.polarity

        # https://medium.com/@umarsmuhammed/how-to-perform-sentiment-analysis-using-python-step-by-step-tutorial-with-code-snippets-4ac3e9747fff
        polarity = sia.polarity_scores(text)
        
        
        return subjectivity, polarity['neg'], polarity['neu'], polarity['pos'] #, polarity['compound']

    def analyze_dataframe_polarity_subjectivity(self, df, column_name):
        # Apply the analyze_text function to each entry in the specified column
        analysis_results = df[column_name].apply(self.analyze_text)

        # Split the results into two separate lists
        subjectivity_scores = analysis_results.apply(lambda x: x[0]) # subjectivity
        # polarity_scores = = analysis_results.apply(lambda x: x[1]) # Polarity
        negative_scores = analysis_results.apply(lambda x: x[1]) # Negative
        neutral_scores = analysis_results.apply(lambda x: x[2]) # Neutral
        positive_scores = analysis_results.apply(lambda x: x[3]) # Positive
        # compound_scores = analysis_results.apply(lambda x: x[4]) # Compound
        
        # Append the new columns to the DataFrame
        df['subjectivity'] = subjectivity_scores
        df['negative'] = negative_scores
        df['neutral'] = neutral_scores
        df['positive'] = positive_scores
        # df['compound'] = compound_scores
        
        return df
    
    def extract_all_features(self, df, text_column, terms, liwc_dict_path=None):
        """
        Extracts various features from the dataframe.

        Args:
            df (DataFrame): The input dataframe.
            text_column (str): The column name containing text data.
            terms (list): List of terms to analyze for presence.
            liwc_dict_path (str, optional): Path to the LIWC dictionary for analysis. Defaults to None.

        Returns:
            DataFrame: The dataframe with extracted features.
        """
        
        # Analyze the polarity and subjectivity of the text
        df = self.analyze_dataframe_polarity_subjectivity(df, text_column)
        
        # Analyze the presence of specific terms in the text
        df = self.analyze_word_presence(df, text_column, terms)

        # If a LIWC dictionary path is provided, process the dataframe using LIWC
        if liwc_dict_path:
            my_liwc = CustomLiwc(liwc_dict_path)
            df = my_liwc.process_data_frame(df, text_column)

        return df


            