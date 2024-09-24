import emoji
import preprocessor as p
import re
from collections import Counter

import pandas as pd
# from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle


nltk.download('stopwords')
# Create a set of stop words
stop_words = set(stopwords.words('english'))

class DataHandler():
    def __init__(self, df, text_column, label_column, extra_columns, random_state=42):
        self.random_state = random_state
        self.df = df
        self.text_column = text_column
        self.processed_text_column = None
        self.label_column = label_column

        self.extra_columns = extra_columns

        if label_column:
            self.number_of_labels = len(df[label_column].value_counts())

        self.contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                     "can't": "cannot","can't've": "cannot have",
                     "'cause": "because","could've": "could have","couldn't": "could not",
                     "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                     "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                     "hasn't": "has not","haven't": "have not","he'd": "he would",
                     "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                     "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                     "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                     "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                     "it'd": "it would","it'd've": "it would have","it'll": "it will",
                     "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                     "mayn't": "may not","might've": "might have","mightn't": "might not", 
                     "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                     "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                     "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                     "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have","should've": "should have",
                     "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                     "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                     "there'd've": "there would have", "they'd": "they would",
                     "they'd've": "they would have","they'll": "they will",
                     "they'll've": "they will have", "they're": "they are","they've": "they have",
                     "to've": "to have","wasn't": "was not","we'd": "we would",
                     "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                     "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                     "what'll've": "what will have","what're": "what are", "what've": "what have",
                     "when've": "when have","where'd": "where did", "where've": "where have",
                     "who'll": "who will","who'll've": "who will have","who've": "who have",
                     "why've": "why have","will've": "will have","won't": "will not",
                     "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                     "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have","y'all're": "you all are",
                     "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                     "you'll": "you will","you'll've": "you will have", "you're": "you are",
                     "you've": "you have"}

    # Function for expanding contractions
    def __expand_contractions(self, text):
        """
        Expands contractions in the input text using a predefined dictionary of contractions.

        Args:
            text (str): The input text containing contractions to expand.

        Returns:
            str: The text with contractions expanded.
        """
        # Compile a regular expression to match contractions
        contractions_re = re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

        def replace(match):
            # Replace the matched contraction with its expanded form from the dictionary
            return self.contractions_dict[match.group(0)]
        
        return contractions_re.sub(replace, text)

    def __remove_emojis(self, text):
        """
        Removes emojis from the input text.

        Args:
            text (str): The input text from which to remove emojis.

        Returns:
            str: The text with emojis removed.
        """
        return emoji.replace_emoji(text, replace='')

    def __replace_emojis_by_text(self, text, language='en'):
        """
        Replaces emojis in the input text with their textual representation.

        Args:
            text (str): The input text containing emojis.
            language (str): The language for the emoji conversion. Default is 'en'.

        Returns:
            str: The text with emojis replaced by their textual representation.
        """
        return emoji.demojize(text, language=language)

    def __remove_terms_and_hashtags(self, text, hashtags_terms_list):
        """
        Removes specified terms and hashtags from the input text.

        Args:
            text (str): The input text from which to remove terms and hashtags.
            hashtags_terms_list (list): A list of terms and hashtags to remove.

        Returns:
            str: The text with specified terms and hashtags removed.
        """
        for word in hashtags_terms_list:
            # Replace each specified term or hashtag with an empty string
            text = text.replace(word, '')
        
        return text.strip()  # Return the cleaned text, stripped of leading/trailing whitespace

    
    def __remove_stop_words(self, sentence):
        # Split the sentence into individual words
        words = sentence.split()
        # Use a list comprehension to remove stop words
        filtered_words = [word for word in words if word not in stop_words]
        # Join the filtered words back into a sentence
        return ' '.join(filtered_words)
    
    def __remove_random_symbols(self, text, symbols_to_remove):
        # Construct the symbol pattern dynamically from the list of symbols
        symbol_pattern = r'\b(?:' + '|'.join(re.escape(symbol) for symbol in symbols_to_remove) + r')+\b'
        
        # Use re.sub to replace matched symbol patterns with an empty string
        text = re.sub(symbol_pattern, '', text)
        
        return text

    def __remove_money_values(self, text):
        # Define a regular expression pattern to match money values
        pattern = r'\d+\$'
        
        # Use re.sub to replace matched patterns with an empty string
        cleaned_string = re.sub(pattern, '', text)
        
        return cleaned_string
    
    def __remove_text_between_patterns(self, text, substring_1, substring_2):
        """
        Removes text between two specified substrings from the input text, including the substrings themselves.

        Args:
            text (str): The input text from which to remove content.
            substring_1 (str): The starting substring indicating the text to be removed.
            substring_2 (str): The ending substring indicating the text to be removed.

        Returns:
            str: The modified text with the specified content removed.
        """
        while True:
            # Find the starting index of the first substring
            start_index = text.find(substring_1)
            if start_index == -1:  # If start substring not found, break the loop
                break
            
            # Find the ending index of the second substring
            end_index = text.find(substring_2, start_index)
            if end_index == -1:  # If end substring not found after start, break the loop
                break
            
            # Remove text between start and end substrings, inclusive
            text = text[:start_index] + text[end_index + len(substring_2):]
        
        return text


    def preprocess_sentence(self, text, setup):
        """
        Preprocesses a given text based on specified settings.

        Args:
            text (str): The input text to preprocess.
            setup (dict): A configuration dictionary containing preprocessing settings.

        Returns:
            str: The processed text after applying the specified preprocessing steps.
        """
        
        # Convert text to lowercase if specified
        if setup['lower_case']:
            text = text.lower()

        # Remove emojis if specified
        if setup['remove_emojis']:
            text = self.__remove_emojis(text)
        
        # Replace emojis with text if specified and emojis are not removed
        if setup['replace_emojis_by_text'] and not setup['remove_emojis']:
            text = self.__replace_emojis_by_text(text)

        # Remove stop words if specified
        if setup['remove_stop_words']:
            text = self.__remove_stop_words(text)

        # Remove specified random symbols if any
        if setup['symbols_to_remove']:
            text = self.__remove_random_symbols(text, setup['symbols_to_remove'])        

        # Remove numbers from the text if specified
        if setup['remove_numbers']:
            text = text.replace('\d+', '')  # Removing numbers

        # Expand contractions if specified
        if setup['expand_contractions']:
            text = self.__expand_contractions(text)

        # Remove text between specified substrings if specified
        if setup['remove_between_substrings']:
            for substring_1, substring_2 in setup['remove_between_substrings']:
                text = self.__remove_text_between_patterns(text, substring_1, substring_2)

        # Remove specific terms and hashtags if specified
        if setup['remove_terms_hashtags']:
            text = self.__remove_terms_and_hashtags(text, setup['remove_terms_hashtags'])

        # Remove hashtags from the text if specified
        if setup['remove_hashtags']:
            hashtag_pattern = r'#\w+'
            text = re.sub(hashtag_pattern, '', text)

        # Remove user mentions from the text if specified
        if setup['remove_users']:
            username_pattern = r'@\w+'
            text = re.sub(username_pattern, '', text)

        # Remove URLs from the text if specified
        if setup['remove_urls']:
            url_pattern = r'http\S+'
            text = re.sub(url_pattern, '', text)

            # Also remove the placeholder <url> if specified
            url_pattern = r'<url>'
            text = re.sub(url_pattern, '', text)

        # Remove monetary values from the text if specified
        if setup['remove_money_values']:
            text = self.__remove_money_values(text)

        # Lemmatize words in the text if specified
        if setup['lemmatize']:
            wnl = WordNetLemmatizer()
            list2 = nltk.word_tokenize(text)
            text = ' '.join([wnl.lemmatize(words) for words in list2])
        
        return text


    def get_text_column_name(self):
        """
        Retrieves the appropriate text column name based on the processing status.

        Returns:
            str: The name of the text column, either processed or original.
        """
        # Check if a processed text column exists and return its name
        if self.processed_text_column:
            return self.processed_text_column
        else:
            # If no processed column, return the original text column name
            return self.text_column


    def get_top_words(self, n=100):

        temp_text_column = self.get_text_column_name()

        # Combine all tweets into a single string
        all_tweets = " ".join(self.df[temp_text_column])

        # Tokenize the text
        words = word_tokenize(all_tweets)

        # Remove stopwords and non-alphabetic words
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        # Calculate word frequencies
        word_freq = Counter(words)

        # Get the top n words
        top_words_and_count = word_freq.most_common(n)
        top_words = [word for word, counter in top_words_and_count]
        counters = [counter for word, counter in top_words_and_count]

        return {'words':top_words, 'counters':counters}


    def get_top_words_tfidf(self, n):

        temp_text_column = self.get_text_column_name()

        # Create a TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features=n, stop_words='english')

        # Fit and transform the text data
        tfidf_matrix = tfidf_vectorizer.fit_transform(self.df[temp_text_column])

        # Get feature names (words)
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Sum the TF-IDF scores for each word across all tweets
        word_scores = tfidf_matrix.sum(axis=0)

        # Sort words by their TF-IDF scores
        top_indices = word_scores.argsort()[0, ::-1][:n]

        # Get the top n words and their TF-IDF scores
        top_words = [(feature_names[i], word_scores[0, i]) for i in top_indices]

        return top_words[0][0][0]

    def preprocess(self, setup):
        """
        Preprocesses the text data in the dataframe by cleaning and applying specified transformations.

        Args:
            setup (dict): A configuration dictionary containing preprocessing settings.

        Returns:
            pd.DataFrame: The dataframe with processed text column added.
        """
        # Remove rows where the text column is NaN
        self.df.dropna(subset=[self.text_column], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Define the name for the processed text column
        self.processed_text_column = 'processed_' + self.text_column
        
        # Apply the preprocessing function to each row in the text column
        self.df[self.processed_text_column] = self.df.apply(lambda x: self.preprocess_sentence(x[self.text_column], setup), axis=1)

        # If the setup specifies to remove non-ASCII characters, apply the regex substitution
        if setup['remove_non_text_characters']:
            pattern = re.compile(r'[^\x00-\x7F]+')
            self.df[self.processed_text_column] = self.df.apply(lambda x: pattern.sub('', x[self.processed_text_column]), axis=1)

        return self.df


    def unsample(self):
        """
        Downsamples the dataframe to have an equal number of samples for each class label.

        Returns:
            pd.DataFrame: The downsampled dataframe with equal class distribution.
        """
        # Define columns to be used from the dataframe
        columns = [self.text_column, self.label_column]+self.extra_columns

        # If a processed text column exists, add it to the list of columns
        if self.processed_text_column:
            columns.append(self.processed_text_column)

        # Group the dataframe by the label column
        processed_df_grouped = self.df[columns].groupby(self.label_column)

        # Generate a list of dataframes where each dataframe is a sample of the minimum group size
        frames_of_groups = [x.sample(processed_df_grouped.size().min(), random_state=self.random_state) 
                            for y, x in processed_df_grouped]

        # Concatenate the sampled dataframes to form a new dataframe
        self.df = pd.concat(frames_of_groups)

        # Shuffle the new dataframe
        self.df = self.df.sample(frac=1, random_state=self.random_state)
        
        # Reset the index of the new dataframe
        self.df.reset_index(drop=True, inplace=True)

        return self.df

    def split_train_test_dataset(self, test_percentage=0.2, random_state=None):

        if random_state is None:
            random_state = self.random_state

        # Shuffle the dataframe to ensure randomness
        df = shuffle(self.df, random_state=random_state).reset_index(drop=True)
        
        # Get the minimum number of samples in any class
        min_class_size = df[self.label_column].value_counts().min()
        
        # Calculate the number of samples to include in the test set for each class
        n_test_samples = int(min_class_size * test_percentage)
        
        test_indices = []
        train_indices = []
        
        # Loop through each class and select samples
        for class_label in df[self.label_column].unique():
            class_indices = df[df[self.label_column] == class_label].index.tolist() # getting index of the rows per class

            # Select test samples
            test_indices.extend(class_indices[:n_test_samples])
            # Select train samples
            train_indices.extend(class_indices[n_test_samples:])
        
        # Create the test and train dataframes
        test_df = df.loc[test_indices].reset_index(drop=True)
        train_df = df.loc[train_indices].reset_index(drop=True)
        
        return train_df, test_df

    # def split_train_test_dataset(self, train_size=0.8):
    #     # Training dataset
    #     train_data = self.df[[self.text_column, self.get_text_column_name(), self.label_column]].sample(frac=train_size, random_state=self.random_state)

    #     # Testing dataset
    #     test_data = self.df[[self.text_column, self.get_text_column_name(), self.label_column]].drop(train_data.index)

    #     return train_data, test_data

# ----------------------------------------------------------------------------------------------------

# preprocessing_setup = {
#     'lower_case': True,
#     'remove_emojis': False,
#     'remove_stop_words': True,
#     'remove_numbers': False,
#     'remove_users': True,
#     'remove_urls': True,
#     'remove_non_text_characters': True,
#     'lemmatize': False
# }


# data_handler = DataHandler(df=df, text_column=original_text_column, label_column=label_column)

# data_handler.preprocess(setup=preprocessing_setup)

# data_handler.unsample()

# # print(data_handler.get_top_words(100))
# # print(data_handler.get_top_words_tfidf(100))

# train_data, test_data = data_handler.split_train_test_dataset()
# # # data_handler.df