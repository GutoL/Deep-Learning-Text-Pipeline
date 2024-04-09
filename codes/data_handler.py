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

nltk.download('stopwords')
# Create a set of stop words
stop_words = set(stopwords.words('english'))

class DataHandler():
    def __init__(self, df, text_column, label_column, random_state=42):
        self.random_state = random_state
        self.df = df
        self.text_column = text_column
        self.processed_text_column = None
        self.label_column = label_column

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
        contractions_re=re.compile('(%s)' % '|'.join(self.contractions_dict.keys()))

        def replace(match):
            return self.contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    def __remove_emojis(self, text):
        return emoji.replace_emoji(text, replace='')
    
    def __replace_emojis_by_text(self, text, language='en'):
        return emoji.demojize(text, language=language)

    def __remove_words_with_euro(self, input_string):
        # Define a regular expression pattern to match words containing 'euro'
        pattern = r'\b\w*#?euro\w*\b'
        # Use re.sub to replace matching words with an empty string
        result = re.sub(pattern, '', input_string)

        return result
    
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
        while True:
            start_index = text.find(substring_1)
            if start_index == -1:  # If start substring not found, break the loop
                break
            
            end_index = text.find(substring_2, start_index)
            if end_index == -1:  # If end substring not found after start, break the loop
                break
            
            # Remove text between start and end substrings, inclusive
            text = text[:start_index] + text[end_index + len(substring_2):]
    
        return text

    def __preprocess_sentence(self, text, setup):

        if setup['lower_case']:
            text = text.lower()

        if setup['remove_emojis']:
            text = self.__remove_emojis(text)
        
        if setup['replace_by_text'] == True and setup['remove_emojis'] == False:
            text = self.__replace_emojis_by_text(text)

        if setup['remove_stop_words']:
            text = self.__remove_stop_words(text)

        if setup['symbols_to_remove']:
            text = self.__remove_random_symbols(text, setup['symbols_to_remove'])        

        if setup['remove_numbers']:
            text = text.replace('\d+', '') # Removing numbers

        if setup['expand_contractions']:
            text = self.__expand_contractions(text)

        # text = p.clean(text) #heavy cleaning

        if setup['remove_between_substrings']:
            for substring_1, substring_2 in setup['remove_between_substrings']:
                text = self.__remove_text_between_patterns(text, substring_1, substring_2)

        if setup['remove_hashtags']:
            hashtag_pattern = r'#\w+'
            text = re.sub(hashtag_pattern, '', text)

        if setup['remove_users']:
            username_pattern = r'@\w+'
            text = re.sub(username_pattern, '', text)

        if setup['remove_urls']:
            url_pattern = r'http\S+'
            text = re.sub(url_pattern, '', text)

            url_pattern = r'<url>'
            text = re.sub(url_pattern, '', text)

        if setup['remove_money_values']:
            text = self.__remove_money_values(text)

        if setup['lemmatize']:
            wnl = WordNetLemmatizer()
            list2 = nltk.word_tokenize(text)
            text = ' '.join([wnl.lemmatize(words) for words in list2])
        
        return text

    def get_text_column_name(self):
        if self.processed_text_column:
            return self.processed_text_column
        else:
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

        self.df.dropna(subset=[self.text_column], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.processed_text_column = 'processed_'+self.text_column
        self.df[self.processed_text_column] = self.df.apply(lambda x: self.__preprocess_sentence(x[self.text_column], setup), axis=1)

        if setup['remove_non_text_characters']:
            pattern = re.compile(r'[^\x00-\x7F]+')
            self.df[self.processed_text_column] = self.df.apply(lambda x: pattern.sub('', x[self.processed_text_column]), axis=1)

        return self.df

    def unsample(self):

        # temp_text_column = self.get_text_column_name()
        # columns = [temp_text_column, self.label_column]

        columns = [self.text_column, self.processed_text_column, self.label_column]

        processed_df_grouped = self.df[columns].groupby(self.label_column)
        processed_df_grouped.groups.values()

        frames_of_groups = [x.sample(processed_df_grouped.size().min(), random_state=self.random_state) for y, x in processed_df_grouped]
        self.df = pd.concat(frames_of_groups)

        # self.df = shuffle(self.df, random_state=self.random_state)
        self.df = self.df.sample(frac=1, random_state=self.random_state)
        
        self.df.reset_index(drop=True, inplace=True)

        return self.df

    def split_train_test_dataset(self, train_size=0.8):
        # Training dataset
        train_data = self.df[[self.text_column, self.get_text_column_name(), self.label_column]].sample(frac=train_size, random_state=self.random_state)

        # Testing dataset
        test_data = self.df[[self.text_column, self.get_text_column_name(), self.label_column]].drop(train_data.index)

        return train_data, test_data

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