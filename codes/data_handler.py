import emoji
import preprocessor as p
import re
from collections import Counter

import pandas as pd
from sklearn.utils import shuffle
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
        self.number_of_labels = len(df[label_column].value_counts())


    def __demojize_text(self, text):
        return emoji.demojize(text)

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

    def __preprocess_sentence(self, text, setup):

        if setup['lower_case']:
            text = text.lower()

        if setup['remove_emojis']:
            text = self.__demojize_text(text)

        if setup['remove_stop_words']:
            text = self.__remove_stop_words(text)

        if setup['remove_numbers']:
            text = text.replace('\d+', '') # Removing numbers

        # text = p.clean(text) #heavy cleaning

        new_text = []
        for t in text.split(" "):
            # t = remove_words_with_euro(t)

            if setup['remove_users']:
                t = '' if t.startswith('@') and len(t) > 1 else t
                # t = '@user' if t.startswith('@') and len(t) > 1 else t
            if setup['remove_urls']:
                t = '' if t.startswith('http') else t
                # t = 'http' if t.startswith('http') else t

            new_text.append(t)

        new_text = " ".join(new_text)

        if setup['lemmatize']:
            wnl = WordNetLemmatizer()
            list2 = nltk.word_tokenize(new_text)
            new_text = ' '.join([wnl.lemmatize(words) for words in list2])

        return new_text

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

        self.df = shuffle(self.df, random_state=self.random_state)
        self.df.reset_index(drop=True, inplace=True)

        return self.df

    def split_train_test_dataset(self, train_size=0.8):
        # Training dataset
        train_data = self.df[[self.get_text_column_name(), self.label_column]].sample(frac=train_size, random_state=self.random_state)

        # Testing dataset
        test_data = self.df[[self.get_text_column_name(), self.label_column]].drop(train_data.index)

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