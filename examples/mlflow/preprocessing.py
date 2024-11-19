import sys
sys.path.append("/home/guto/Documents/GitHub/Deep-Learning-Text-Pipeline/")

from codes.data_handler import DataHandler

def preprocess(df, text_column, label_column, id_column=None, random_state=42):
    # Data Handler
    # ---------------------------------------------------------------------------

    preprocessing_setup = {

        'lower_case': True,
        'remove_emojis': False,
        'replace_emojis_by_text': True,
        'remove_stop_words': False, #True,
        'remove_numbers': False,
        'remove_hashtags': True,
        'remove_users': True,
        'remove_urls': True,
        'remove_non_text_characters': True,
        'lemmatize': False,
        'expand_contractions': False,
        'remove_money_values': False,
        'remove_apostrophe_contractions': False,
        'symbols_to_remove': ['*', '@', '<url>'],
        'remove_between_substrings': None, # [('_x0','d_')]
        'remove_terms_hashtags': None
    }


    if id_column:
        extra_columns = [id_column]
    else:
        extra_columns = None

    data_handler = DataHandler(df=df, text_column=text_column, label_column=label_column, 
                            random_state=random_state, extra_columns=extra_columns)

    # data_handler.unsample()

    train_data, test_data = data_handler.split_train_test_dataset(test_percentage=0.2, random_state=random_state)

    return data_handler, train_data, test_data