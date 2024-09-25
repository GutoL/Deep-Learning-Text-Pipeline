import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from config import config

import sys
sys.path.append("../")

from codes.data_handler import DataHandler
from icecream import ic

st.title('Preprocessing Datasets')

# if config['complete_df_name'] not in st.session_state:


preprocessing_finished = False

if config['complete_df_name'] in st.session_state:
    complete_df = st.session_state[config['complete_df_name']]

    data_handler = DataHandler(df=complete_df, 
                               text_column=st.session_state.text_column, 
                               label_column=st.session_state.label_column, 
                               random_state=config['random_state'], 
                               extra_columns=[])
    
    train_df, test_df = data_handler.split_train_test_dataset(test_percentage=st.session_state.test_dataset_percentage)

    st.session_state[config['train_df_name']] = train_df
    st.session_state[config['test_df_name']] = test_df

else:
    train_df = st.session_state[config['train_df_name']]
    test_df = st.session_state[config['test_df_name']]

st.markdown('Number of samples in training dataset: '+ str(train_df.shape[0]))
st.markdown('Number of samples in testing dataset: '+ str(test_df.shape[0]))

preprocessing_config = {}

st.write('Select the cleaning process steps you want to do in your text dataset:')
preprocessing_config['lower_case'] = st.checkbox('Convert to lower case')
preprocessing_config['remove_emojis'] = st.checkbox('Remove emoji')

if preprocessing_config['remove_emojis'] == False:
    preprocessing_config['replace_emojis_by_text'] = st.checkbox('Replace emoji by text (if it is not removed)')
else:
    preprocessing_config['replace_emojis_by_text'] = False

preprocessing_config['remove_stop_words'] = st.checkbox('Remove stop words')
preprocessing_config['remove_numbers'] = st.checkbox('Remove numbers')
preprocessing_config['remove_hashtags'] = st.checkbox('Remove hashtags')
preprocessing_config['remove_users'] = st.checkbox('Remove user mentions')
preprocessing_config['remove_urls'] = st.checkbox('Remove urls')
preprocessing_config['remove_non_text_characters'] = st.checkbox('Remove non text characters')
preprocessing_config['expand_contractions'] = st.checkbox('Expand text contractions')
preprocessing_config['remove_money_values'] = False
preprocessing_config['remove_apostrophe_contractions'] = False
preprocessing_config['remove_apostrophe_contractions'] = False
preprocessing_config['symbols_to_remove'] = False
preprocessing_config['remove_between_substrings'] = False
preprocessing_config['remove_terms_hashtags'] = False
preprocessing_config['lemmatize'] = False

st.session_state['selected_preprocessing_steps'] = [key for key, value in preprocessing_config.items() if value == True]

if st.button("Run preprocessing"):
    with st.spinner('Running preprocessing...'):
        st.session_state.train_data_handler = DataHandler(df=train_df, text_column=st.session_state.text_column, 
                                        label_column=st.session_state.label_column, 
                                        extra_columns=[], 
                                        random_state=config['random_state'])
        
        st.session_state.test_data_handler = DataHandler(df=test_df, text_column=st.session_state.text_column, 
                                        label_column=st.session_state.label_column, 
                                        extra_columns=[], 
                                        random_state=config['random_state'])

        train_data = st.session_state.train_data_handler.preprocess(setup=preprocessing_config)
        test_data = st.session_state.test_data_handler.preprocess(setup=preprocessing_config)

        st.markdown('Preprocessing completed!')
        preprocessing_finished = True


        st.session_state[config['train_df_name']] = train_data
        st.session_state[config['test_df_name']] = test_data

st.session_state['processed_text_column'] = data_handler.get_text_column_name()


col1, col2, col3 = st.columns([3, 0.6, 0.6], gap='small')


previous_step = col2.button("Previous")

if previous_step:
    switch_page("import_datasets")


next_step = col3.button("Next")
if next_step:
    switch_page("select_llm")
    preprocessing_finished = False
