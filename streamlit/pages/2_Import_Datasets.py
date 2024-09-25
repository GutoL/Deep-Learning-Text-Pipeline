import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from config import config

def read_data_frame(message, dataset_name, extensions=['csv', 'xlsx']):
    # if dataset_name not in st.session_state:
    #     st.session_state[dataset_name] = None

    uploaded = st.file_uploader(message, type=extensions)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state[dataset_name] = df #.head(10)


st.title('Import Datasets')

st.session_state.text_column = ''
st.session_state.label_column = ''
st.session_state[config['complete_df_name']] = None
st.session_state[config['train_df_name']] = None
st.session_state[config['test_df_name']] = None
main_data_set = None

disable_next_button = True

split_dataset = st.checkbox("Split dataset in training/testing?")

if split_dataset:
    if config['complete_df_name'] is not None:
        read_data_frame('Choose the complete dataset file', config['complete_df_name'])

    main_data_set = config['complete_df_name']
    
    st.session_state.test_dataset_percentage = st.number_input("Define the percentage of the testing dataset", min_value=0.1, max_value=0.99, value=0.2)


else:
    if config['train_df_name'] is not None:
        read_data_frame('Choose the training dataset file', config['train_df_name'])
        read_data_frame('Choose the testing dataset file', config['test_df_name'])

    main_data_set = config['train_df_name']


if (st.session_state[main_data_set] is not None):
    
    st.session_state.text_column = st.selectbox(
        "Select the text column",
        list(st.session_state[main_data_set].columns),
    )

    st.session_state.label_column = st.selectbox(
        "Select the label column",
        list(st.session_state[main_data_set].columns),
    )

# st.session_state.text_column = 'OriginalTweet'
# st.session_state.label_column = 'Sentiment'

# st.write(st.session_state.text_column)
# st.write(st.session_state.label_column)

col1, col2, col3 = st.columns([3, 0.6, 0.6], gap='small')


previous_step = col2.button("Previous")

if previous_step:
    switch_page("homepage")

if (st.session_state[config['complete_df_name']] is not None) or (st.session_state[config['train_df_name']] is not None and st.session_state[config['test_df_name']] is not None ):
    disable_next_button = False

next_step = col3.button("Next", disabled=disable_next_button)
if next_step:
    switch_page("preprocessing_datasets")