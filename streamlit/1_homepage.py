import streamlit as st
import pandas as pd
from streamlit_extras.switch_page_button import switch_page

from config import config

from streamlit_option_menu import option_menu
st.set_page_config(layout='wide')

# st.title('LLM fine-tuning pipeline')

with st.sidebar:
    selected = option_menu(
        menu_title='LLM fine-tuning pipeline',
        options=['Import data sets', 'Pre-processing','Select LLM', 'Run experiment'],
        orientation= 'vertical'
    )

if selected == 'Import data sets':
    split_dataset = st.checkbox("Split dataset in training/testing?")

    if split_dataset:
        train_dataset = st.file_uploader('Choose the complete dataset file')
        number = st.number_input("Define the percentage of the testing dataset", min_value=0.1, max_value=0.99, value=0.2)
        # st.write("The current percentage is ", number)
    else:
        train_dataset = st.file_uploader('Choose the training dataset file')
        test_dataset = st.file_uploader('Choose the testing dataset file')

    want_to_contribute = st.button("Next")
    if want_to_contribute:
        switch_page("Pre-processing")
       

elif selected == 'Pre-processing':
    st.title(selected)

elif selected == 'Select LLM':
    st.title(selected)

elif selected == 'Run experiment':
    st.title(selected)