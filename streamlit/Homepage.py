import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from config import config

st.set_page_config(
    page_title='LLM fine-tuning pipeline'
)

st.title('Main Page')

start_process = st.button("Start")
if start_process:
    switch_page("import_datasets")

st.sidebar.success('Select a page above')