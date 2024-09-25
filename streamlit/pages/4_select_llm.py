import streamlit as st
from streamlit_extras.switch_page_button import switch_page


st.title('Select LLM')

# st.write(st.session_state)

disable_next_button = True

st.session_state.model_name = st.text_input("Enter your model name", "FacebookAI/roberta-base")

col1, col2, col3 = st.columns([3, 0.6, 0.6], gap='small')


previous_step = col2.button("Previous")

if previous_step:
    switch_page("preprocessing_datasets")

if len(st.session_state.model_name) > 0:
    disable_next_button = False

next_step = col3.button("Next", disabled=disable_next_button)
if next_step:
    switch_page("run_experiment")