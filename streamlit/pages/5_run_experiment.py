import sys
sys.path.append("../")
from codes.language_model_handlers.pytorch_language_model_handler import PytorchLanguageModelHandler
import torch

import streamlit as st
from config import config
from streamlit_extras.switch_page_button import switch_page
import matplotlib.pyplot as plt


def train_llm(training_parameters):
    
    classes = list(st.session_state[config['train_df_name']][st.session_state.label_column].value_counts().index)
    new_labels = {i: class_name.lower() for i, class_name in enumerate(classes)}

    model_name = training_parameters['model_name']
    

    language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                                    text_column = st.session_state.text_column,
                                                    processed_text_column=st.session_state.processed_text_column,
                                                    label_column=st.session_state.label_column,
                                                    new_labels=new_labels,
                                                    output_hidden_states=True)
    
    model_performance, model = language_model_manager.train_evaluate_model(training_parameters=training_parameters)

    # language_model_manager.load_llm_model(path='saved_models/', name_file=model_name)
    # _, metrics, llm_classifications_df = language_model_manager.evaluate_model(training_parameters['dataset_test'])

    return model_performance


# st.write(st.session_state)

# Assuming you have already captured these variables from the previous pages
# You might need to load them using session state or however you're passing state between pages
training_dataset = st.session_state.get(config['train_df_name'], "Not specified")
testing_dataset = st.session_state.get(config['test_df_name'], "Not specified")
selected_preprocessing_steps = st.session_state.get('selected_preprocessing_steps', [])


# Summary of Parameters
st.header("Summary of Parameters")

st.subheader("Datasets")

st.write(f"Training Dataset: {training_dataset.shape}")
st.write(f"Testing Dataset: {testing_dataset.shape}")


st.subheader("Preprocessing Steps")
st.write(", ".join(selected_preprocessing_steps) if selected_preprocessing_steps else "No preprocessing steps selected.")

st.subheader("Selected LLM Model")
st.write(f"Model: {st.session_state.model_name}")

st.subheader("Hyperparameters")

epochs = st.number_input("Define the number of epochs:", min_value=1, value=10)
learning_rate = st.number_input("Define the learning rate:", min_value=0.0, max_value=0.1, value=0.000001)
beta1 = st.number_input("Define Beta1:", min_value=0.1, max_value=0.99, value=0.9)
beta2 = st.number_input("Define Beta2:", min_value=0.0001, max_value=1.0, value=0.999)
weight_decay = st.number_input("Define weight decay:", min_value=0.0001, max_value=0.1, value=0.01)
path_to_save = st.text_input("Path to save the trained model:", "")


# Set up training arguments
training_parameters = {
    'model_name':st.session_state.model_name,
    'model_file_name': st.session_state.model_name+st.session_state.model_name,
    
    'loss_function':torch.nn.CrossEntropyLoss(),
    'dataset_train':st.session_state[config['train_df_name']],
    'dataset_test':st.session_state[config['test_df_name']],
    
    'learning_rate':learning_rate,
    'betas':(beta1, beta2),
    'weight_decay':weight_decay,
    'epochs':epochs,
    'seed':42,
    'repetitions':1,
    #Early stop mechanism
    'patience': 2,
    'min_delta': 0.1
}


# Buttons to go back to respective pages
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Edit Dataset"):
        switch_page("import_datasets")

with col2:
    if st.button("Edit Preprocessing"):
        switch_page("preprocessing_datasets")

with col3:
    if st.button("Edit LLM Model"):
        switch_page("select_llm")


col1, col2, col3 = st.columns([3, 0.6, 0.6], gap='small')


previous_step = col2.button("Previous")

if previous_step:
    switch_page("select_llm")


start_training = col3.button("Train", type='primary')
if start_training:

    # Simulate training progress
    with st.expander("Training Progress", expanded=True):
        
        st.write('Training is running... Please check your terminal to see logs.')
        status_text = st.empty()
        
        model_performance = train_llm(training_parameters)

        # model_performance = {'accuracy': [0.91], 'precision': [0.88], 'recall': [0.90], 'f1': [0.889887640449438]}

        # Extract the values for plotting
        metrics = list(model_performance.keys())
        values = [val[0] for val in model_performance.values()]  # Flatten the list (as they are inside lists)

        # Plot the performance metrics as a bar chart
        fig, ax = plt.subplots()

        colours = ['#581845']*4 # ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']
        bars = ax.bar(metrics, values, color=colours)
        
        min_value = min(values) - 0.05
        max_value = max(values) + 0.05

        # Adding titles and labels
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Scores')
        ax.set_ylim([min_value, max_value])  # Since performance metrics are typically between 0 and 1
        
        # Add the values on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',  # Format with 2 decimal places
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Offset to avoid overlapping with the bar
                        textcoords="offset points",
                        ha='center', va='bottom')

        # Display the plot in Streamlit
        st.pyplot(fig)

        status_text.write("Training complete!")
        st.success("Model training finished!")

    # After training, show additional options (e.g., saving the model)
    st.balloons()  # Fun effect to celebrate the end of training
