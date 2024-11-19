import torch
from read_data import read_dataset
from icecream import ic
from preprocessing import preprocess
from train_model import train_log_model

## Local
path = '../../../../' # '/home/guto/Documents/Projects/hate_speech/' 
datasets_path = path+'datasets/'

# ## Colab
# path = 'drive/My Drive/hate_speech/'
# datasets_path = path+'datasets/'

random_state = 42

hate_speech_file_names = {
    'racism': {'manual':'Manually_racism_EUROS.csv', 'GPT':'GPT_Manually_racism_EUROS.xlsx', 'true_positive_tab':'TP-RC', 'true_negative_tab':'ABT-RC', 'false_positive_tab':'FP-RC'},
    # 'sexism': {'manual':'Manually_sexism_EUROS.csv', 'GPT':'GPT_Manually_sexism_EUROS.xlsx', 'true_positive_tab':'TP-FS-SW', 'true_negative_tab':'ABT-FS-SW', 'false_positive_tab':'FP-FS-SW'},
    # 'ableism': {'manual':'Manually_ableism_EUROS.csv', 'GPT':'GPT_Manually_ableism_EUROS.csv'}
}

id_column = 'id'
text_column = 'text'
label_column = 'label'

df, labels = read_dataset(hate_speech_file_names, datasets_path, text_column, label_column, random_state=42)

data_handler, train_data, test_data = preprocess(df, text_column=text_column, label_column=label_column)

# test_data.to_csv('test_dataset.csv', index=False)

# Set up training arguments
training_parameters = {
    'labels': labels,
    'dataset_train':train_data,
    'dataset_test':test_data,
    
    'learning_rate':0.000001,
    # 'eps':1e-8,
    'betas':(0.9, 0.999),
    'weight_decay':0.01,
    'loss_function':torch.nn.CrossEntropyLoss(),
    
    'epochs':1,
    'seed':42,
    'repetitions':1,

    'model_file_name': None, #'saved_models/'+model_name,
    #Early stop mechanism
    'patience': 2,
    'min_delta': 0.1
}

parameters_to_track = [
    'learning_rate',
    'weight_decay'
]

train_log_model(experiment_name='LLM_evaluation', model_name='FacebookAI/roberta-base', data_handler=data_handler,
            training_parameters=training_parameters,parameters_to_track=parameters_to_track, save_model=True)
