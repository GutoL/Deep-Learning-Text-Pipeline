
import pandas as pd 
import numpy as np

from language_model_handler.pytorch_language_model_handler import PytorchLanguageModelHandler

class ExperimentManager():
    def __init__(self, data_handler, dataset_type):
        self.data_handler = data_handler
        self.dataset_type = dataset_type
        self.metrics = ('eval_accuracy','eval_precision','eval_recall', 'eval_f1')

    def start_experiment(self, experiment_design, preprocessing_setup):
        self.data_handler.preprocess(setup=preprocessing_setup)

        train_data, test_data = self.data_handler.split_train_test_dataset()

        if experiment_design['unsample']:
            self.data_handler.unsample()

        experiment_results = {}

        for model_name in experiment_design['model_list']:

            print('----------------------------------------')
            print('Training:', model_name)

            language_model_manager = PytorchLanguageModelHandler(model_name=model_name,
                                              text_column=self.data_handler.get_text_column_name(),
                                              label_column=self.data_handler.label_column,
                                              new_labels=experiment_design['new_labels'])

            language_model_manager.prepare_training_testing_datasets(train_data, test_data)

            language_model_manager.create_model()

            results, self.model = language_model_manager.train_evaluate_model(training_args=experiment_design['training_args'],
                                                               early_stopping_patience=experiment_design['early_stopping_patience'],
                                                               iterations=experiment_design['iterations'])

            df_results = pd.DataFrame()
            df_results['Dataset'] = [self.dataset_type] * len(self.metrics)
            df_results['Model'] = [model_name] * len(self.metrics)
            df_results['Metric'] = [metric.replace('eval_', '').capitalize() for metric in self.metrics]
            df_results['Value'] = [np.mean(results[k]) for k in self.metrics if k in results]

            experiment_results[model_name] = {'results':df_results, 'model':language_model_manager}

        return experiment_results


# ---------------------------------------------------------------------------------------------------------------------------


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

# # No preprocessing
# # preprocessing_setup = {key: False for key in preprocessing_setup}

# # Set up training arguments
# training_args = TrainingArguments(
#     output_dir="./sentiment_transfer_learning_transformer/",
#     logging_dir='./sentiment_transfer_learning_transformer/logs',
#     logging_strategy='epoch',
#     logging_steps=100,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     learning_rate=5e-6,
#     save_strategy='epoch',
#     save_steps=100,
#     evaluation_strategy='epoch',
#     eval_steps=100,
#     load_best_model_at_end=True,
#     num_train_epochs=10,
#     # seed=42
# )

# experiment_design = {
#     'model_list': [
#         # 'bert-base-uncased',
#         # 'vinai/bertweet-base',
#         # 'cardiffnlp/twitter-roberta-base-offensive', # Offensive speech Roberta
#         # 'Hate-speech-CNERG/dehatebert-mono-english' # Hate speech Roberta

#         # 'Pablo94/racism-finetuned-detests-29-10-2022', ## Racism model
#         'bitsanlp/Homophobia-Transphobia-v2-mBERT-EDA' ## Homophobia model
#     ],
#     'unsample': True,
#     'early_stopping_patience': 2,
#     'training_args': training_args,
#     'iterations': 1
# }

# data_handler = DataHandler(df=hate_speech_df, text_column=original_text_column, label_column=label_column)

# experiment_manager = ExperimentManager(data_handler, dataset_type=dataset_type)
# results = experiment_manager.start_experiment(experiment_design, preprocessing_setup)

# results[experiment_design['model_list'][0]]['results']