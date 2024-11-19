import pandas as  pd
from icecream import ic

# ---------------------------------------------------------------------------
##### Reading the data set

def read_dataset(hate_speech_file_names, datasets_path, text_column, label_column, random_state=42):

    hate_speech_df = pd.DataFrame()
    non_hate_speech = pd.DataFrame()

    new_labels = {0: 'non_hate_speech'}

    for i, hate_speech_type in enumerate(hate_speech_file_names):

        i += 1
        
        new_labels[i] = hate_speech_type

        # Reading only manually labelled datasets
        
        manual_file_name  = datasets_path+hate_speech_type+'/'+hate_speech_file_names[hate_speech_type]['manual']

        manual_hate_speech_df  = pd.read_csv(manual_file_name)
        manual_hate_speech_df[label_column] = [i]*manual_hate_speech_df.shape[0]

        # Reading GPT and manually labelled datasets

        GPT_file_name  = datasets_path+hate_speech_type+'/'+hate_speech_file_names[hate_speech_type]['GPT']
        
        if '.csv' in GPT_file_name:
            gpt_hate_speech_df = pd.read_csv(GPT_file_name)
            gpt_hate_speech_df[label_column] = [i]*gpt_hate_speech_df.shape[0]

        else:
            tab_true_positive = hate_speech_file_names[hate_speech_type]['true_positive_tab']
            tab_false_positive = hate_speech_file_names[hate_speech_type]['false_positive_tab']
            tab_true_negative = hate_speech_file_names[hate_speech_type]['true_negative_tab']

            gpt_dataframes = pd.read_excel(GPT_file_name, sheet_name=[tab_true_positive, tab_false_positive, tab_true_negative])

            true_positive_gpt = gpt_dataframes[tab_true_positive]
            false_positive_gpt = gpt_dataframes[tab_false_positive]
            true_negative_gpt = gpt_dataframes[tab_true_negative]

            true_positive_gpt[label_column] = [i]*true_positive_gpt.shape[0] # Hate speech
            true_negative_gpt[label_column] = [0]*true_negative_gpt.shape[0] # non-hate speech
            false_positive_gpt[label_column] = [0]*false_positive_gpt.shape[0] # non-hate speech

            gpt_hate_speech_df = pd.concat([true_positive_gpt[[text_column, label_column]]], axis=0)

            non_hate_speech = pd.concat([non_hate_speech, true_negative_gpt[[text_column, label_column]], false_positive_gpt[[text_column, label_column]]], axis=0)

        # Concatenating hate speech datasets
        hate_speech_df = pd.concat([hate_speech_df, manual_hate_speech_df, gpt_hate_speech_df], axis=0)

    # Reading non-hate speech dataset
    non_hate_speech_df = pd.read_csv(datasets_path+'non_hate_speech/GPT_non_hate_speech_EUROS.csv', low_memory=False)

    non_hate_speech_df = non_hate_speech_df.sample(frac=1, random_state=random_state) # shuffling
    non_hate_speech_df[label_column] = [0]*non_hate_speech_df.shape[0]


    # if we have false positives and true negatives from GPT/manually classifications, let's concatenate with our data set
    if non_hate_speech.shape[0] > 0:
        non_hate_speech_df = pd.concat([non_hate_speech_df, non_hate_speech], axis=0)

    maximum_values_from_classes = hate_speech_df[label_column].value_counts().max()
    non_hate_speech_df = non_hate_speech_df.head(maximum_values_from_classes)

    hate_speech_df = pd.concat([hate_speech_df, non_hate_speech_df], axis=0)

    hate_speech_df = hate_speech_df.drop_duplicates(subset=[text_column]) # droping duplicates samples

    hate_speech_df = hate_speech_df.sample(frac=1, random_state=random_state) # shuffling

    return hate_speech_df, new_labels

