import pandas as pd
pd.set_option("future.no_silent_downcasting", True)

from icecream import ic

class CustomLiwc():

    def __init__(self, dictionary_file_name) -> None:
        self.dictionary = pd.read_csv(dictionary_file_name)
        self.dictionary.set_index('DicTerm', inplace=True)
        self.dictionary.fillna(0, inplace=True)

        self.dictionary.replace({'X': 1}, inplace=True)
    
    def process_text(self, text):
        words = text.split()

        result = self.dictionary[self.dictionary.index.isin(words)].sum()

        result = result.to_frame().T

        # result['text'] = [text]

        result = result.to_dict()

        result = {key:result[key][0] for key in result}

        return result
    
    def process_data_frame(self, df, text_column):
        
        # Create a list to store the results
        results = []

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Get the text from the specified column
            text = row[text_column]
            
            # Process the text using the process_text method
            processed_data = self.process_text(text)
            
            # Append the processed data to the results list
            results.append(processed_data)
        
        # Convert the list of dictionaries to a DataFrame
        results_df = pd.DataFrame(results)
        
        # Combine the original DataFrame with the new DataFrame
        combined_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
    
        return combined_df


        
# my_liwc = CustomLiwc('dehumanization-dictionary.dicx')

# # result = my_liwc.process_text('cockroach and automatic aliens are very cool')

# result = my_liwc.process_data_frame(pd.DataFrame({'text':['cockroach and automatic aliens are very cool', 'I am angry']}), 'text')
# ic(result)
