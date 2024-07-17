# https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074
# https://levelup.gitconnected.com/huggingface-transformers-interpretability-with-captum-28e4ff4df234

from transformers import pipeline 
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import re
import string
import emoji
import numpy as np
from captum.attr import LayerIntegratedGradients
from matplotlib.colorbar import ColorbarBase

plt.rcParams['figure.dpi'] = 200

class ExplainableTransformerPipeline():
    """Wrapper for Captum framework usage with Huggingface Pipeline"""

    def __init__(self, model, tokenizer, device, pipeline_name='text-classification'):

        if 'Roberta' in model.__class__.__name__:
            self.__name = 'roberta'
        elif 'Bert' in model.__class__.__name__:
            self.__name = 'bert'

        self.__pipeline = pipeline(pipeline_name, model=model, tokenizer=tokenizer, device=device)
        self.__cls_explainer = SequenceClassificationExplainer(model, tokenizer)
        self.__device = device

    def forward_func(self, inputs, position = 0):
        """
            Wrapper around prediction method of pipeline
        """
        pred = self.__pipeline.model(inputs, attention_mask=torch.ones_like(inputs))
        return pred[position]

    def __generate_inputs(self, text: str):
        """
            Convenience method for generation of input ids as list of torch tensors
        """
        return torch.tensor(self.__pipeline.tokenizer.encode(text, add_special_tokens=False),
                            device = self.__device).unsqueeze(0)

    def generate_baseline(self, sequence_len: int):
        """
            Convenience method for generation of baseline vector as list of torch tensors
        """
        return torch.tensor([self.__pipeline.tokenizer.cls_token_id] + [self.__pipeline.tokenizer.pad_token_id] * (sequence_len - 2) + [self.__pipeline.tokenizer.sep_token_id], device = self.__device).unsqueeze(0)

    def __clean_text_for_explanation(self, text):
        text = re.sub(r'(?<=:)\s+|\s+(?=:)', '', text)
        text = emoji.emojize(text)

        regular_punct = list(string.punctuation) # python punctuations
        special_punct = ['©', '^', '®',' ','¾', '¡','!'] # user defined special characters to remove

        for punc in regular_punct:
            if punc in text:
                text = text.replace(punc, ' ')

        return text.strip()

    

    ## INTEGRATED GRADIENTS

    def calculate_word_scores_using_transformers_interpret(self, text):
        word_attributions = self.__cls_explainer(text)
        return word_attributions
    
    def visualize_word_importance_in_sentence(self, text:str, file_name:str=None):

        word_attributions = self.calculate_word_scores_using_transformers_interpret(text)

        self.__cls_explainer.visualize(html_filepath=file_name)

    
    def calculate_word_scores_using_captum(self, text):
        prediction = self.__pipeline.predict(text)
        inputs = self.__generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])

        # print(self.__pipeline.model.config.label2id)

        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, self.__name).embeddings)

        # For some reason we need to swap the label dictionary
        labels_swaped = {v: k for k, v in self.__pipeline.model.config.id2label.items()}

        word_attributions, delta = lig.attribute(inputs=inputs, baselines=baseline, target=labels_swaped[prediction[0]['label']], return_convergence_delta=True)
        
        return inputs, prediction, word_attributions, delta

    def explain(self, text: str, file_name:str, bar:bool=True):
        
        inputs, prediction, word_attributions, delta = self.calculate_word_scores_using_captum(text)

        print('prediction:', prediction)

        words_list, scores_list = self.aggregate_subtokens_into_words(inputs, word_attributions)

        scores_list = [np.sum(scores) for scores in scores_list]

        if bar:
            self.plot_word_scores_bar(words_list, scores_list, file_name, prediction[0]['label']+': '+str(round(prediction[0]['score']*100, 2))+'%')
        else:
            self.plot_colored_text(words_list, scores_list)


    def aggregate_subtokens_into_words(self, inputs: list, word_attributions: list):
        
        attr_sum = word_attributions.sum(-1)

        attr = attr_sum / torch.norm(attr_sum)

        words = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0], skip_special_tokens=True)
        scores = attr.cpu().numpy()[0]

        token_scores_list = [(word, score) for word, score in zip(words, scores)]

        print('token_scores_list', token_scores_list)
        
        if self.__name == 'roberta':
            words_list, scores_list = self.combine_tokens_into_words_roberta(token_list=token_scores_list)

        elif self.__name == 'bert':
            words_list, scores_list = self.combine_tokens_into_words_bert(token_tuples=token_scores_list)
          
        return words_list, scores_list

    def combine_tokens_into_words_roberta(self, token_list):
        words = []
        scores = []
        current_word = ''
        current_score_list = []

        for token, score in token_list:
            # Check if the token starts with 'Ġ'
            if token.startswith('Ġ'):
                # If the current_word is not empty, add it to the words list
                if current_word:
                    words.append(current_word)
                    scores.append(current_score_list)
                    current_word = ''  # Reset the current word
                    current_score_list = []  # Reset the current score list

                # Remove the 'Ġ' from the token and add it to the current word
                current_word += token.lstrip('Ġ')
                current_score_list.append(score)
            else:
                # Concatenate the token with the current word
                current_word += token
                current_score_list.append(score)

        # Append the last word and its score list
        if current_word:
            words.append(current_word)
            scores.append(current_score_list)

        return words, scores

    def combine_tokens_into_words_bert(self, token_tuples):
        def combine_and_clean(words_list):
            cleaned_words_list = []
            for sublist in words_list:
                combined_words = ''.join(sublist)
                cleaned_sublist = combined_words.replace('##', '')
                cleaned_words_list.append(cleaned_sublist)
            return cleaned_words_list
        

        self.tokens_to_exclude = ['[CLS]', '[SEP]']
        tokens_list = []
        scores_list = []

        current_tokens_list = []
        current_scores_list = []

        for i, (token, score) in enumerate(token_tuples):
          if token in self.tokens_to_exclude:
              continue

          if i < len(token_tuples)-1:
            next_token = token_tuples[i+1][0]

            if '##' not in next_token and len(current_tokens_list) > 0:
              current_tokens_list.append(token)
              current_scores_list.append(score)

              tokens_list.append(current_tokens_list)
              scores_list.append(current_scores_list)

              current_tokens_list = []
              current_scores_list = []

            elif '##' not in next_token and len(current_tokens_list) == 0:
              tokens_list.append([token])
              scores_list.append([score])

            elif '##' in next_token:
              current_tokens_list.append(token)
              current_scores_list.append(score)

        last_token = token_tuples[-1][0]
        last_score = token_tuples[-1][1]

        if '##' in last_token and last_token not in self.tokens_to_exclude:
          tokens_list.append(current_tokens_list+[last_token])
          scores_list.append(current_scores_list+[last_score])

        elif '##' not in last_token and last_token not in self.tokens_to_exclude:
          tokens_list.append([last_token])
          scores_list.append([last_score])

        return combine_and_clean(tokens_list), scores_list


    def __get_most_impactful_words_integrated_gradients(self, text_to_evaluate, threshold, keyword, results):

        word_attributions = self.__cls_explainer(text=text_to_evaluate)

        # print(self.__cls_explainer.predicted_class_name)
        tokens_list, scores_list = self.combine_tokens_into_words_bert(word_attributions)

        new_word_attributions = []
        for i, tokens in enumerate(tokens_list):
            new_word_attributions.append((self.__pipeline.tokenizer.convert_tokens_to_string(tokens), np.mean(scores_list[i])))

        if self.__cls_explainer.predicted_class_name == keyword:

            for word in new_word_attributions:
                if word[1] > threshold:
                    if word[0] in results:
                        results[word[0]] += 1
                    else:
                        results[word[0]] = 1

        return results

    def plot_word_scores_bar(self, words, scores, file_name, title):
        # Set the color for the bars
        color = '#5B2C6F'
        font_size = 15

        # Create a unique index for each word
        word_indices = np.arange(len(words))

        # Create a bar plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
        bars = plt.bar(word_indices, scores, color=color)

        # Add word labels on x-axis
        plt.xticks(word_indices, words, fontsize=font_size)

        # Add scores on top or below each bar
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score >= 0:
                plt.text(bar.get_x() + bar.get_width() / 2, 
                        bar.get_height(), 
                        str(round(score, 2)), 
                        ha='center', 
                        va='bottom',
                        fontsize=font_size)
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()-0.01, 
                        str(round(score, 2)), 
                        ha='center', 
                        va='top',
                        fontsize=font_size)

        # Add labels and title
        plt.title(title, fontsize=font_size)
        plt.xlabel('Words', fontsize=font_size)
        plt.ylabel('Word Importance Scores', fontsize=font_size)
        
        # Rotate x-axis labels for better readability if needed
        plt.xticks(rotation=45, ha='right')
        plt.ylim(min(scores)-0.1, max(scores)+0.1)

        # Save the plot
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(file_name)  # Save the figure with the specified file name
        plt.close()  # Close the plot to free up memory


    def plot_colored_text(self, text, word_scores):

        # Create a colormap based on the 'viridis' colormap
        cmap = plt.get_cmap('winter')
        # cmap = plt.get_cmap('viridis')

        # Normalize word scores to the range [0, 1]
        norm = plt.Normalize(min(word_scores), max(word_scores))

        # Create a color map using the normalized scores and the colormap
        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])

        # Split the text into words
        words = text.split()

        # Calculate the horizontal spacing between words
        total_word_count = len(words)
        spacing = 1.0 / total_word_count

        # Create a figure and axis for the text
        fig, ax = plt.subplots(figsize=(10, 2))

        for i, (word, score) in enumerate(zip(words, word_scores)):
            color = cmap(norm(score))
            x_position = i * spacing
            ax.text(x_position, 0, word, color=color, fontsize=12, ha='center', rotation=45)

        # Add a color scale just above the text
        colorbar = ColorbarBase(ax=fig.add_axes([0.2, 0.8, 0.6, 0.02]),
                                cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
        colorbar.set_label('Word Impact Scores')

        # Remove the axis and display the plot
        ax.axis('off')
        plt.show()


    # def plot_word_importance(self, sentence, clean_text, bar=True):
        
    #     if clean_text:
    #         sentence = self.__clean_text_for_explanation(sentence)

    #     word_attributions = self.__cls_explainer(text=sentence)

    #     print(word_attributions)

    #     tokens_list, scores_list = self.combine_tokens_into_words_bert(word_attributions)

    #     scores_list = [np.mean(scores) for scores in scores_list]

    #     print(scores_list)

    #     if bar:
    #         print(len(sentence.split()), len(scores_list))
    #         self.plot_vertical_bar(sentence, scores_list)
    #     else:
    #         self.plot_colored_text(sentence, scores_list)
    #         # self.plot_sentence(sentence)


    def get_most_impactful_words_for_dataset(self, dataset, column_text,
                                             threshold, keyword, method, n=20):
        results = {}

        i = 0
        for index, row in dataset.iterrows():

            if i % 100 == 0:
                print('Processing:', i)

            i += 1

            text = self.__clean_text_for_explanation(row[column_text])

            if method == 'integrated_gradients':
                results = self.__get_most_impactful_words_integrated_gradients(text_to_evaluate=text,
                                                            threshold=threshold,
                                                            results=results,
                                                            keyword=keyword)
            elif method == 'lime':
                results, graphic_explanation = self.get_most_impactful_words_lime(
                                                                    text=text,
                                                                    keyword=keyword,
                                                                    word_importance_results=results)

            # if i > 5:
            #     break

        return pd.DataFrame([(key, value) for key, value in dict(sorted(results.items(), key=lambda item: item[1], reverse=True)).items()], columns=['word','frequency']).head(n)
