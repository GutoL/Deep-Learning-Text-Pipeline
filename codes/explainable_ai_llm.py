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
        """
        Calculate word-level attributions for a given text using Captum's LayerIntegratedGradients.

        Args:
            text (str): The input text for which to calculate word attributions.

        Returns:
            inputs (tensor): The tokenized inputs for the text.
            prediction (list): The model's prediction for the text.
            word_attributions (tensor): The attributions for each word in the text.
            delta (float): The convergence delta value.
        """
        # Get model prediction for the text
        prediction = self.__pipeline.predict(text)
        
        # Generate tokenized inputs for the text
        inputs = self.__generate_inputs(text)
        
        # Generate a baseline for the inputs with the same length as the inputs
        baseline = self.generate_baseline(sequence_len=inputs.shape[1])

        # Initialize Layer Integrated Gradients with the embedding layer of the model
        lig = LayerIntegratedGradients(self.forward_func, getattr(self.__pipeline.model, self.__name).embeddings)

        # Swap label dictionary keys and values for target identification
        labels_swaped = {v: k for k, v in self.__pipeline.model.config.id2label.items()}

        # Compute word attributions using Layer Integrated Gradients
        word_attributions, delta = lig.attribute(
            inputs=inputs,
            baselines=baseline,
            target=labels_swaped[prediction[0]['label']],
            return_convergence_delta=True
        )
        
        return inputs, prediction, word_attributions, delta


    def explain(self, text: str, file_name: str, bar: bool = True):
        """
        Generate and visualize word attributions for a given text.

        Args:
            text (str): The input text to explain.
            file_name (str): The file name to save the plot.
            bar (bool): Whether to plot a bar chart of word scores (default: True). If False, plots colored text.

        Returns:
            None
        """
        # Calculate word scores using Captum
        inputs, prediction, word_attributions, delta = self.calculate_word_scores_using_captum(text)

        print('Prediction:', prediction)

        # Aggregate subtokens into words and their corresponding scores
        words_list, scores_list = self.aggregate_subtokens_into_words(inputs, word_attributions)

        # Sum the scores for each word
        scores_list = [np.sum(scores) for scores in scores_list]

        # Visualize the word scores
        if bar:
            # Plot bar chart of word scores
            self.plot_word_scores_bar(
                words_list,
                scores_list,
                file_name,
                prediction[0]['label'] + ': ' + str(round(prediction[0]['score'] * 100, 2)) + '%'
            )
        else:
            # Plot colored text based on word scores
            self.plot_colored_text(words_list, scores_list)



    def aggregate_subtokens_into_words(self, inputs: list, word_attributions: list):
        """
        Aggregate subtokens into words and their corresponding attributions.

        Args:
            inputs (list): The tokenized inputs.
            word_attributions (list): The attributions for each subtoken.

        Returns:
            words_list (list): List of words.
            scores_list (list): List of scores for each word.
        """
        # Sum the attributions over the last dimension
        attr_sum = word_attributions.sum(-1)
        # Normalize the attributions
        attr = attr_sum / torch.norm(attr_sum)

        # Convert token IDs to tokens, skipping special tokens
        words = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0], skip_special_tokens=True)
        scores = attr.cpu().numpy()[0]

        # Pair each word with its score
        token_scores_list = [(word, score) for word, score in zip(words, scores)]

        print('token_scores_list', token_scores_list)
        
        # Combine tokens into words based on the model type
        if self.__name == 'roberta':
            words_list, scores_list = self.combine_tokens_into_words_roberta(token_list=token_scores_list)
        elif self.__name == 'bert':
            words_list, scores_list = self.combine_tokens_into_words_bert(token_tuples=token_scores_list)
        
        return words_list, scores_list


    def combine_tokens_into_words_roberta(self, token_list):
        """
        Combine sub-tokens into full words for RoBERTa tokenization, and aggregate their associated scores.

        Args:
            token_list (list of tuples): A list where each element is a tuple containing:
                - token (str): A token, potentially starting with 'Ġ' which indicates a new word.
                - score (float): The attribution score for the token.

        Returns:
            tuple: A tuple containing:
                - words (list of str): List of combined words where sub-tokens have been merged.
                - scores (list of lists of float): List of lists where each sub-list contains scores for the corresponding word.
        """
        
        words = []  # List to hold the combined words
        scores = []  # List to hold the scores for each word
        current_word = ''  # Accumulator for the current word being built
        current_score_list = []  # Accumulator for the scores associated with the current word

        for token, score in token_list:
            # Check if the token starts with 'Ġ', which indicates the start of a new word
            if token.startswith('Ġ'):
                # If there's a currently accumulated word, add it to the words list
                if current_word:
                    words.append(current_word)
                    scores.append(current_score_list)
                    current_word = ''  # Reset the current word
                    current_score_list = []  # Reset the current score list

                # Remove the leading 'Ġ' from the token and append it to the current word
                current_word += token.lstrip('Ġ')
                current_score_list.append(score)
            else:
                # Append the token to the current word
                current_word += token
                current_score_list.append(score)

        # Append the last word and its score list, if any
        if current_word:
            words.append(current_word)
            scores.append(current_score_list)

        return words, scores


    def combine_tokens_into_words_bert(self, token_tuples):
        """
        Combine sub-tokens into full words for BERT tokenization, and aggregate their associated scores.

        Args:
            token_tuples (list of tuples): A list where each element is a tuple containing:
                - token (str): A token, where sub-tokens may start with '##' indicating continuation of a word.
                - score (float): The attribution score for the token.

        Returns:
            tuple: A tuple containing:
                - words (list of str): List of combined words where sub-tokens have been merged.
                - scores (list of lists of float): List of lists where each sub-list contains scores for the corresponding word.
        """
        
        def combine_and_clean(words_list):
            """
            Combine sub-tokens into single words and clean up the word tokens by removing '##' prefixes.

            Args:
                words_list (list of lists of str): List where each sub-list contains tokens for a word.

            Returns:
                list of str: List of cleaned and combined words.
            """
            cleaned_words_list = []
            for sublist in words_list:
                combined_words = ''.join(sublist)
                cleaned_sublist = combined_words.replace('##', '')  # Remove '##' that indicates sub-token
                cleaned_words_list.append(cleaned_sublist)
            return cleaned_words_list

        self.tokens_to_exclude = ['[CLS]', '[SEP]']  # Tokens to be ignored (special tokens)
        tokens_list = []  # List to hold combined tokens for each word
        scores_list = []  # List to hold scores for each word

        current_tokens_list = []  # Accumulator for the current word's tokens
        current_scores_list = []  # Accumulator for the current word's scores

        for i, (token, score) in enumerate(token_tuples):
            # Skip tokens that are to be excluded
            if token in self.tokens_to_exclude:
                continue

            if i < len(token_tuples) - 1:
                next_token = token_tuples[i + 1][0]

                # If the next token is not a continuation token (doesn't start with '##') and current tokens exist
                if '##' not in next_token and len(current_tokens_list) > 0:
                    current_tokens_list.append(token)
                    current_scores_list.append(score)

                    tokens_list.append(current_tokens_list)
                    scores_list.append(current_scores_list)

                    # Reset accumulators for the next word
                    current_tokens_list = []
                    current_scores_list = []

                # If the next token is not a continuation token and no current tokens
                elif '##' not in next_token and len(current_tokens_list) == 0:
                    tokens_list.append([token])
                    scores_list.append([score])

                # If the next token is a continuation token
                elif '##' in next_token:
                    current_tokens_list.append(token)
                    current_scores_list.append(score)

        # Handle the last token and its score
        last_token = token_tuples[-1][0]
        last_score = token_tuples[-1][1]

        if '##' in last_token and last_token not in self.tokens_to_exclude:
            tokens_list.append(current_tokens_list + [last_token])
            scores_list.append(current_scores_list + [last_score])

        elif '##' not in last_token and last_token not in self.tokens_to_exclude:
            tokens_list.append([last_token])
            scores_list.append([last_score])

        # Combine and clean the tokens into words
        return combine_and_clean(tokens_list), scores_list



    def __get_most_impactful_words_integrated_gradients(self, text_to_evaluate, threshold, keyword, results):
        """
        Identifies and updates the most impactful words for a given text based on Integrated Gradients attributions.

        Args:
            text_to_evaluate (str): The text for which to evaluate word attributions.
            threshold (float): The minimum attribution score required for a word to be considered impactful.
            keyword (str): The class name for which to filter the results.
            results (dict): A dictionary where keys are words and values are their counts.

        Returns:
            dict: Updated dictionary of impactful words with their counts.
        """
        
        # Get word attributions using Integrated Gradients
        word_attributions = self.__cls_explainer(text=text_to_evaluate)

        # Combine sub-tokens into full words and their associated scores
        tokens_list, scores_list = self.combine_tokens_into_words_bert(word_attributions)

        # Prepare a list of words with their average attribution scores
        new_word_attributions = []
        for i, tokens in enumerate(tokens_list):
            word = self.__pipeline.tokenizer.convert_tokens_to_string(tokens)
            average_score = np.mean(scores_list[i])
            new_word_attributions.append((word, average_score))

        # If the predicted class matches the keyword, update the results dictionary
        if self.__cls_explainer.predicted_class_name == keyword:
            for word, score in new_word_attributions:
                if score > threshold:
                    if word in results:
                        results[word] += 1
                    else:
                        results[word] = 1

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
