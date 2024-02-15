# https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074
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
from IPython.display import display, HTML


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

    def visualize_word_importance_in_sentence(self, text:str, file_name:str=None):

        word_attributions = self.__cls_explainer(text)

        self.__cls_explainer.visualize(html_filepath=file_name)
        
    def visualize_word_importance(self, inputs: list, attributes: list, prediction:str):
        """
            Visualization method.
            Takes list of inputs and correspondent attributs for them to visualize in a barplot
        """
        attr_sum = attributes.sum(-1)

        attr = attr_sum / torch.norm(attr_sum)

        word_importance = pd.Series(attr.cpu().numpy()[0],
                         index = self.__pipeline.tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy()[0],skip_special_tokens=False))

        print(word_importance)

        plt.title(prediction)
        plt.show(word_importance.plot.barh(figsize=(10,20)))

        return word_importance

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

    ## LIME
    def model_adapter(self, texts):

        all_scores = []
        batch_size = 64

        for i in range(0, len(texts), batch_size):

            batch = texts[i:i+batch_size]

            # use bert encoder to tokenize text
            encoded_input = self.__pipeline.tokenizer(batch,
                              return_tensors='pt',
                              padding=True,
                              truncation=True,
                              max_length=self.__pipeline.model.config.max_position_embeddings-2)

            for key in encoded_input:
                encoded_input[key] = encoded_input[key].to(self.__device)

            output = self.__pipeline.model(**encoded_input)
            # by default this model gives raw logits rather
            # than a nice smooth softmax so we apply it ourselves here

            scores = output[0].softmax(1).detach().cpu().numpy()

            all_scores.extend(scores)

        return np.array(all_scores)


    # def get_most_impactful_words_lime(self, text, keyword, word_importance_results):

    #     prediction = self.__pipeline(text)[0]['label']

    #     if prediction == keyword:
    #         print(text)
    #         te = TextExplainer(n_samples=500, random_state=42)
    #         te.fit(text, self.model_adapter)

    #         graphic_explanation = te.explain_prediction(target_names=list(self.__pipeline.model.config.id2label.values()))

    #         print(graphic_explanation.targets)

    #         for element in graphic_explanation.targets:
    #             for f in element.feature_weights.pos:
    #                 for word in f.feature.split():
    #                     if word in word_importance_results:
    #                         word_importance_results[word] += f.weight
    #                     else:
    #                         word_importance_results[word] = f.weight
    #             return word_importance_results, graphic_explanation
    #     else:
    #         return word_importance_results, None



    ## INTEGRATED GRADIENTS
    def explain(self, text: str):
        """
            Main entry method. Passes text through series of transformations and through the model.
            Calls visualization method.
        """
        prediction = self.__pipeline.predict(text)
        inputs = self.__generate_inputs(text)
        baseline = self.generate_baseline(sequence_len = inputs.shape[1])

        print('inputs', len(inputs[0]))
        # print('se liga:', self.__pipeline.model.config.label2id)

        lig = LayerIntegratedGradients(self.forward_func,
                                       getattr(self.__pipeline.model, self.__name).embeddings)

        # For some reason we need to swap the label dictionary
        labels_swaped = {v: k for k, v in self.__pipeline.model.config.id2label.items()}

        attributes, delta = lig.attribute(inputs=inputs,
                                  baselines=baseline,
                                  target=labels_swaped[prediction[0]['label']],
                                  return_convergence_delta=True)

        self.visualize_word_importance(inputs, attributes, prediction)


    def join_tokens_into_words(self, token_tuples):
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

        return tokens_list, scores_list


    def __get_most_impactful_words_integrated_gradients(self, text_to_evaluate, threshold, keyword, results):

        word_attributions = self.__cls_explainer(text=text_to_evaluate)
        # print(self.__cls_explainer.predicted_class_name)
        tokens_list, scores_list = self.join_tokens_into_words(word_attributions)

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

    def plot_vertical_bar(self, text, word_scores):
        # Split the text into words
        words = text.split()

        # Create a vertical bar plot
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(words, word_scores, color='#5B2C6F') # color='skyblue'

        margin = 0.02

        for word, score, bar in zip(words, word_scores, bars):
            if score >= 0:
                ax.text(word, score + margin, f'{score:.2f}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(word, score - margin, f'{score:.2f}', ha='center', va='top', fontsize=10)

        # Rotate word labels by 45 degrees for readability
        ax.set_xticklabels(words, rotation=45, ha='right')

        # Set labels and title
        ax.set_xlabel('Words')
        ax.set_ylabel('Word Impact Scores')
        # ax.set_title('Word Scores Vertical Bar Plot')

        # Adjust the position of the y-axis labels
        ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.set_ylim([np.min(word_scores)-0.1, np.max(word_scores)+0.1])

        # plt.grid(True, color = "grey", linewidth = "1")
        plt.axhline(y=0, color='black', linestyle='-', linewidth="0.5")

        # Display the plot
        plt.tight_layout()
        plt.show()

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


    def plot_word_importance(self, sentence, bar=True):

        sentence = self.__clean_text_for_explanation(sentence)

        word_attributions = self.__cls_explainer(text=sentence)

        print(word_attributions)

        tokens_list, scores_list = self.join_tokens_into_words(word_attributions)

        scores_list = [np.mean(scores) for scores in scores_list]

        print(scores_list)

        if bar:
            print(len(sentence.split()), len(scores_list))
            self.plot_vertical_bar(sentence, scores_list)
        else:
            self.plot_colored_text(sentence, scores_list)
            # self.plot_sentence(sentence)


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
