class ExploratoryDataAnalysis():

    def plot_text_size_distribution(self, dataframe, column_name):
        # Extract text from the specified column
        texts = dataframe[column_name]

        # Calculate text lengths
        text_lengths = [len(str(text)) for text in texts]

        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(text_lengths, bins=50, color='skyblue', edgecolor='black')
        plt.title('Text Size Histogram')
        plt.xlabel('Text Size')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def generate_word_cloud(self, dataframe, column_name):
        # Concatenate all texts in the specified column
        text_corpus = ' '.join(dataframe[column_name].astype(str))

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)

        # Plot the word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud')
        plt.show()