import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title=None, filename=None):

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.set_theme(font_scale=1.2)  # Adjust font scale for better readability
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    if title:
        plt.title(title)

    # Save plot if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()

def plot_text_size_distribution(dataframe, column_name, file_name=None):
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

    if file_name:
        plt.savefig(file_name)
    
    plt.close()

def generate_word_cloud(dataframe, column_name, file_name=False):
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

    if file_name:
        plt.savefig(file_name)
    
    plt.close()