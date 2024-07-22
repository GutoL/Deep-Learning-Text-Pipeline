import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, title=None, filename=None):
    """
    Plots a confusion matrix to evaluate the performance of a classification model.

    Args:
        y_true (list or np.array): True labels for the test data.
        y_pred (list or np.array): Predicted labels by the classification model.
        labels (list): List of label names (classes) to be used for x and y axis ticks.
        title (str, optional): Title of the confusion matrix plot. If None, no title is set.
        filename (str, optional): Path to save the plot as an image file. If None, the plot is displayed but not saved.

    Returns:
        None
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap
    plt.figure(figsize=(10, 8))  # Set the size of the figure
    sns.set_theme(font_scale=2.5)  # Adjust font scale for better readability

    # Plot confusion matrix with a heatmap
    sns.heatmap(
        cm,
        annot=True,                  # Annotate cells with numerical values
        fmt='d',                     # Format the annotations as decimal integers
        cmap='Blues',                # Use a blue color map
        cbar=False,                  # Disable color bar
        xticklabels=labels,          # Set x-axis labels
        yticklabels=labels           # Set y-axis labels
    )
    
    plt.xlabel('Predicted Label')  # Label for x-axis
    plt.ylabel('True Label')       # Label for y-axis

    plt.xticks(rotation=90)         # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)          # Keep y-axis labels horizontal

    # Set plot title if provided
    if title:
        plt.title(title)

    # Save plot to file if filename is provided, otherwise display the plot
    if filename:
        plt.savefig(filename, bbox_inches='tight')  # Save the plot with tight bounding box
    else:
        plt.show()  # Display the plot


def plot_text_size_distribution(dataframe, column_name, file_name=None):
    """
    Plots the distribution of text sizes from a specified column in a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column containing the text.
        file_name (str, optional): Path to save the plot as an image file. If None, the plot is displayed but not saved.

    Returns:
        None
    """

    # Extract text data from the specified column
    texts = dataframe[column_name]

    # Calculate the length of each text entry
    text_lengths = [len(str(text)) for text in texts]

    # Plot histogram of text lengths
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    plt.hist(
        text_lengths,              # Data to plot
        bins=50,                   # Number of bins
        color='skyblue',           # Color of the bars
        edgecolor='black'          # Color of the bar edges
    )
    plt.title('Text Size Distribution')  # Title of the plot
    plt.xlabel('Text Size')              # X-axis label
    plt.ylabel('Frequency')              # Y-axis label
    plt.grid(True)                       # Show grid

    # Save plot to file if filename is provided, otherwise display the plot
    if file_name:
        plt.savefig(file_name)          # Save the plot as an image file
    else:
        plt.show()                      # Display the plot

    plt.close()                          # Close the plot to free up memory


def generate_word_cloud(dataframe, column_name, file_name=False):
    """
    Generates and plots a word cloud from the text data in a specified column of a DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column containing the text data for generating the word cloud.
        file_name (str, optional): Path to save the word cloud as an image file. If False, the word cloud is displayed but not saved.

    Returns:
        None
    """

    # Concatenate all text entries from the specified column into a single string
    text_corpus = ' '.join(dataframe[column_name].astype(str))

    # Generate the word cloud with specified dimensions and background color
    wordcloud = WordCloud(
        width=800,                # Width of the word cloud image
        height=400,               # Height of the word cloud image
        background_color='white'  # Background color of the word cloud
    ).generate(text_corpus)

    # Plot the word cloud
    plt.figure(figsize=(10, 6))  # Set the size of the figure
    plt.imshow(wordcloud, interpolation='bilinear')  # Display the word cloud with smooth interpolation
    plt.axis('off')  # Hide the axes
    plt.title('Word Cloud')  # Title of the plot
    plt.show()  # Display the plot

    # Save the plot as an image file if a filename is provided
    if file_name:
        plt.savefig(file_name, bbox_inches='tight')  # Save with tight bounding box

    plt.close()  # Close the plot to free up memory
