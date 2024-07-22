# https://github.com/milindmalshe/Fully-Connected-Neural-Network-PyTorch/blob/master/FCN_MNIST_Classification_PyTorch.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from icecream import ic

class TabularNeuralNetworkModel:
    def __init__(self, feature_columns, label_column):
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_model(self, layer_sizes, output_size):
        """
        Create a neural network model with the specified layer sizes and output size.

        Args:
            layer_sizes (list of int): A list containing the sizes of each hidden layer.
            output_size (int): The size of the output layer, typically the number of classes for classification tasks.

        Returns:
            nn.Sequential: The constructed neural network model.
        """

        # Initialize an empty list to hold the layers of the model
        layers = []

        # Set the input size to the number of feature columns
        input_size = len(self.feature_columns)

        # Loop over the specified layer sizes to create hidden layers
        for size in layer_sizes:
            # Add a linear layer
            layers.append(nn.Linear(input_size, size))
            # Add a ReLU activation function
            layers.append(nn.ReLU())
            # Update the input size for the next layer
            input_size = size

        # Set the number of output classes
        self.num_classes = output_size

        # Add the output layer
        layers.append(nn.Linear(input_size, output_size))

        # Create a sequential model with the defined layers
        self.model = nn.Sequential(*layers)

        # Move the model to the specified device (CPU or GPU)
        self.model.to(self.device)

        # Return the constructed model
        return self.model


    def train_model(self, data, params):
        """
        Train the neural network model.

        Args:
            data (pd.DataFrame): The dataset containing features and labels.
            params (dict): Dictionary of parameters including 'epochs', 'batch_size', and 'learning_rate'.

        Returns:
            None
        """
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.0001)

        # Prepare data
        y = data[self.label_column].values

        # One-hot encode the labels for multi-class classification
        y_one_hot = np.zeros((y.size, y.max() + 1), dtype=int)
        y_one_hot[np.arange(y.size), y] = 1

        # Convert features and labels to PyTorch tensors
        features = torch.tensor(data[self.feature_columns].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(y_one_hot, dtype=torch.float32).to(self.device)

        # Debugging information
        print(f'Labels shape: {labels.shape}')

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        if self.num_classes == 2:
            loss_function = nn.BCEWithLogitsLoss()  # Binary classification
        else:
            loss_function = nn.CrossEntropyLoss()  # Multi-class classification

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Set the model to training mode
        self.model.train()

        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}')

    def evaluate_model(self, data):
        """
        Evaluate the neural network model.

        Args:
            data (pd.DataFrame): The dataset containing features and labels.

        Returns:
            metrics (dict): Dictionary containing accuracy, precision, recall, and F1 score.
            predictions_df (pd.DataFrame): DataFrame containing the original texts, predictions, and true labels.
            test_logits (torch.Tensor): Logits output from the model.
        """
        # Extract true labels and one-hot encode them
        y = data[self.label_column].values
        y_one_hot = np.zeros((y.size, y.max() + 1), dtype=int)
        y_one_hot[np.arange(y.size), y] = 1

        # Convert features and labels to PyTorch tensors
        features = torch.tensor(data[self.feature_columns].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(y_one_hot, dtype=torch.float32).to(self.device)

        # Debugging information
        print(f'Labels: {labels}')

        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Get model predictions
            test_logits = self.model(features).squeeze()

            # Apply softmax to logits to get probabilities
            test_logits = torch.softmax(test_logits, dim=-1)

            # Convert logits to class predictions
            preds = np.argmax(test_logits.cpu(), axis=1)
            true_labels = np.argmax(labels.cpu(), axis=1)

        # Set the averaging method for multi-class classification metrics
        average_mode = 'weighted'

        # Calculate performance metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, preds),
            'precision': precision_score(true_labels, preds, average=average_mode),
            'recall': recall_score(true_labels, preds, average=average_mode),
            'f1': f1_score(true_labels, preds, average=average_mode)
        }

        # Create a DataFrame to store texts, predictions, and true labels
        predictions_df = pd.DataFrame({
            'text': data[self.text_column].to_list(),
            'predictions': preds,
            'labels': true_labels
        })

        return metrics, predictions_df, test_logits.cpu()


    def plot_confusion_matrix(self, labels, preds, save_path=None):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(self.num_classes), yticklabels=range(self.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()