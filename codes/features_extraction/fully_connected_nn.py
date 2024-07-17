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
        layers = []
        input_size = len(self.feature_columns)

        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU()) # activation function
            input_size = size

        self.num_classes = output_size
        layers.append(nn.Linear(input_size, output_size))  # Output layer
        self.model = nn.Sequential(*layers)

        self.model.to(self.device)

    def train_model(self, data, params):
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        learning_rate = params.get('learning_rate', 0.001)

        # Prepare data
        y = data[self.label_column].values

        y_one_hot = np.zeros((y.size, y.max()+1), dtype=int)
        y_one_hot[np.arange(y.size), y] = 1

        
        features = torch.tensor(data[self.feature_columns].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(y_one_hot, dtype=torch.float32).to(self.device) #.view(-1, 1)

        ic(labels.shape)

        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        if self.num_classes == 2:
            loss_function = nn.BCEWithLogitsLoss()  # binary classification
        else:
            loss_function = nn.CrossEntropyLoss() # multi class classification

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        # Training loop
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate_model(self, data):
        
        y = data[self.label_column].values

        y_one_hot = np.zeros((y.size, y.max()+1), dtype=int)
        y_one_hot[np.arange(y.size), y] = 1

        features = torch.tensor(data[self.feature_columns].values, dtype=torch.float32).to(self.device)
        labels = torch.tensor(y_one_hot, dtype=torch.float32)

        ic(labels)

        self.model.eval()

        with torch.no_grad():
            
            # preds = torch.round(torch.sigmoid(outputs.cpu())).numpy().flatten()

            test_logits = self.model(features).squeeze()

            # test_pred = torch.round(torch.sigmoid(test_logits))
            test_logits = torch.softmax(test_logits, dim=-1)

            preds = np.argmax(test_logits.cpu(), axis=1)
            labels = np.argmax(labels, axis=1)


        average_mode = 'weighted'
        
        # self.plot_confusion_matrix(labels, preds, 'confusion_matrix.png')

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average=average_mode),
            'recall': recall_score(labels, preds, average=average_mode),
            'f1': f1_score(labels, preds, average=average_mode)
        }

        predictions_df = pd.DataFrame({'text':data['text'].to_list(), 'predictions': preds, 'labels': labels})
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