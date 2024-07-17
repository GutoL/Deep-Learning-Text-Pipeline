
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from codes.features_extraction.fully_connected_nn import TabularNeuralNetworkModel # NeuralNet

import torch

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# import numpy as np

# import torch.nn as nn
# from torch import optim

from torch.utils.data import TensorDataset, DataLoader

from transformers import EvalPrediction

from icecream import ic

class MachineLearningManager():

    def __init__(self, features_names, label_column) -> None:
        self.features_names = features_names
        self.label_column = label_column
    
    def compute_metrics(self, predictions_labels):
        # preds_flat = np.argmax(preds, axis=1).flatten()
        # labels_flat = labels.flatten()
        # return f1_score(labels_flat, preds_flat, average='weighted')
        predictions, labels = predictions_labels

        average_mode = 'weighted'
        
        results = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average=average_mode),
            'recall': recall_score(labels, predictions, average=average_mode),
            'f1': f1_score(labels, predictions, average=average_mode)
        }
        
        return results
    
    def train_evaluate_ml_model(self, training_args):
        
        X_train = training_args['dataset_train'][self.features_names]
        X_test = training_args['dataset_test'][self.features_names]

        y_train = training_args['dataset_train'][self.label_column]
        y_test = training_args['dataset_test'][self.label_column]

        if training_args['ml_model'].lower() == 'random forest':
            ml_model = RandomForestClassifier(n_estimators=100, random_state=training_args['seed'])
            
        elif training_args['ml_model'].lower() == 'svm':
            ml_model = SVC(kernel='linear')

        elif training_args['ml_model'].lower() == 'decision tree':
            ml_model = DecisionTreeClassifier()

        elif training_args['ml_model'].lower() == 'naive bayes':
            ml_model = GaussianNB()

        elif training_args['ml_model'].lower() == 'logistic regression':
            ml_model = LogisticRegression()
        
        elif training_args['ml_model'].lower() == 'gradient boosting':
            ml_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=training_args['seed'])

        print('Training model...')
        ml_model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = ml_model.predict(X_test)

        # Calculate performance
        performance_metrics = self.compute_metrics(EvalPrediction(predictions=y_pred, label_ids=y_test))

        classifications_df = pd.DataFrame()

        classifications_df['text'] = training_args['dataset_test']['text']
        classifications_df['labels'] = training_args['dataset_test'][self.label_column]
        classifications_df['predictions'] = y_pred

        return performance_metrics, classifications_df
    

    def _created_data_loader(self, data, shuffle=True):
        
        input_ids, att_masks = self.tokenize_dataset(data)

        Y = torch.LongTensor(data[self.label_column].to_list())
        
        # move on device (GPU)
        input_ids = input_ids.to(self.device)
        att_masks = att_masks.to(self.device)
        Y = Y.to(self.device)
        
        dataset = TensorDataset(input_ids, att_masks, Y)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return data_loader
    
    def _scale_min_max(self, df):

        for col_name in self.features_names:
            xmin = df[col_name].min()
            xmax = df[col_name].max()
            df[col_name] = (df[col_name] - xmin) / (xmax - xmin)

        return df
    
    def train_evaluate_nn_model(self, training_args):
        
        self.tabular_model = TabularNeuralNetworkModel(feature_columns=self.features_names, label_column=self.label_column)

        self.tabular_model.create_model(layer_sizes=training_args['hidden_size'], output_size=training_args['num_classes'])  # Example layer sizes
        
        self.tabular_model.train_model(training_args['dataset_train'], {'epochs': training_args['epochs'], 'batch_size': training_args['batch_size'], 'learning_rate': training_args['learning_rate']})
        
        metrics, predictions_df, _ = self.tabular_model.evaluate_model(training_args['dataset_test'])

        return metrics, predictions_df

        
    
    # def train_evaluate_nn_model(self, training_args):
        
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #     hidden_size = training_args['hidden_size']
    #     num_classes = training_args['num_classes']
    #     learning_rate = training_args['learning_rate']
    #     num_epochs = training_args['num_epochs']

    #     X_train = torch.tensor(self._scale_min_max(training_args['dataset_train'])[self.features_names].values).to(torch.float32).to(device)
    #     X_test = torch.tensor(self._scale_min_max(training_args['dataset_test'])[self.features_names].values).to(torch.float32).to(device)
        
    #     y_train = training_args['dataset_train'][self.label_column].values
    #     y_test = training_args['dataset_test'][self.label_column].values

    #     y_train_one_hot = np.zeros((y_train.size, y_train.max()+1), dtype=int)
    #     y_train_one_hot[np.arange(y_train.size), y_train] = 1

    #     y_test_one_hot = np.zeros((y_test.size, y_test.max()+1), dtype=int)
    #     y_test_one_hot[np.arange(y_test.size), y_test] = 1

    #     y_train = torch.tensor(y_train_one_hot).to(torch.float32).to(device)
    #     y_test = torch.tensor(y_test_one_hot).to(torch.float32).to(device)

    #     input_size = X_train.shape[1]
        
    #     model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    #     loss_fn = nn.BCEWithLogitsLoss()

    #     optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    #     for epoch in range(num_epochs):
            
    #         # Put the model in training mode
    #         model.train()
            
    #         y_logits = model(X_train).squeeze() # forward pass to get predictions; squeeze the logits into the same shape as the labels

    #         loss = loss_fn(y_logits, y_train) # compute the loss   
            
    #         ## y_pred = torch.round(torch.sigmoid(y_logits)) # convert logits into prediction probabilities
    #         # y_pred = torch.softmax(y_logits, dim=-1) # convert logits into prediction probabilities

            

    #         optimizer.zero_grad() # reset the gradients so they don't accumulate each iteration
    #         loss.backward() # backward pass: backpropagate the prediction loss
    #         optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
            
    #         # Put the model in evaluation mode
    #         model.eval() 

    #         with torch.inference_mode():
    #             test_logits = model(X_test).squeeze()

    #             # test_pred = torch.round(torch.sigmoid(test_logits))
    #             test_pred = torch.softmax(test_logits, dim=-1)
    #             test_loss = loss_fn(test_logits, y_test)
            
    #         predictions = np.argmax(test_pred.cpu(), axis=1)
    #         label_ids = np.argmax(y_test.cpu(), axis=1)

    #         ic(predictions)
    #         ic(label_ids)

    #         metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=label_ids))

    #         print(metrics)

    #         # epoch_count.append(epoch)
    #         # train_loss_values.append(loss.detach().numpy())
    #         # valid_loss_values.append(valid_loss.detach().numpy())
