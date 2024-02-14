# https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
# https://www.intodeeplearning.com/bert-multiclass-text-classification/

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler #, SequentialSampler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score #, classification_report
import numpy as np 
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from transformers.trainer_pt_utils import get_parameter_names
import pandas as pd
from transformers import EvalPrediction

from codes.language_model_handlers.language_model_handler import LanguageModelHandler

class PytorchLanguageModelHandler(LanguageModelHandler):
    def __init__(self, model_name, new_labels, text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        super().__init__(model_name, new_labels, text_column, label_column, output_hidden_states, batch_size, text_size_limit)
        self.handler_type = 'pytorch'
    
    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']


    def evaluate_model(self, test_dataset):

        dataloader_test = self.prepare_dataset(test_dataset)

        self.model.eval()
    
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_test:
            
            batch = tuple(b.to(self.device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[2],
                     }
    
            with torch.no_grad():        
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()
    
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        loss_val_avg = loss_val_total/len(dataloader_test) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)

        # metrics = self.compute_metrics({'predictions':np.argmax(predictions, axis=-1),'label_ids':true_vals})
        metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=true_vals))
        
        return loss_val_avg, metrics
        
    def train_evaluate_model(self, training_parameters):
        
        # seed_val = training_parameters['seed']
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        # torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)
        
        self.num_labels = len(training_parameters['dataset_train'][self.label_column].value_counts())

        dataloader_train = self.prepare_dataset(training_parameters['dataset_train'])
        
        epochs = training_parameters['epochs']

        metrics_results = {}
        
        for x in range(training_parameters['repetitions']):
            self.create_dl_model()            

            # optimizer = AdamW(self.model.parameters(), lr=training_parameters['learning_rate']) # eps=training_args['eps']

            # Create adamw_torch optimizer manually
            decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": training_parameters['weight_decay'],
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]

            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=training_parameters['learning_rate'],
                betas=(training_parameters['betas'][0],training_parameters['betas'][1])            
            )


            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)

            
            train_loss_per_epoch = []
            test_loss_per_epoch = []


            for epoch_num in range(epochs):
                print('Epoch: ', epoch_num + 1)
                
                self.model.train()

                loss_train_total = 0

                for step_num, batch_data in enumerate(tqdm(dataloader_train,desc='Training')):

                    self.model.zero_grad()

                    ##############
                    # input_ids, attention_mask, labels = [data for data in batch_data] # data.to(self.device)
                    ## output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    # output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    ##############

                    batch = tuple(b.to(self.device) for b in batch_data)

                    inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'labels':         batch[2],
                            }
                    
                    outputs = self.model(**inputs)

                    # loss = outputs[0] # output.loss
                    loss = training_parameters['loss_function'](outputs.logits, inputs['labels'].long())
                    
                    loss_train_total += loss.item()

                    # Backward pass
                    loss.backward()
            
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
                    optimizer.step()
                    scheduler.step()
                    
                    del loss

                train_loss_per_epoch.append(loss_train_total / (step_num + 1))

                # Testing
                loss_val_avg, metrics = self.evaluate_model(test_dataset=training_parameters['dataset_test'])

                test_loss_per_epoch.append(loss_val_avg)

                for m in metrics:
                    if m in metrics_results:
                        metrics_results[m].append(metrics[m])
                    else:
                        metrics_results[m] = [metrics[m]]
                        
                print('Test performance:', metrics)

        # epochs = range(1, epochs +1 )
        # fig, ax = plt.subplots()
        # ax.plot(epochs, train_loss_per_epoch,label ='Training loss')
        # ax.plot(epochs, test_loss_per_epoch, label = 'Test loss' )
        # ax.set_title('Training and Test loss')
        # ax.set_xlabel('Epochs')
        # ax.set_ylabel('Loss')
        # ax.legend()
        # plt.show()

        for metric in metrics_results:
            print(metric, np.mean(metrics_results[metric]))

        return metrics_results, self.model