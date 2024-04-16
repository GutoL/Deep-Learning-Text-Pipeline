# https://towardsdatascience.com/multi-class-text-classification-with-deep-learning-using-bert-b59ca2f5c613
# https://www.intodeeplearning.com/bert-multiclass-text-classification/

import torch
import numpy as np 
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from transformers.trainer_pt_utils import get_parameter_names
import pandas as pd
from transformers import EvalPrediction

import matplotlib.pyplot as plt

from codes.language_model_handlers.language_model_handler import LanguageModelHandler

class PytorchLanguageModelHandler(LanguageModelHandler):
    def __init__(self, model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states=True, batch_size=32, text_size_limit=512):
        super().__init__(model_name, new_labels, text_column, processed_text_column, label_column, output_hidden_states, batch_size, text_size_limit)
        self.handler_type = 'pytorch'
    
    def tokenize_dataset(self, data):

        '''
        This function takes list of texts and returns input_ids and attention_mask of texts
        '''
        encoded_dict = self.tokenizer.batch_encode_plus(data, add_special_tokens=True, max_length=128, padding='max_length',
                                                   return_attention_mask=True, truncation=True, return_tensors='pt')

        return encoded_dict['input_ids'], encoded_dict['attention_mask']


    def evaluate_model(self, test_dataset):
        
        # classifications_df = pd.DataFrame(test_dataset[self.text_column].to_list(), columns=[self.text_column])
        classifications_df = pd.DataFrame()

        dataloader_test = self.prepare_dataset(test_dataset)

        self.model.eval()
    
        loss_val_total = 0
        predictions, true_vals = [], []

        texts = []
        labels = []

        for batch in dataloader_test:
            
            batch = tuple(b.to(self.device) for b in batch)            

            for i in range(len(batch[0])):
                text = [token for token in self.tokenizer.decode(batch[0][i], skip_special_tokens=True).split()] # removing special tokens from BERT-based model
                texts.append(' '.join(text))
                labels.append(batch[2][i].cpu().numpy())

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
        
        predictions = np.concatenate(predictions, axis=0).argmax(-1)
        true_vals = np.concatenate(true_vals, axis=0)

        classifications_df['texts'] = texts
        classifications_df['labels'] = labels #true_vals
        classifications_df['predictions'] = predictions

        # metrics = self.compute_metrics({'predictions':np.argmax(predictions, axis=-1),'label_ids':true_vals})
        metrics = self.compute_metrics(EvalPrediction(predictions=predictions, label_ids=true_vals))
        
        return loss_val_avg, metrics, classifications_df
        
    def train_evaluate_model(self, training_parameters):

        # seed_val = training_parameters['seed']
        # random.seed(seed_val)
        # np.random.seed(seed_val)
        # torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)

        path_to_model = '/'.join(training_parameters['model_file_name'].split('/')[:-1])+'/'
        model_name_file = training_parameters['model_file_name'].split('/')[-1]

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
            
            ## Start the training
            best_val_loss = float('inf')

            for epoch_num in range(epochs):
                epoch_num = epoch_num + 1

                print('Epoch: ', epoch_num)
                
                self.model.train()
                loss_train_total = 0
                
                for step_num, batch_data in enumerate(tqdm(dataloader_train, desc='Training')):

                    # self.model.zero_grad()

                    batch = tuple(b.to(self.device) for b in batch_data)

                    inputs = {
                                'input_ids': batch[0], 
                                'attention_mask': batch[1], 
                                'labels': batch[2]
                            }
                    
                    outputs = self.model(**inputs)

                    # loss = outputs[0] # output.loss
                    loss = training_parameters['loss_function'](outputs.logits, inputs['labels'].long())
                    
                    loss_train_total += loss.item()

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
            
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    
                    del loss

                train_loss_per_epoch.append(loss_train_total / (step_num + 1))

                # Testing
                loss_val_avg, metrics, _ = self.evaluate_model(test_dataset=training_parameters['dataset_test'])

                test_loss_per_epoch.append(loss_val_avg)

                print(f'Test performance after epoch {epoch_num}:', metrics)

                # Save the model if it has the best validation loss
                if (loss_val_avg < best_val_loss) and training_parameters['model_file_name']:
                    best_val_loss = loss_val_avg
                    self.save_model(path=path_to_model, name_file=model_name_file)

                    fp = open(path_to_model+model_name_file+'/epoch_number.txt', "w")
                    fp.write(str(epoch_num))
                    fp.close()

            for m in metrics:
                if m in metrics_results:
                    metrics_results[m].append(metrics[m])
                else:
                    metrics_results[m] = [metrics[m]]

        epochs = range(1, epochs +1 )
        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss_per_epoch,label ='Training loss')
        ax.plot(epochs, test_loss_per_epoch, label = 'Test loss' )
        ax.set_title('Training and Test loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show()

        for metric in metrics_results:
            print(metric, np.mean(metrics_results[metric]))

        return metrics_results, self.model # '''