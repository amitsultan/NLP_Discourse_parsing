from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import trange
import random
from helper import *
import json
import torch
import torch.nn as nn

class BertController:
    '''
    Class is responsible for handling transformers.bert
    this includes preparing the model for various input sizes
    as well as output sizes. SUPPORT CUDA / CPU.
    '''
    
    def __init__(self, max_seq, num_labels, name=None):
        '''
        Initialize bert controller according to
        the max sequence allowed and the number
        of labels required for the output.
        The base transformer here is BertForSequenceClassification.
        :param max_seq: int - Maximum sequence allowed.
        :param num_labels: int - Number of labels available for output.
        '''
        self.__init_tokenizer()
        self.max_length = max_seq
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if name:
            self.model = BertForSequenceClassification.from_pretrained(name)
            # Move the model to the GPU
            # Change the output layer to produce 1 outputs
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_labels)
            # Use the BinaryCrossEntropyLoss for multilabel classification
            loss_fn = nn.BCEWithLogitsLoss()
            loss_fn = loss_fn.to(self.device)
            self.model = self.model.to(self.device)
            # Set optimizer and scheduler
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels = num_labels,
                output_attentions = False,
                output_hidden_states = False,
            )
            self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                  lr = 5e-5,
                                  eps = 1e-08)
            self.model.cuda()
        self.metrics = {}

    def train(self, train_dataloader, validation_dataloader, epochs=1):
        '''
        Train function will loop for given amount of epochs
        (default 1) and train the bert model based on the
        training dataloader given and evaluate it's performance
        each epoch based on the validation dataloader.
        :param train_dataloader: TensorDataset - Train dataset dataloader object.
        :param validation_dataloader: TensorDataset - Validation dataset dataloader object.
        :param epochs: Integer - Number of desired epochs.
        '''
        ep = 1
        for _ in trange(epochs, desc = 'Epoch'):
            # ========== Training ==========
            # Set model to training mode
            self.model.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                b_labels = b_labels.type(torch.LongTensor).to(self.device)
                self.optimizer.zero_grad()
                # Forward pass
                train_output = self.model(b_input_ids, 
                                     token_type_ids = None, 
                                     attention_mask = b_input_mask, 
                                     labels = b_labels)
                # Backward pass
                train_output.loss.backward()
                self.optimizer.step()
                # Update tracking variables
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            # ========== Validation ==========

            # Set model to evaluation mode
            self.model.eval()

            # Tracking variables 
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_specificity = []

            for batch in validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                  # Forward pass
                  eval_output = self.model(b_input_ids, 
                                      token_type_ids = None, 
                                      attention_mask = b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                # Calculate validation metrics
                b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
                val_accuracy.append(b_accuracy)
                # Update precision only when (tp + fp) !=0; ignore nan
                if b_precision != 'nan': val_precision.append(b_precision)
                # Update recall only when (tp + fn) !=0; ignore nan
                if b_recall != 'nan': val_recall.append(b_recall)
                # Update specificity only when (tn + fp) !=0; ignore nan
                if b_specificity != 'nan': val_specificity.append(b_specificity)
            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
            print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')
            if b_precision != 'nan' and b_recall != 'nan':
                b_f1 = (2 * b_precision * b_recall) / (b_precision + b_recall)
            else:
                b_f1 = 'nan'
            print(f'F-score: {b_f1}')
            self.update_metrics(epoch=ep,
                               loss=round(tr_loss / nb_tr_steps, 4),
                               acc=round(sum(val_accuracy)/len(val_accuracy), 4),
                               precision=round(sum(val_precision)/len(val_precision) if len(val_precision)>0 else 0, 4),
                               recall=round(sum(val_recall)/len(val_recall) if len(val_recall)>0 else 0, 4),
                                specificity=round(sum(val_specificity)/len(val_specificity) if len(val_specificity)>0 else 0, 4),
                                f_score=b_f1)
            ep +=1

    def update_metrics(self, epoch, loss, acc, precision, recall, specificity, f_score):
        '''
        Helper method - Append an epoch results to a history
        dictionary.
        :param epoch: Integer - Epoch number.
        :param loss: Float - training loss value.
        :param acc: Float - evaluation accuracy.
        :param precision: Float - evaluation precision.
        :param recall: Float - evaluation recall.
        :param specificity: Float - evaluation specificity.
        :param f_score: Float - evaluation f-score.
        '''
        self.metrics[f'epoch_{epoch}'] = {
            'train_loss': loss,
            'validation_acc': acc,
            'validation_precision': precision,
            'validation_recall': recall,
            'validation_specificity': specificity,
            'f_score': f_score
        }

    def save_model(self, name):
        '''
        Save model weights to a folder named "saved_models".
        :param name: String - output file name for model weights.
        '''
        if not os.path.isdir('saved_models'):
            os.makedirs('saved_models')
        self.model.save_pretrained(os.path.join(ROOT_DIR, 'saved_models', name))
        with open(os.path.join(ROOT_DIR, 'saved_models', name, 'metrics.json'), 'w') as fp:
            json.dump(self.metrics, fp)
    
    def extract_tokens_labels(self, texts, labels):
        '''
        Method receive both texts and labels and output
        the 3 components for training the bert model.
        :param texts: np.NdArray - array of input texts.
        :param labels: np.NdArray - array of labels.
        :output Dictionary - A dictionary with token_ids
        attention_masks and labels.
        '''
        token_id = []
        attention_masks = []
        for sample in texts:
            encoding_dict = self.preprocessing(sample)
            token_id.append(encoding_dict['input_ids']) 
            attention_masks.append(encoding_dict['attention_mask'])
        token_id = torch.cat(token_id, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        labels = torch.tensor(labels)
        return {'token_ids': token_id,
               'attention_masks': attention_masks,
               'labels': labels
               }

    def preprocessing(self, input_text):
        '''
        Helper method to define and configure the
        tokenizer used in creating the tokens mask and id's.
        :param input_text: String - name of the tokenizer.
        '''
        return self.tokenizer.encode_plus(
                            input_text,
                            add_special_tokens = True,
                            max_length = self.max_length,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt')
    
    def __init_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case = True)