from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import trange
import random
from helper import *
import torch


class DataLoaderGeneric:
    '''
    Class is responsible for loading the different dataset
    and uniformly create text and labels suited for bert model.
    '''

    def __init__(self, name):
        '''
        Initialize a pandas data set along with splitting
        text and labels depending on the data set given.
        :param name: String - abbreviation of the dataset name.
        '''
        self.df = None
        self.text = None
        self.labels = None
        self.num_labels = -1
        self.cat_codes_dict = None
        if name == 'c3':
            self.df = pd.read_csv(datasets_dict[name])[['comment_text', 'constructive_binary']]
            self.df.columns = ['text','label']
            self.text = self.df['text'].values
            self.labels = self.df['label'].apply(lambda v: int(v)).values
            self.num_labels = len(np.unique(self.labels))
        elif name == 'feedback':
            self.df = pd.read_csv(datasets_dict[name])[['discourse_text', 'discourse_type']]
            self.df.columns = ['text','label']
            self.df['label'] = self.df['label'].astype('category').cat.codes.values
            self.cat_codes_dict = dict( enumerate(self.df['label'].astype('category').cat.categories))
            self.text = self.df['text'].values
            self.labels = self.df['label'].apply(lambda v: int(v)).values
            self.num_labels = len(np.unique(self.labels))
        elif name == 'discourse':
            self.df = pd.read_csv(datasets_dict[name])
            self.text, self.labels, self.tree_depth = self.create_sequence_for_ddiscourse(org_to_fix, num_prev_posts=2)
            self.num_labels = 31
            
    def train_test_split(self, data_dict,  batch_size=16, validation_ratio=0.2):
        '''
        Create a train test split of the data. recieves
        a dictionary of the tokens, mask_ids and labels.
        :param data_dict: Dictionary - containing the token_ids, masks, and labels.
        :param batch_size: Integer - Size of batch.
        :param validation_ratio: Float - between 0-1, define the validation\test size.
        '''
        token_id = data_dict['token_ids']
        attention_masks = data_dict['attention_masks']
        labels = data_dict['labels']
        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size = validation_ratio,
            shuffle = True,
            stratify = labels)

        # Train and validation sets
        train_set = TensorDataset(token_id[train_idx], 
                              attention_masks[train_idx], 
                              labels[train_idx])

        val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])

        # Prepare DataLoader
        train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

        validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = batch_size
            )
        return train_dataloader, validation_dataloader
    
    def train_test_split_discourse(self, labels, tree_d, data_dict,  batch_size=16, validation_ratio=0.2):
        '''
        Create a train test split of the data. recieves
        a dictionary of the tokens, mask_ids and labels.
        Differs from the train_test_split method by enabling the splitting
        according to trees of discussion.
        :param data_dict: Dictionary - containing the token_ids, masks, and labels.
        :param batch_size: Integer - Size of batch.
        :param tree_d:  Integer - maximum tree depth.
        :param validation_ratio: Float - between 0-1, define the validation\test size.
        '''
        test_index = []
        for i in range(0, 81, 20):
            test_index.extend(list(range(sum(tree_d[0:i]), sum(tree_d[0:i+3]))))
        test_index = test_index
        train_index = [x for x in list(range(0, sum(tree_d))) if x not in test_index]
        token_id = data_dict['token_type_ids']
        attention_masks = data_dict['attention_mask']
        labels = labels
        
        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
            np.array(train_index),
            test_size = validation_ratio,
            shuffle = True
        )
        
        tokens_train = np.array([token_id[i] for i in train_idx])
        masks_train = np.array([attention_masks[i] for i in train_idx])
        labels_train = np.array([labels[i] for i in train_idx])
        
        tokens_val = np.array([token_id[i] for i in val_idx])
        masks_val = np.array([attention_masks[i] for i in val_idx])
        labels_val = np.array([labels[i] for i in val_idx])
        
        tokens_test = np.array([token_id[i] for i in test_index])
        masks_test = np.array([attention_masks[i] for i in test_index])
        labels_test = np.array([labels[i] for i in test_index])
        
        
        input_ids_train = torch.tensor(tokens_train)
        input_ids_test = torch.tensor(tokens_test)
        input_ids_val = torch.tensor(tokens_val)
        attention_masks_train = torch.tensor(masks_train)
        attention_masks_test = torch.tensor(masks_test)
        attention_masks_val = torch.tensor(masks_val)
        labels_train = torch.tensor(labels_train)
        labels_test = torch.tensor(labels_test)
        labels_val = torch.tensor(labels_val)


        # gpu
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # train
        input_ids_train = input_ids_train.to(device)
        attention_masks_train = attention_masks_train.to(device)
        labels_train = labels_train.to(device)
        # test
        input_ids_test = input_ids_test.to(device)
        attention_masks_test = attention_masks_test.to(device)
        labels_test = labels_test.to(device)
        # val
        input_ids_val = input_ids_val.to(device)
        attention_masks_val = attention_masks_val.to(device)
        labels_val = labels_val.to(device)
        
        # Train and validation sets
        train_set = TensorDataset(input_ids_train, 
                              attention_masks_train, 
                              labels_train)

        val_set = TensorDataset(input_ids_val, 
                            attention_masks_val, 
                            labels_val)
        
        test_set = TensorDataset(input_ids_test, 
                            attention_masks_test, 
                            labels_test)

        # Prepare DataLoader
        train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

        validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = batch_size
            )
        
        test_dataloader = DataLoader(
                test_set,
                sampler = SequentialSampler(test_set),
                batch_size = batch_size
            )
        return train_dataloader, validation_dataloader, test_dataloader
    
    
    def create_sequence_for_ddiscourse(self, org_to_fix, num_prev_posts=2):
        '''
        Method create the sequences required for the train_test split
        made on the discourse data. It creates the tress themselves
        and save them locally on the object.
        :param org_to_fix: dictionary containing the label and relevant wording which is similar
        :param num_prev_posts: Integer - number of previous posts to take into
        consideration.
        '''
        text_lst, labels, num_tree_comments = [], [], []
        curr_root_idx, comments = -1, 0
        labels_for_curr_tree = {}
        for idx, row in self.df.iterrows():
          parent = row['parent']
          if parent == -1:
            num_tree_comments.append(comments)
            comments = 0
            curr_root_idx = idx
            labels_for_curr_tree = {}
            continue
          text = row['text']
          labels_for_curr_tree[idx] = [org_to_fix[label] for label in self.df.columns[7:] if row[label] == 1]
          if parent == curr_root_idx:  # comment on root
              text_lst.append('previous tags: current post: ' + text)
              labels.append(list(self.df.iloc[idx:idx+1, 7:].values.flatten().tolist()))
              comments += 1
              continue
          else:
            prev_labels = []
            i = 0
            while i < num_prev_posts and parent != curr_root_idx:
              prev_labels.append(" ".join(labels_for_curr_tree[parent]))
              parent = self.df.loc[parent, 'parent']
              i += 1
          
            tags = "tag {}".format(' tag '.join(prev_labels))
            text_lst.append('previous tags: ' + tags + ', current post: ' + text)
            labels.append(list(self.df.iloc[idx:idx+1, 7:].values.flatten().tolist()))
            comments += 1
        num_tree_comments.append(comments)
        return text_lst, np.array(labels), np.array(num_tree_comments[1:])