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


ROOT_DIR = os.getcwd()
DATA_DIR = 'data'
datasets_dict = {
    'c3': os.path.join(ROOT_DIR, DATA_DIR, 'archive', 'C3_anonymized.csv'),
    'feedback': os.path.join(ROOT_DIR, DATA_DIR, 'feedback-prize-effectiveness', 'train.csv'),
    'discourse': os.path.join(ROOT_DIR, DATA_DIR, 'course', 'annotated_trees_101.csv')
}

labels_lst_fix = ['Aggressive', 'Agree But', 'Agree To Disagree', 'Alternative', 'Answer', 'Attack Validity', 'Bad', 'Clarification',	
              'Complaint', 'Convergence', 'Counter Argument', 'Critical Question', 'Direct No', 'Double Voicing', 'Extension', 
              'Irrelevance', 'Moderation', 'Negative Transformation', 'Nitpicking', 'No Reason Disagreement', 'Personal', 'Positive', 
              'Repetition', 'Rephrase Attack', 'Request Clarification', 'Ridicule', 'Sarcasm', 'Softening', 'Sources', 'Viable Transformation', 'Weakening Qualifiers']

labels_lst_org = ['Aggressive', 'AgreeBut', 'AgreeToDisagree', 'Alternative', 'Answer', 'AttackValidity', 'BAD', 'Clarification', 'Complaint', 'Convergence', 'CounterArgument', 
                  'CriticalQuestion', 'DirectNo', 'DoubleVoicing', 'Extension', 'Irrelevance', 'Moderation','NegTransformation', 'Nitpicking', 'NoReasonDisagreement', 'Personal', 
                  'Positive', 'Repetition', 'RephraseAttack', 'RequestClarification', 'Ridicule', 'Sarcasm', 'Softening', 'Sources', 'ViableTransformation', 'WQualifiers']
org_to_fix = {labels_lst_org[i]: labels_lst_fix[i] for i in range(len(labels_lst_fix))}



def print_rand_sentence(text):
    '''Displays the tokens and respective IDs of a random text sample'''
    index = random.randint(0, len(text)-1)
    table = np.array([tokenizer.tokenize(text[index]), 
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text[index]))]).T
    print(tabulate(table,
                 headers = ['Tokens', 'Token IDs'],
                 tablefmt = 'fancy_grid'))
    
    
    
def print_rand_sentence_encoding(text):
    '''Displays tokens, token IDs and attention mask of a random text sample'''
    index = random.randint(0, len(text) - 1)
    tokens = tokenizer.tokenize(tokenizer.decode(token_id[index]))
    token_ids = [i.numpy() for i in token_id[index]]
    attention = [i.numpy() for i in attention_masks[index]]

    table = np.array([tokens, token_ids, attention]).T
    print(tabulate(table, 
                 headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                 tablefmt = 'fancy_grid'))
    
    
def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity