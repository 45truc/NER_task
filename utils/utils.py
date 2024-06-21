import codecs
from skseq.sequences.label_dictionary import *
from skseq.sequences.sequence import *
from skseq.sequences.sequence_list import *
from os.path import dirname
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# for the bi-lstm
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

class NerCorpus(object):
    def __init__(self):
        # Word dictionary.
        self.word_dict = LabelDictionary()

        self.tag_dict = LabelDictionary()

        # Initialize sequence list.
        self.sequence_list = SequenceList(self.word_dict, self.tag_dict)

    def read_sequence_list(self, train_file,
                           max_sent_len=100):
        instance_list = self.read_instances(train_file,
                                            max_sent_len)

        seq_list = SequenceList(self.word_dict, self.tag_dict)

        for sent_x, sent_y in instance_list:
            seq_list.add_sequence(sent_x, sent_y,  self.word_dict, self.tag_dict)

        return seq_list

    def read_instances(self, file, max_sent_len):
   
        contents = codecs.open(file, "r", "utf-8")
        
        nr_sent = 0
        instances = []
        ex_x = []
        ex_y = []
        prev_sent_id = 0
        
        #Skip header
        next(contents)
        for line in contents:
            toks = line.split(',')
            #Some lines are weird
            if len(toks) <3:
                continue
            toks[-1] = toks[-1][:-1] #Get rid of /n
            #print(toks)
            sent_id = int(toks[0])
            
            #When a sentence finishes
            if sent_id > prev_sent_id:
                # print "sent n %i size %i"%(nr_sent,len(ex_x))
                if len(ex_x) < max_sent_len and len(ex_x) > 1:
                    # print "accept"
                    nr_sent += 1
                    instances.append([ex_x, ex_y])
                # else:
                #     if(len(ex_x) <= 1):
                #         print "refusing sentence of len 1"
                ex_x = []
                ex_y = []
                
                #Update previous sentence
                prev_sent_id = sent_id
          
            entity = toks[-1]
            word = toks[1]
            
            #The commas are misread sometimes
            if word == '"':
                word = ','

            #There is at least one empty entity
            if entity == '':
                entity = 'O'  
            #The entity type sometimes has an extra invisible character   
            if entity[-1] == '\r':
                entity = entity[:-1]

                      
            
            #Ignorando lo del mapping
            if word not in self.word_dict:
                self.word_dict.add(word)
            if entity not in self.tag_dict:
                self.tag_dict.add(entity)
            ex_x.append(word)
            ex_y.append(entity)
            
        return instances

def decode_pred(pred, corpus):
    rep = ""
    for sec in pred:
        for xi, yi in zip(sec.x, sec.y):
            rep += "%s/%s " % (list(corpus.word_dict.keys())[xi],
                               list(corpus.tag_dict.keys())[yi])
        rep += '\n'
    return rep

def Evaluate_metrics(sequences, sequences_predictions, corpus, save=False, plotname=None):
    '''
        Returns ACC. and F1 with the requested requieremetns.
    '''
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    y_true = np.concatenate([np.array(sequences[s].y) for s in range(len(sequences))])
    y_pred = np.concatenate([sequences_predictions[s].y for s in range(len(sequences_predictions))])
    
    mask = y_true!= corpus.tag_dict['O']
    y_pred_f = y_pred[mask]
    y_true_f = y_true[mask]

    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    fig.colorbar(cax)
    
    # Set axes labels and tick marks
    class_names  = np.array(list(corpus.tag_dict.keys()))[list(set(y_pred) | set(y_true))]
    ax.set_xticklabels([''] + list(class_names), rotation=45)
    ax.set_yticklabels([''] + list(class_names))
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Annotate each cell in the matrix with the numeric value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    if save:
        plt.savefig(plotname)    
    plt.show()
    
    # Acc. ignoring 'O', weighted f1, 
    return accuracy_score(y_true_f, y_pred_f), f1(y_true, y_pred)



###################### utils to analyze data #############################

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def analysis_tiny_dataset(df):
    # Calculate the number of sentences
    num_sentences = df['sentence_id'].nunique()

    # Identify the unique entities and their counts
    entity_tags = df[df['tags'] != 'O']['tags']
    entity_counts = Counter(entity_tags)

    # Identify the entities and their occurrences
    entities = df[df['tags'] != 'O'][['words', 'tags']]
    entity_list = entities.groupby('tags')['words'].apply(list).to_dict()

    # Print the analysis
    print(f'Number of sentences: {num_sentences}')
    print('Entity counts:')
    for entity, count in entity_counts.items():
        print(f'  {entity}: {count}')
    print('Entities and their occurrences:')
    for tag, words in entity_list.items():
        print(f'  {tag}: {words}')


def analysis_train_test_data(train_df, test_df):

    # Calculate the number of sentences
    num_sentences_train = train_df['sentence_id'].nunique()
    num_sentences_test = test_df['sentence_id'].nunique()
    print(f'Number of sentences in train: {num_sentences_train}')
    print(f'Number of sentences in test: {num_sentences_test}')

    # Count the tags in train and test datasets, ignoring 'O' tags
    train_entity_counts = Counter(train_df[train_df['tags'] != 'O']['tags'])
    test_entity_counts = Counter(test_df[test_df['tags'] != 'O']['tags'])

    # Sort entity counts in descending order for consistency in both datasets
    sorted_train_counts = dict(sorted(train_entity_counts.items(), key=lambda item: item[1], reverse=True))
    sorted_test_counts = {tag: test_entity_counts.get(tag, 0) for tag in sorted_train_counts.keys()}

    print('Entity counts in train:')
    for entity, count in sorted_train_counts.items():
        print(f'  {entity}: {count}')

    print('Entity counts in test:')
    for entity, count in sorted_test_counts.items():
        print(f'  {entity}: {count}')

    # Prepare data for plotting
    tags = list(sorted_train_counts.keys())
    train_counts = [sorted_train_counts[tag] for tag in tags]
    test_counts = [sorted_test_counts[tag] for tag in tags]

    # Define colors for train and test bars
    train_color = 'blue'
    test_color = 'green'

    # Width of the bars
    bar_width = 0.4

    # Positions of the bars on the x-axis
    r1 = range(len(tags))
    r2 = [x + bar_width for x in r1]

    # Plotting the combined histogram
    plt.figure(figsize=(14, 8))
    plt.bar(r1, train_counts, color=train_color, width=bar_width, edgecolor='grey', label='Train Data')
    plt.bar(r2, test_counts, color=test_color, width=bar_width, edgecolor='grey', label='Test Data')

    # Add labels and title
    plt.xlabel('Entity Tags', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Entity Tag Counts in Train and Test Data', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(tags))], tags, rotation=45)
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

###################### utils to analyze data #############################

##########################################################################
############################## BI-LSTM  ##################################
##########################################################################

# create list of sentences and tags
class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                     s["tags"].values.tolist())]
        self.grouped = self.data.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]

def encode_and_pad_sequences_bilstm(sentences, tags, word2idx, tag2idx, n_words, max_len=100):
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
    y = [torch.tensor(i, dtype=torch.long) for i in y]

    return X, y

def create_vocabulary_bilstm(df):
    words = list(set(df["words"].values))
    words.append("ENDPAD") # appending end/padding token
    n_words = len(words)

    tags = list(set(df["tags"].values))
    n_tags = len(tags)

    word2idx = {w: i for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}

    return word2idx, tag2idx, n_words, n_tags, tags

class NERDataset(Dataset):
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.tags[idx], dtype=torch.long)

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, padding_idx, embedding_dim=100, hidden_dim=256, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=2)
        return tag_scores

# Function to calculate accuracy
def calculate_accuracy(preds, labels):
    preds = torch.argmax(preds, dim=2).flatten()
    labels = labels.flatten()
    return (preds == labels).float().mean()

# Evaluate the model
def evaluate_bilstm(model, dataloader, loss_function, n_tags):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    total_batches = len(dataloader)
    if total_batches == 0:
        return float('inf'), 0.0  # Return large loss and zero accuracy if no batches

    with torch.no_grad():
        for sentences, tags in dataloader:
            tag_scores = model(sentences)
            loss = loss_function(tag_scores.view(-1, n_tags), tags.view(-1))
            eval_loss += loss.item()
            eval_accuracy += calculate_accuracy(tag_scores, tags).item()
    return eval_loss / len(dataloader), eval_accuracy / len(dataloader)

def create_class_weights(tag_column):
    """
    Creates class weights inversely proportional to the frequency of each tag.

    Args:
        tag_column (list of list of str): List containing lists of tags for each sentence.

    Returns:
        torch.Tensor: Tensor containing the computed weights for each tag.
        list: List of unique tags in the order they appear in the weight tensor.
    """

    # Get tag counts
    tag_counts = {}
    for tag in tag_column:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1

    # Total number of tags
    total_tags = sum(tag_counts.values())

    # Compute weights (inversely proportional to the frequency)
    tag_weights = {tag: total_tags / count for tag, count in tag_counts.items()}

    # Normalize weights
    max_weight = max(tag_weights.values())
    tag_weights = {tag: weight / max_weight for tag, weight in tag_weights.items()}

    # List of unique tags
    unique_tags = list(tag_weights.keys())

    # Convert tag weights to a list, with each index corresponding to the tag index
    weights = torch.tensor([tag_weights[tag] for tag in unique_tags], dtype=torch.float)

    return weights, unique_tags

def train_bilstm(train_loader, val_loader, model, loss_function, optimizer, n_epochs, n_tags):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0

        for sentences, tags in train_loader:
            optimizer.zero_grad()
            tag_scores = model(sentences)
            batch_size, seq_len, _ = tag_scores.shape

            loss = loss_function(tag_scores.view(-1, tag_scores.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(tag_scores, tags).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)

        val_loss, val_accuracy = evaluate_bilstm(model, val_loader, loss_function, n_tags)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(avg_train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_metrics_bilstm(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

def get_predictions_and_labels(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentences, tags in dataloader:
            tag_scores = model(sentences)
            preds = torch.argmax(tag_scores, dim=2).cpu().numpy()
            labels = tags.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return all_preds, all_labels

# Calculate accuracy without 'O'
def calculate_accuracy_without_o(preds, labels, tag2idx):
    preds = torch.argmax(preds, dim=2).flatten()
    labels = labels.flatten()
    mask = labels != tag2idx["O"]
    return (preds[mask] == labels[mask]).float().mean()
    
def evaluate_bilstm_tiny_test(model, dataloader, loss_function, n_tags, tag2idx):
    model.eval()
    eval_loss = 0
    eval_accuracy = 0
    eval_accuracy_without_o = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sentences, tags in dataloader:
            tag_scores = model(sentences)
            loss = loss_function(tag_scores.view(-1, n_tags), tags.view(-1))
            eval_loss += loss.item()
            eval_accuracy += calculate_accuracy(tag_scores, tags).item()
            eval_accuracy_without_o += calculate_accuracy_without_o(tag_scores, tags, tag2idx).item()

            # Store predictions and labels for F1 score calculation
            preds = torch.argmax(tag_scores, dim=2).flatten().cpu().numpy()
            labels = tags.flatten().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=list(tag2idx.values()))

    return eval_loss / len(dataloader), eval_accuracy / len(dataloader), eval_accuracy_without_o / len(dataloader), f1

def tags_to_sentences(sentences, tags, idx2word, idx2tag):
    """
    Convert token indices and tag indices back to sentences with predicted tags.

    Args:
    - sentences (list of list of str): List of tokenized sentences.
    - tags (list of list of int): List of predicted tag indices for each sentence.
    - idx2word (dict): Mapping from token index to token word.
    - idx2tag (dict): Mapping from tag index to tag name.

    Returns:
    - list of tuple: List of tuples where each tuple contains a sentence and its predicted tags.
    """
    tagged_sentences = []

    for sentence, tag_indices in zip(sentences, tags):
        tagged_sentence = []
        for token_idx, tag_idx in zip(sentence, tag_indices):
            token_word = token_idx[0]  # Using the word directly from the input sentence
            tag_name = idx2tag.get(tag_idx, "O")  # Handle unknown tags
            tagged_sentence.append((token_word, tag_name))
        tagged_sentences.append(tagged_sentence)

    return tagged_sentences

def plot_confusion_matrix_bilstm(y_true, y_pred, idx2tag, title='Confusion Matrix', save=False, plotname='confusion_matrix.png'):
    """
    Plot and save the confusion matrix.

    Args:
    - y_true (list of int): List of true tag indices.
    - y_pred (list of int): List of predicted tag indices.
    - idx2tag (dict): Mapping from tag index to tag name.
    - title (str): Title of the plot.
    - save (bool): Whether to save the plot.
    - plotname (str): Filename to save the plot.
    """
    # Flatten the lists of tags
    y_true_flat = [tag for sentence in y_true for tag in sentence]
    y_pred_flat = [tag for sentence in y_pred for tag in sentence]

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title(title, pad=20)
    fig.colorbar(cax)

    # Set axes labels and tick marks
    class_names = [idx2tag[i] for i in range(len(idx2tag))]
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Annotate each cell in the matrix with the numeric value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    # Save the plot if required
    if save:
        plt.savefig(plotname)

    plt.show()
