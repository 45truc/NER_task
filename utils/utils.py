import codecs
from skseq.sequences.label_dictionary import *
from skseq.sequences.sequence import *
from skseq.sequences.sequence_list import *
from os.path import dirname
import numpy as np
from sklearn.metrics import f1_score


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

def Evaluate_metrics(sequences, sequences_predictions, corpus):
    '''
        Returns ACC. and F1 with the requested requieremetns.
    '''
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    def evaluate_corpus(sequences, sequences_predictions, corpus):
        '''
            Evaluate classification accuracy at corpus level, comparing with
            gold standard. Modified for ignoring class O.
    
        '''
        total = 0.0
        correct = 0.0
        for i, sequence in enumerate(sequences):
            pred = sequences_predictions[i]
            for j, y_hat in enumerate(pred.y):
                #Ignore class O 
                if sequence.y[j] == corpus.tag_dict['O']:
                    continue
                    
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
        return correct / total

    y_true = np.concatenate([np.array(sequences[s].y) for s in range(len(sequences))])
    y_pred = np.concatenate([sequences_predictions[s].y for s in range(len(sequences_predictions))])

    # Acc. ignoring 'O', weighted f1, 
    return evaluate_corpus(sequences, sequences_predictions, corpus), f1(y_true, y_pred)



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