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
                #Ignore class O mmm ns si está con número o con el nombre tengo qeu verlo
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