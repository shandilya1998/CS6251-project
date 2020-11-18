import pickle
import copy

import pandas as pd

import spacy
nlp = spacy.load('en_core_web_sm')

import string

TAG2INDEX = '../tag2index.pickle'
ADJ_LIST = 'ED_adjacency_list.pickle'
WORDLIST = 'ED_wordlist.pickle'
INDEX2WORD = 'ED_index2word.pickle'
ED_DATA = 'mod_dataset.csv'

data = pd.read_csv(ED_DATA)
data = data.dropna()
tag2index = pickle.load(open(TAG2INDEX, 'rb'))
wordlist = pickle.load(open(WORDLIST, 'rb'))
index2word = pickle.load(open(INDEX2WORD, 'rb'))
adj = pickle.load(open(ADJ_LIST, 'rb'))


def get_pos_seq(sent):
    index = len(wordlist.keys())
    sent = nlp(sent)
    pos = []
    for token in sent:
        if token.lemma_ not in wordlist.keys():
            if token.text in string.punctuation or token.text == '\'s' or token.text == '' or token.text == ' ': 
                continue
            if token.is_stop:
                continue
            wordlist[token.lemma_] = index
            index2word[index] = token.lemma_
            adj[index] = []
            index += 1
        pos.append(str(tag2index[token.tag_]))  
    return ' '.join(pos)

def get_index_seq(sent):
    sent = nlp(sent)
    lst = []
    for token in sent:
        if token.lemma_ not in wordlist.keys():
            lst.append(0)
        else:
            lst.append(wordlist[token.lemma_])
    return ' '.join([str(item) for item in lst])

def update_pos_seq(seq):
    seq = [str(int(val)+1) for val in seq.split()]
    return ' '.join(seq)

def logged_apply(g, func, *args, **kwargs):
    """
        func - function to apply to the dataframe
        *args, **kwargs are the arguments to func
        The method applies the function to all the elements of the dataframe and shows progress
    """
    step_percentage = 100. / len(g)
    import sys
    print('\rApply progress:   0%', end = "")
    sys.stdout.flush()

    def logging_decorator(func):
        def wrapper(*args, **kwargs):
            progress = wrapper.count * step_percentage
            #sys.stdout.write('\033[D \033[D' * 4 + format(progress, '3.0f') + '%')
            print('\rApply progress:   {prog}%'.format(prog=progress), end = "")
            sys.stdout.flush()
            wrapper.count += 1
            return func(*args, **kwargs)
        wrapper.count = 0
        return wrapper

    logged_func = logging_decorator(func)
    res = g.apply(logged_func, *args, **kwargs)
    print('\rApply progress:   1 00%', end = "")
    sys.stdout.flush()
    print('\n')
    return res

print('columns of original data')
print(data.columns)
print('length of original wordlist:')
print(len(wordlist.keys()))
print('columns of modified data:')
#data['pos_seq'] = logged_apply(data['sentence'], get_pos_seq)
#data['word index seq'] = logged_apply(data['sentence'], get_index_seq)
data['pos_seq'] = logged_apply(data['pos_seq'], update_pos_seq)
print(data.columns)
print('length of modified wordlist:')
print(len(wordlist.keys()))
print('length of modified index2word list:')
print(len(index2word.keys()))
print('length of modified adjacency list:')
print(len(adj.keys()))
data.to_csv('updated_mod_dataset.csv', index = False)
#pickle.dump(wordlist, open('ED_wordlist.pickle', 'wb'))
#pickle.dump(index2word, open('ED_index2word.pickle', 'wb'))
#pickle.dump(adj, open('ED_adjacency_list.pickle', 'wb'))
