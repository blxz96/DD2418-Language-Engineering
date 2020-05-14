#!/usr/bin/env python
# coding: utf-8
import argparse
import string
import codecs
import csv
from tqdm import tqdm
from terminaltables import AsciiTable
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from GRU import GRU2

PADDING_WORD = '<PAD>'
UNKNOWN_WORD = '<UNK>'
CHARS = ['<UNK>', '<space>', '’', '—'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)


def load_glove_embeddings(embedding_file, padding_idx=0, padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD):
    """
    The function to load GloVe word embeddings
    
    :param      embedding_file:  The name of the txt file containing GloVe word embeddings
    :type       embedding_file:  str
    :param      padding_idx:     The index, where to insert padding and unknown words
    :type       padding_idx:     int
    :param      padding_word:    The symbol used as a padding word
    :type       padding_word:    str
    :param      unknown_word:    The symbol used for unknown words
    :type       unknown_word:    str
    
    :returns:   (a vocabulary size, vector dimensionality, embedding matrix, mapping from words to indices)
    :rtype:     a 4-tuple
    """
    word2index, embeddings, N = {}, [], 0
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            data = line.split()
            word = data[0]
            vec = [float(x) for x in data[1:]]
            embeddings.append(vec)
            word2index[word] = N
            N += 1
    D = len(embeddings[0]) # 50 for glove 50d
    
    if padding_idx is not None and type(padding_idx) is int:
        embeddings.insert(padding_idx, [0]*D)           # for '<PAD>'
        embeddings.insert(padding_idx + 1, [-1]*D)      # for '<UNK>'
        for word in word2index:
            if word2index[word] >= padding_idx:
                word2index[word] += 2 # originally was only += 1 but I think it is wrong
        word2index[padding_word] = padding_idx          # for '<PAD>'
        word2index[unknown_word] = padding_idx + 1      # for '<UNK>'
        # Add in N to account for <PAD> and <UNK>
        N += 2
                
    return N, D, np.array(embeddings, dtype=np.float32), word2index


class NERDataset(Dataset):
    """
    A class loading NER dataset from a CSV file to be used as an input to PyTorch DataLoader.
    """
    def __init__(self, filename):
        reader = csv.reader(codecs.open(filename, encoding='ascii', errors='ignore'), delimiter=',')
        
        self.sentences = [] # will keep a record of a list of sentences
        self.labels = []    # will keep a record of a list of labels
        
        sentence, labels = [], []
        for row in reader:
            # e.g of row : 
            # ['Sentence: 4544', 'Demonstrators', 'NNS', 'O']
            # ['', 'chanting', 'VBG', 'O']
            # ['', '"', '``', 'O']
            # ['', 'Death', 'NN', 'O']
            # ['', 'to', 'TO', 'O']
            # ['', 'America', 'NNP', 'B-geo']
            # ['', '"', '``', 'I-geo']
            # ...
            # ['', '.', '.', 'O']
            if row:
                if row[0].strip(): # e.g Sentence: 4544, i.e Start of a new sentence
                    if sentence and labels: # if sentence and labels are not empty, i.e. a new sentence, then append previous sentence and labels to self.sentences and self.labels
                        # e.g of sentence: ['Demonstrators', 'chanting', '"', 'Death', 'to', 'America', '"', 'marched', 'through', 'streets', 'Wednesday', ',', 'smashing', 'cars', ',', 'damaging', 'shops', 'and', 'throwing', 'stones', 'at', 'U.S.', 'troops', '.']
                        # e.g of labels:   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                        self.sentences.append(sentence) 
                        self.labels.append(labels)      
                    sentence = [row[1].strip()] # If first word in the sentence, create a list for the new sentence, e.g. ['Demonstrators']
                    labels = [self.__bio2int(row[3].strip())] # If first label in the sentence, create a list for the new sentence, e.g. [0]
                else: 
                    sentence.append(row[1].strip()) # If not first word in the sentence, append to sentence, e.g ['Demonstrators','chanting']
                    labels.append(self.__bio2int(row[3].strip())) # If not first label in the sentence, append to sentence, e.g [0,0]
                
    def __bio2int(self, x):
        return 0 if x == 'O' else 1
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]


class PadSequence:
    """
    A callable used to merge a list of samples to form a padded mini-batch of Tensor
    """
    def __call__(self, batch, pad_data=PADDING_WORD, pad_labels=0):
        batch_data, batch_labels = zip(*batch)
        max_len = max(map(len, batch_labels))
        padded_data = [[b[i] if i < len(b) else pad_data for i in range(max_len)] for b in batch_data]
        padded_labels = [[l[i] if i < len(l) else pad_labels for i in range(max_len)] for l in batch_labels]
        return padded_data, padded_labels


class NERClassifier(nn.Module):
    def __init__(self, word_emb_file, char_emb_size=16, char_hidden_size=25, word_hidden_size=100,
                 padding_word=PADDING_WORD, unknown_word=UNKNOWN_WORD, char_map=CHARS):
        """
        Constructs a new instance.
        
        :param      word_emb_file:     The filename of the file with pre-trained word embeddings
        :type       word_emb_file:     str
        :param      char_emb_size:     The character embedding size
        :type       char_emb_size:     int
        :param      char_hidden_size:  The character-level BiRNN hidden size
        :type       char_hidden_size:  int
        :param      word_hidden_size:  The word-level BiRNN hidden size
        :type       word_hidden_size:  int
        :param      padding_word:      A token used to pad the batch to equal-sized tensor
        :type       padding_word:      str
        :param      unknown_word:      A token used for the out-of-vocabulary words 
        :type       unknown_word:      str
        :param      char_map:          A list of characters to be considered
        :type       char_map:          list
        """
        super(NERClassifier, self).__init__()
        self.padding_word = padding_word
        self.unknown_word = unknown_word
        self.char_emb_size = char_emb_size
        self.char_hidden_size = char_hidden_size
        self.word_hidden_size = word_hidden_size
        
        self.c2i = {c: i for i, c in enumerate(char_map)} # Mapping CHARS to index
        # print(self.c2i)
        self.char_emb = nn.Embedding(len(char_map), char_emb_size, padding_idx=0)  # 98 16 0
        # self.char_emb has 98 different character and each embedding has a length of 16, if index 0, will be padded with 0.
        
        vocabulary_size, self.word_emb_size, embeddings, self.w2i = load_glove_embeddings(
            word_emb_file, padding_word=self.padding_word, unknown_word=self.unknown_word
        )
        # print(vocabulary_size,len(embeddings),len(self.w2i)) # 400002 400002 400002
        self.word_emb = nn.Embedding(vocabulary_size, self.word_emb_size) # 400002 x 50 
        self.word_emb.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=False)
        # print(self.word_emb.weight)
        
        self.char_birnn = GRU2(self.char_emb_size, self.char_hidden_size, bidirectional=True) # input size: 16, hidden size: 25
        self.word_birnn = GRU2(
            self.word_emb_size + 2 * self.char_hidden_size, # input size  # 50 + 2 * 25 = 100
            self.word_hidden_size,                          # hidden size # 100
            bidirectional=True
        )
        
        # Binary classification - 0 if not part of the name, 1 if a name
        # self.final_pred = nn.Linear(2 * self.word_hidden_size, 2) # size of each input sample = 2 x 100 = 200; size of each output sample = 2
        self.final_pred = nn.Linear(self.word_hidden_size, 2)
        
    def forward(self, x):
        """
        Performs a forward pass of a NER classifier
        Takes as input a 2D list `x` of dimensionality (B, T),
        where B is the batch size;
              T is the max sentence length in the batch (the sentences with a smaller length are already padded with a special token <PAD>)
        
        Returns logits, i.e. the output of the last linear layer before applying softmax.

        :param      x:    A batch of sentences
        :type       x:    list of strings
        """
        
        # YOUR CODE HERE

        # x is a batch of sentence in the form of a list of list. The inner list is a sentence padded at the back.

        B, T = len(x), len(x[0]) # 128,53
     
        # Finding out the max_word_length
        max_word_length = 0

        for i in range(B):
            for j in range(T):
                word = x[i][j]
                if word != self.padding_word: # '<PAD>'
                    word_length = len(word)
                    if word_length > max_word_length:
                        max_word_length = word_length # 15
                else:
                    break

        # Splitting every word to character
        # char_id_lists will be a list containing 128 x 53 = 6784 inner lists. 
        # Each inner list correspond to a word in a sentence.
        # Elements of each inner list will be the char_ids that made up the word.

        char_lists, char_id_lists = [] , []
        for i in range(B):
            sentence = x[i]
            for j in range(T):
                word = sentence[j]
                if word != self.padding_word: # '<PAD>'
                    char_list = list(word)
                else:
                    char_list = list()
            
                # create the corresponding char_id_list
                char_id_list = [self.c2i[char] for char in char_list]
                while len(char_id_list) < max_word_length:
                    char_list.append('<PAD>')
                    char_id_list.append(0) 
                    
                char_lists.append(char_list)
                char_id_lists.append(char_id_list)
        # print(len(char_id_lists)) # Verification: 6784
        
        # Converting list to tensor so that it can be used as input to self.char_emb()
        input = torch.LongTensor(char_id_lists)
        char_embeddings = self.char_emb(input)

        # Since char_id_lists has already collapsed the first 2 dimension, 
        # i.e. batch-size(B) and max-sentence-length(T), into a dimension with size (B x T),
        # it has 3 dimension and there is no need to reshape. 
        # print(char_embeddings.size()) # torch.Size([6784, 15, 16])

        # Here, we use our implementation of the bidirectional GRU in GRU.py to get 
        # the last hidden states for the forward and backward cells of the character-level BiRNN
        outputs, h_fw, h_bw = self.char_birnn(char_embeddings)

        concatenated_char_hidden_states = torch.cat((h_fw,h_bw),axis=1)
        # print(concatenated_char_hidden_states.size()) # torch.Size([6784, 50])

        # Reshaping it back to a 3D tensor
        # This will be our character-level word vectors! 
        concatenated_char_hidden_states = concatenated_char_hidden_states.reshape(B,T,-1)
        # print(concatenated_char_hidden_states.size()) # torch.Size([128, 53, 50])

        # Now we retrieve the respective GloVe vectors
        word_indices_lists = []
        for sentence in x:
            word_indices = [self.w2i[word] if word in self.w2i else 1 for word in sentence] # 1 is the w2i index for unknown
            word_indices_lists.append(word_indices)
        # print(len(word_indices_lists)) # 128
        
        # Converting list to tensor so that it can be used as input to self.word_emb()
        input = torch.LongTensor(word_indices_lists)
        word_embeddings = self.word_emb(input)
        # print(word_embeddings.size())  # torch.Size([128, 53, 50])

        # Contenate GloVe vectors with character-level word vectors
        concatenated_word_char = torch.cat((word_embeddings,concatenated_char_hidden_states),axis=2)

        # Here, we use our implementation of the bidirectional GRU in GRU.py to get 
        # the outputs of the word-level BiRNN
        outputs, h_fw, h_bw = self.word_birnn(concatenated_word_char)
        # print(outputs.size()) # torch.Size([128, 53, 100])

        return self.final_pred(outputs)
      



#
# MAIN SECTION
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-tr', '--train', default='data/ner_training.csv',
                        help='A comma-separated training file')
    parser.add_argument('-t', '--test', default='data/ner_test.csv',
                        help='A comma-separated test file')
    parser.add_argument('-wv', '--word-vectors', default='glove.6B.50d.txt',
                        help='A txt file with word vectors')
    parser.add_argument('-lr', '--learning-rate', default=0.02, help='A learning rate') #0.002
    parser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    args = parser.parse_args()

    training_data = NERDataset(args.train)
    training_loader = DataLoader(training_data, batch_size=128, collate_fn=PadSequence())

    ner = NERClassifier(args.word_vectors)

    optimizer = optim.Adam(ner.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(args.epochs):
        ner.train()
        for x, y in tqdm(training_loader, desc="Epoch {}".format(epoch + 1)):
            optimizer.zero_grad()
            logits = ner(x) # will invoke the forward function 
            logits_shape = logits.shape
            
            loss = criterion(logits.reshape(-1, logits_shape[2]), torch.tensor(y).reshape(-1,))
            loss.backward()
        
            clip_grad_norm_(ner.parameters(), 5)
            optimizer.step()
    
    # Evaluation
    ner.eval()
    confusion_matrix = [[0, 0],
                        [0, 0]]
    test_data = NERDataset(args.test)
    for x, y in test_data:
        pred = torch.argmax(ner([x]), dim=-1).detach().numpy().reshape(-1,)
        y = np.array(y)
        tp = np.sum(pred[y == 1])
        tn = np.sum(1 - pred[y == 0])
        fp = np.sum(1 - y[pred == 1])
        fn = np.sum(y[pred == 0])

        confusion_matrix[0][0] += tn
        confusion_matrix[1][1] += tp
        confusion_matrix[0][1] += fp
        confusion_matrix[1][0] += fn

    table = [['', 'Predicted no name', 'Predicted name'],
             ['Real no name', confusion_matrix[0][0], confusion_matrix[0][1]],
             ['Real name', confusion_matrix[1][0], confusion_matrix[1][1]]]

    t = AsciiTable(table)
    print(t.table)
    print("Accuracy: {}".format(
        round((confusion_matrix[0][0] + confusion_matrix[1][1]) / np.sum(confusion_matrix), 4))
    )
