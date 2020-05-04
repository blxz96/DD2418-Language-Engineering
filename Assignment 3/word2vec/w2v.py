import os
import time
import argparse
import string
from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import random

from tqdm import tqdm


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2020 by Dmytro Kalpakchi.
"""

class Word2Vec(object):
    def __init__(self, filenames, dimension=300, window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True):
        """
        Constructs a new instance.
        
        :param      filenames:      A list of filenames to be used as the training material
        :param      dimension:      The dimensionality of the word embeddings
        :param      window_size:    The size of the context window
        :param      nsample:        The number of negative samples to be chosen
        :param      learning_rate:  The learning rate
        :param      epochs:         A number of epochs
        :param      use_corrected:  An indicator of whether a corrected unigram distribution should be used
        """
        self.__pad_word = '<pad>'
        self.__sources = filenames
        self.__H = dimension
        self.__lws = window_size
        self.__rws = window_size
        self.__C = self.__lws + self.__rws
        self.__init_lr = learning_rate
        self.__lr = learning_rate
        self.__nsample = nsample
        self.__epochs = epochs
        self.__nbrs = None
        self.__use_corrected = use_corrected
        self.__use_lr_scheduling = use_lr_scheduling

        

    def init_params(self, W, w2i, i2w, U, u2i, i2u): # if focus == True, init param for W, else init param of U


        self.__W = W        # numpy array of size (V*H)
        self.__w2i = w2i    # dictionary {word:indices}
        self.__i2w = i2w    # list containing words

        self.__U = U        # numpy array of size (V*H)
        self.__u2i = u2i    # dictionary {word:indices}
        self.__i2u = i2u

        
    @property
    def vocab_size(self):
        return self.__V
        

    def clean_line(self, line):
        """
        The function takes a line from the text file as a string,
        removes all the punctuation and digits from it and returns
        all words in the cleaned line as a list
        
        :param      line:  The line
        :type       line:  str
        """
        cleaned_string = ''
        cleaned_line = []
        for element in line:
            if element not in string.punctuation and element not in string.digits:
                cleaned_string += element
        cleaned_line = cleaned_string.split()
        return cleaned_line


    def text_gen(self):
        """
        A generator function providing one cleaned line at a time

        This function reads every file from the source files line by
        line and returns a special kind of iterator, called
        generator, returning one cleaned line a time.

        If you are unfamiliar with Python's generators, please read
        more following these links:
        - https://docs.python.org/3/howto/functional.html#generators
        - https://wiki.python.org/moin/Generators
        """
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    def get_context(self, sent, i):
        """
        Returns the context of the word `sent[i]` as a list of word indices
        
        :param      sent:  The sentence
        :type       sent:  list
        :param      i:     Index of the focus word in the sentence
        :type       i:     int
        """
        
        # REPLACE WITH YOUR CODE

        lws = self.__lws
        rws = self.__rws

        # Retrieiving the context indices in the sentence
        sent_length = len(sent)
        context_indices_in_sent = list(range(i-lws,i)) + list(range(i+1,i+rws+1))
        # left context of first word and right context of the last word is empty
        context_indices_in_sent = [item for item in context_indices_in_sent if item >= 0 and item < sent_length] 

        # Getting the respective context words
        context_words = []
        for index in context_indices_in_sent:
            context_words.append(sent[index])

        # Getting the respective context indices in the entire corpus
        context_indices = []
        for word in context_words:
            context_indices.append(self.w2i[word])

        return context_indices


    def skipgram_data(self):
        """
        A function preparing data for a skipgram word2vec model in 3 stages:
        1) Build the maps between words and indexes and vice versa
        2) Calculate the unigram distribution and corrected unigram distribution
           (the latter according to Mikolov's article)
        3) Return a tuple containing two lists:
            a) list of focus words
            b) list of respective context words
        """
        
        # REPLACE WITH YOUR CODE
        
        # Building the maps between words and indexes and vice versa
            # Need to iterate through all the words by iterating through the lines in the generator
            # Recommend to create 2 mappings:
            # w2i : converting strings to unique indices
            # i2w : converting indices back to words
            # Additional consideration : Unigram count for calculation of unigram distribution

        self.w2i , self.i2w, self.unigram_count, index = {}, {}, {}, 0

        for line in self.text_gen():
            for word in line:
                # if word not in w2i mapping
                if word not in self.w2i:
                    self.w2i[word] = index
                    self.i2w[index] = word
                    self.unigram_count[word] = 1
                    index += 1
                else: 
                    self.unigram_count[word] += 1
                

        # Calculating the unigram distribution and corrected unigram distribution

        # Unigram distribution 
        c = Counter(self.unigram_count)
        total_words = sum(c.values()) 
        self.unigram_dist = {}
        for k,v in self.unigram_count.items():
            self.unigram_dist[k] = v / total_words

        # Corrected Unigram distribution
        denominator = sum(list(map(lambda x: x**0.75, self.unigram_dist.values())))
        self.corrected_unigram_dist = {}
        for k,v in self.unigram_dist.items():
            self.corrected_unigram_dist[k] = (v**0.75) / denominator


        # Return a tuple containing 2 lists: 
            # a) a list of focus words
            # b) list of respective context indices (will be a list of lists)
        
        focus_words = []
        context_indices = []
        for line in self.text_gen():
            for i, word in enumerate(line):
                if word not in focus_words: 
                    focus_words.append(word)
                    context_indices.append(self.get_context(line,i))
                if word in focus_words:
                    focus_index = focus_words.index(word)
                    context_indices[focus_index].extend(self.get_context(line,i))


        return focus_words, context_indices


    def sigmoid(self, x):
        """
        Computes a sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, xb, pos):
        """
        Sample a `number` of negatives examples with the words in `xb` and `pos` words being
        in the taboo list, i.e. those should be replaced if sampled.
        
        :param      number:     The number of negative examples to be sampled
        :type       number:     int
        :param      xb:         The index of the current focus word
        :type       xb:         int
        :param      pos:        The index of the current positive example
        :type       pos:        int
        """
        
        # REPLACE WITH YOUR CODE

        # Note: For every positive word, need to sample n negative words
        # This function passed in number, index of current focus word and index of current positive word
        
        # can be toggled from __init__
        used_corrected = self.__use_corrected

        if used_corrected == False:
            words = list(self.unigram_dist.keys())
            probability = list(self.unigram_dist.values())
        else:
            words = list(self.corrected_unigram_dist.keys())
            probability = list(self.corrected_unigram_dist.values())

        # initialise to track successful sampling
        count = 0 
        negative_samples_indices = [] # store the indices

        while count < number:
            sample_word = random.choices(population=words , weights=probability)[0]
            sample_index = self.w2i[sample_word]
            if sample_index != xb and sample_index != pos and sample_index not in negative_samples_indices:
                negative_samples_indices.append(sample_index)
                count += 1

        return negative_samples_indices


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        x, t = self.skipgram_data()
        N = len(x) 
        self.__V = N
        self.__i2w = x
        self.__w2i = {}
        for id, w in enumerate(self.__i2w):
            self.__w2i[w] = id

        print("Dataset contains {} datapoints".format(N))

        # Initialisation with uniform distribution
        # self.__W = np.random.uniform(size = (N, self.__H))
        # self.__U = np.random.uniform(size = (N, self.__H))

        # Initialisation with normal distribution
        self.__W = np.random.normal(size = (N, self.__H)) # focus word vector for each word form a matrix
        self.__U = np.random.normal(size = (N, self.__H)) # context word vector for each word form a matrix

        # can be adjusted and toggled from __init__
        starting_learning_rate = self.__lr
        use_lr_scheduling = self.__use_lr_scheduling
        learning_rate = starting_learning_rate

        # np.vectorize can be used so that we can use our sigmoid function on vectors
        sigmoid_v = np.vectorize(self.sigmoid) 
        
        for ep in range(self.__epochs):
            for i in tqdm(range(N)):

                if use_lr_scheduling == True:
                    if learning_rate < starting_learning_rate * 0.0001:
                        learning_rate = starting_learning_rate * 0.0001
                    else:
                        learning_rate = starting_learning_rate * (1- (ep*N + i)/(self.__epochs*N + 1))

                        
                ######################################################################### 
                # Algorithm: 
                #
                # We will use i to get the focus word, x[i]
                # For every focus word x[i], initialise gradient_focus = 0 and loop through the context_indices t[i]
                # For every positive context word, pos_id = t[i][j] and perform A and B
                #
                # ########################################################################
                # A:
                #
                #   1. Accumulate gradient of loss function w.r.t focus word v 
                #
                #   gradient_focus += self.__U[pos_id].dot(sigmoid_v(self.__U[pos_id].T.dot(self.__W[i]))-1)
                # 
                #   2. Perform gradient descent of loss function w.r.t the positive word
                #
                #   self.__U[pos_id] -= learning_rate * self.__W[i].dot(sigmoid_V(self.__U[pos_id].T.dot(self.__W[i]))-1)
                # 
                ##########################################################################
                # B:
                #
                #  Perform negative sampling
                #  
                #  negative_samples = self.negative_sampling(self.__nsample, i, pos_id)
                #
                #  for neg_id in negative_samples:
                #  
                #  1. Accumulate gradient of loss function w.r.t focus word v 
                #         
                #  gradient_focus += self.__U[neg_id].dot(sigmoid_v(self.__U[neg_id].T.dot(self.__W[i])))
                # 
                #  2. Perform gradient descent of loss function w.r.t the negative word
                #
                #  self.__U[neg_id] -= learning_rate * self.__W[i].dot(sigmoid_V(self.__U[neg_id].T.dot(self.__W[i])))
                #
                #############################################################################
                # After completing A and B,
                #
                # Perform gradient descent of loss function w.r.t focus word v 
                #
                # self.__W[i] -= learning_rate * gradient_focus
                #
                #############################################################################
                
                gradient_focus = 0
                focus_index = i
                context_indices = t[i]

                # For every positive index
                for pos_id in context_indices:
                    
                    # Accumulate gradient of loss function w.r.t focus word v 
                    gradient_focus += self.__U[pos_id].dot(sigmoid_v(self.__U[pos_id].T.dot(self.__W[i]))-1)
                    # Perform gradient descent of loss function w.r.t the positive word
                    self.__U[pos_id] -= learning_rate * self.__W[i].dot(sigmoid_v(self.__U[pos_id].T.dot(self.__W[i]))-1)
                    
                    # Get a list of negative samples
                    negative_samples = self.negative_sampling(self.__nsample, i, pos_id)

                    # For every negative index
                    for neg_id in negative_samples:
                        # Accumulate gradient of loss function w.r.t focus word v 
                        gradient_focus += self.__U[neg_id].dot(sigmoid_v(self.__U[neg_id].T.dot(self.__W[i])))
                        # Perform gradient descent of loss function w.r.t the negative word
                        self.__U[neg_id] -= learning_rate * self.__W[i].dot(sigmoid_v(self.__U[neg_id].T.dot(self.__W[i])))

                # Perform gradient descent of loss function w.r.t focus word v 
                self.__W[i] -= learning_rate * gradient_focus
                

        

    def find_nearest(self, words, metric):
        """
        Function returning k nearest neighbors with distances for each word in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn 
        (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
        carefully their documentation regarding the parameters passed to the algorithm.
    
        To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
        "Harry" and "Potter" using some distance metric `m`. 
        For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.
        The output of the function would then be the following list of lists of tuples (LLT)
        (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
        
        The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
        list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
        The tuples are sorted either by descending similarity or by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        
        # REPLACE WITH YOUR CODE
        
        all_words = []

        n = NearestNeighbors(n_neighbors=5, metric=metric).fit(self.__U) #U

        #print('self.__i2w[1000]:{}'.format(self.__i2w[1000]))
        #print('self.__i2u[1000]:{}'.format(self.__i2u[1000]))


        for word in words:
            id = self.__w2i[word]
            context_vector = self.__W[id] #W

            distance, indices_of_closest_words = n.kneighbors([context_vector])

            # Now we have indices of closest words
            closest_words = []
            for i in range(len(indices_of_closest_words[0])):
                index = indices_of_closest_words[0][i]
                dist = distance[0][i]
                w = self.__i2w[index] # doesn't matter i2w or i2u since just a word
                closest_words.append((w, dist))
            all_words.append(closest_words)


        return all_words



    def write_to_file(self,name,matrix):
        """
        Write the model to a file `w2v.txt`
        """
        try:
            with open("{}.txt".format(name), 'w') as f:    
                # to store target word matrix
                W = matrix
                f.write("{} {}\n".format(self.__V, self.__H))
                for i, w in enumerate(self.__i2w): # shouldn't it be w2i? unless i2w is a list of words not a dict
                    f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
            
        except:
            print("Error: failing to write model to the file")


    @classmethod
    def load(cls, fname_W, fname_U): # if true, initialise parameter of focus matrix W ; if false, initialise parameter of context matrix U
        """
        Load the word2vec model from a file `fname`
        """
        w2v = None
    
        try:
            fW = open(fname_W, 'r')
            V, H = (int(a) for a in next(fW).split())
            w2v = cls([], dimension=H)

            W, i2w, w2i = np.zeros((V, H)), [], {}

            for i, line in enumerate(fW):
                parts = line.split()
                word = parts[0].strip()
                w2i[word] = i
                W[i] = list(map(float, parts[1:])) # vectors of H dimensions
                i2w.append(word)
    
            
            fU = open(fname_U, 'r')
            V, H = (int(a) for a in next(fU).split())
            w2v = cls([], dimension=H)

            U, i2u, u2i = np.zeros((V, H)), [], {}

            for i, line in enumerate(fU):
                parts = line.split()
                word = parts[0].strip()
                u2i[word] = i
                U[i] = list(map(float, parts[1:])) # vectors of H dimensions
                i2u.append(word)
                
            w2v.init_params(W, w2i, i2w, U, u2i, i2u)


            """ with open(fname_U, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                w2v = cls([], dimension=H)

                W, i2w, w2i = np.zeros((V, H)), [], {}

                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    w2i[word] = i
                    W[i] = list(map(float, parts[1:])) # vectors of H dimensions
                    i2w.append(word)
                
                w2v.init_params(W, w2i, i2w, focus= True)
        """
        except:
            print("Error: failing to load the model to the file")
        return w2v



    def interact(self):
        """
        Interactive mode allowing a user to enter a number of space-separated words and
        get nearest 5 nearest neighbors for every word in the vector space
        """
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text, 'cosine')

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')


    def train_and_persist(self):
        """
        Main function call to train word embeddings and being able to input
        example words interactively
        """
        self.train()
        self.write_to_file('w2v_W',self.__W)
        self.write_to_file('w2v_U',self.__U)
        self.interact()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('-t', '--text', default='harry_potter_1.txt',
                        help='Comma-separated source text files to be trained on')
    #parser.add_argument('-s', '--save', default='w2v.txt', help='Filename where word vectors are saved')
    parser.add_argument('-s_W', '--save_W', default='w2v_W.txt', help='Filename where word vectors are saved') #
    parser.add_argument('-s_U', '--save_U', default='w2v_U.txt', help='Filename where word vectors are saved') #
    parser.add_argument('-d', '--dimension', default=300, help='Dimensionality of word vectors') 
    parser.add_argument('-ws', '--window-size', default=3, help='Context window size')
    parser.add_argument('-neg', '--negative_sample', default=10, help='Number of negative samples')
    parser.add_argument('-lr', '--learning-rate', default=0.025, help='Initial learning rate')
    parser.add_argument('-e', '--epochs', default=5, help='Number of epochs')
    parser.add_argument('-uc', '--use-corrected', action='store_true', default=True,
                        help="""An indicator of whether to use a corrected unigram distribution
                                for negative sampling""")
    parser.add_argument('-ulrs', '--use-learning-rate-scheduling', action='store_true', default=True,
                        help="An indicator of whether using the learning rate scheduling")
    args = parser.parse_args()

    if os.path.exists(args.save_W) and os.path.exists(args.save_U): #
        w2v = Word2Vec.load(args.save_W, args.save_U)  #
        if w2v:
            w2v.interact()
    else:
        # Create a new instance
        w2v = Word2Vec(
            args.text.split(','), dimension=args.dimension, window_size=args.window_size,
            nsample=args.negative_sample, learning_rate=args.learning_rate, epochs=args.epochs,
            use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
        )
        w2v.train_and_persist()
