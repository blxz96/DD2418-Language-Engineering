import os
import math
import random
import nltk
import numpy as np
import argparse
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
"""
Python implementation of the Glove training algorithm from the article by Pennington, Socher and Manning (2014).

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class Glove :

    # Mapping from words to IDs.
    word2id = defaultdict(lambda: None)

    # Mapping from IDs to words.
    id2word = defaultdict(lambda: None)

    # Mapping from focus words to neighbours to counts (called X 
    # to be consistent with the notation in the Glove paper).
    # X is a co-occurrence matrix
    X = defaultdict(lambda: defaultdict(int))

    # Mapping from word IDs to (focus) word vectors. (called w_vector 
    # to be consistent with the notation in the Glove paper).
    w_vector = defaultdict(lambda: None)

    # Mapping from word IDs to (context) word vectors (called w_tilde_vector
    # to be consistent with the notation in the Glove paper)
    w_tilde_vector = defaultdict(lambda: None)

    # Mapping from word IDs to gradients of (focus) word vectors.
    w_vector_grad = defaultdict(lambda: None)

    # Mapping from word IDs to gradients of (context) word vectors.
    w_tilde_vector_grad = defaultdict(lambda: None)

    # The ID of the latest encountered new word.
    latest_new_word = -1

    # Dimension of word vectors.
    dimension = 10

    # Left context window size.
    left_window_size = 2

    # Right context window size.
    right_window_size = 2

    # The local context window.
    window = []

    # The ID of the current focus word.
    focus_word_id = -1

    # The current token number.
    current_token_number = 0

    # Cutoff for gradient descent.
    epsilon = 0.01

    # Learning rate.
    learning_rate = 0.05 #0.001

    # Neighbours
    nbrs = None

    # Final word vectors. Each word vector is the sum of the context vector
    # and the focus vector for that word. The vectors are best represented
    # as a numpy array of size (number of words, vector dimension) in order
    # to use the sklearn NearestNeighbor library.
    vector = None
    
    # Initializes the local context window
    def __init__( self, left_window_size, right_window_size ) :
        self.window = [-1 for i in range(left_window_size + right_window_size)]
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size


    #--------------------------------------------------------------------------
    #
    #  Methods for processing all files and computing all counts

    # Initializes the necessary information for a word.

    def init_word( self, word ) :

        self.latest_new_word += 1

        # This word has never been encountered before. Init all necessary
        # data structures.
        self.id2word[self.latest_new_word] = word   # Mapping from IDs to words.
        self.word2id[word] = self.latest_new_word   # Mapping from words to IDs.

        # Initialize vectors with random numbers in [-0.5,0.5].
        w = [random.random()-0.5 for i in range(self.dimension)]
        self.w_vector[self.latest_new_word] = w # Mapping from word IDs to (focus) word vectors.
        w_tilde = [random.random()-0.5 for i in range(self.dimension)]
        self.w_tilde_vector[self.latest_new_word] = w_tilde # Mapping from word IDs to (context) word vectors

        self.w_vector_grad[self.latest_new_word] = 1
        self.w_tilde_vector_grad[self.latest_new_word] = 1
        return self.latest_new_word



    # Slides in a new word in the local context window
    #
    # The local context is a list of length left_window_size+right_window_size.
    # Suppose the left window size and the right window size are both 2.
    # Consider a sequence
    #
    # ... this  is  a  piece  of  text ...
    #               ^
    #           Focus word
    #
    # Then the local context is a list [id(this),id(is),id(piece),id(of)],
    # where id(this) is the wordId for 'this', etc.
    #
    # Now if we slide the window one step, we get
    #
    # ... is  a  piece  of  text ...
    #              ^
    #         New focus word
    #
    # and the new context window is [id(is),id(a),id(of),id(text)].
    #
    def slide_window( self, word_l2, word_l1, word_r1, word_r2) :
        
        # YOUR CODE HERE

        context = []
        context.append(word_l2)
        context.append(word_l1)
        context.append(word_r1)
        context.append(word_r2)
        return context


    # Update counts based on the local context window
    def update_counts( self, focustoken, contexttoken) :
        
        # YOUR CODE HERE

        # If focus token is not yet a key in X, register it
        if focustoken not in self.X:    
            self.X[focustoken][contexttoken] = 1
        else:
            # If context token wasn't registered as a neighbout of focus token, register it
            if contexttoken not in self.X[focustoken]:
                self.X[focustoken][contexttoken] = 1
            # If context token is already registered as a neighbour of the focus word, increment the no of times it appear as a neighbour    
            else:
                self.X[focustoken][contexttoken] += 1
    

    # Handles one token in the text
    def process_token(self, word, word_l2, word_l1, word_r1, word_r2) :
        # YOUR CODE HERE
        if word not in self.word2id:
            self.init_word(word)
        if (word_l2 != -1) and (word_l2 not in self.word2id):
            self.init_word(word_l2)
        if (word_l1 != -1) and (word_l1 not in self.word2id):
            self.init_word(word_l1)
        if (word_r2 != -1) and (word_r2 not in self.word2id):
            self.init_word(word_r2)
        if (word_r1 != -1) and (word_r1 not in self.word2id):
            self.init_word(word_r1)

        context_words = self.slide_window(word_l2, word_l1, word_r1, word_r2)
        for context_word in context_words:
            if context_word != -1:
                self.update_counts(word, context_word)

    # This function recursively processes all files in a directory
    def process_files( self, file_or_dir ) :
        if os.path.isdir( file_or_dir ) :
            for root,dirs,files in os.walk( file_or_dir ) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            stream = open( file_or_dir, mode='r', encoding='utf-8', errors='ignore' )
            text = stream.read()
            try :
                tokens = nltk.word_tokenize(text) 
            except LookupError :
                nltk.download('punkt')
                tokens = nltk.word_tokenize(text)
            cnt = -1
            tokenslen = len(tokens)
            for token in tokens:
                cnt += 1
                # token located in middle of a line
                if (cnt > 1) and (cnt < tokenslen - 2):
                    self.process_token(token, tokens[cnt - 2], tokens[cnt - 1], tokens[cnt + 1], tokens[cnt + 2])
                elif (cnt == 0):
                # first token in line
                    self.process_token(token, -1, -1, tokens[cnt + 1], tokens[cnt + 2])
                elif (cnt == 1):
                # second token in line
                    self.process_token(token, -1, tokens[cnt - 1], tokens[cnt + 1], tokens[cnt + 2])
                # third token in line
                elif (cnt == tokenslen - 2):
                    self.process_token(token, tokens[cnt - 2], tokens[cnt - 1], tokens[cnt + 1], -1)
                # last token in line
                elif (cnt == tokenslen - 1):
                    self.process_token(token, tokens[cnt - 2], tokens[cnt - 1], -1, -1)
                
                self.current_token_number += 1
                """ if self.current_token_number % 100 == 0 :
                    print( 'Processed ' + str(self.current_token_number) + ' tokens' ) """


        
    #
    #  Methods for processing all files and computing all counts
    #
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #   Loss function, gradient descent, etc.
    #

    # The mysterious "f" function from the article
    def f( self, count ) :
        if count<100 :
            ratio = count/100.0
            return math.pow( ratio, 0.75 )
        return 1.0
    

    # Gradient descent
    def train( self ) :
        # YOUR CODE HERE

        iter = 0
        global_cost= prev_global_cost = 100

        while (prev_global_cost- global_cost) > self.epsilon or iter <= 1:
            iter +=1 
            print("Iter: {}, global_cost: {}, change: {}".format(iter,global_cost,prev_global_cost-global_cost))
            prev_global_cost = global_cost
            global_cost = 0

            for word_i in self.X:
                tmp = self.X[word_i] # contains all neighbour of word i
                for word_j in tmp:
                    i = self.word2id[word_i]
                    j = self.word2id[word_j]

                    # xij is the co-occurrence
                    xij = self.X[word_i][word_j]
                    
                    # computing the inner component of cost function, which is used in both overall cost calculation and in gradient descent
                    # J' = w_i^T.w_j + - log(X_{ij})
                    cost_inner = sum([x*y for x,y in zip(self.w_vector[i],self.w_tilde_vector[j])]) - math.log(xij)
                    
                    # J = f(X_{ij}) (J')^2
                    cost = self.f(xij) * (cost_inner ** 2)
                    global_cost += cost


                    # Computing the gradients
                    # grad = [k * (self.f(xij) * cost_inner) for k in self.w_tilde_vector[j]]
                    # tilde_grad = [k * (self.f(xij) * cost_inner) for k in self.w_vector[i]]

                    self.w_vector_grad[i] = self.f(xij) * np.array(self.w_tilde_vector[j]).dot(cost_inner)
                    self.w_tilde_vector_grad[j] = self.f(xij) * np.array(self.w_vector[i]).dot(cost_inner)


                    # Performing gradient descent
                    # self.w_vector[i] -= np.array([k * self.learning_rate for k in grad])
                    # self.w_tilde_vector[j] -= np.array([k * self.learning_rate for k in tilde_grad])
                    self.w_vector[i] -= self.learning_rate * self.w_vector_grad[i]
                    self.w_tilde_vector[j] -= self.learning_rate * self.w_tilde_vector_grad[j]


                    #self.w_vector[i] -= ([k * self.learning_rate for k in grad] / np.sqrt(self.w_vector_grad[i]))
                    #self.w_tilde_vector[j] -= ([k * self.learning_rate for k in tilde_grad] / np.sqrt(self.w_tilde_vector_grad[j]))

                    #self.w_vector_grad[i] += np.square(grad)
                    #self.w_tilde_vector_grad[j] += np.square(tilde_grad)
            
            

        self.print_word_vectors_to_file("Vector10d.txt")


    
    
    
    ##
    ## @brief      Function returning k nearest neighbors with distances for each word in `words`
    ## 
    ## We suggest using nearest neighbors implementation from scikit-learn 
    ## (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
    ## carefully their documentation regarding the parameters passed to the algorithm.
    ## 
    ## To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
    ## "Harry" and "Potter" using cosine distance (which can be computed as 1 - cosine similarity). 
    ## For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='cosine')`.
    ## The output of the function would then be the following list of lists of tuples (LLT)
    ## (all words and distances are just example values):
    ## \verbatim
    ## [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
    ##  [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
    ## \endverbatim
    ## The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
    ## list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
    ## The tuples are sorted either by descending similarity or by ascending distance.
    ##
    ## @param      words   A list of words, for which the nearest neighbors should be returned
    ## @param      k       A number of nearest neighbors to be returned
    ## @param      metric  A similarity/distance metric to be used (defaults to cosine distance)
    ##
    ## @return     A list of list of tuples in the format specified in the function description
    ##
    def find_nearest(self, words, metric='cosine'):
        # YOUR CODE HERE
        all_words = []
        
        word_get, x = self.load_words_and_embeddings("vectors10d.txt")

        n = NearestNeighbors(n_neighbors=5, metric=metric).fit(x)

        for word in words:

            index = word_get.index(word)
            distance, indices_of_closest_words = n.kneighbors([x[index]])

            # Now we have indices of closest words
            closest_words = []
            for i in range(len(indices_of_closest_words[0])):
                index = indices_of_closest_words[0][i]
                dist = distance[0][i]
                w = word_get[index]
                closest_words.append((w, dist))
            all_words.append(closest_words)


        return all_words


    def load_words_and_embeddings(self,fname): 
    
        try:
            with open(fname, 'r') as f:
                vocab, dim = (int(a) for a in next(f).split())

                words, X =  [] , np.zeros((vocab, dim))

                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    X[i] = list(map(float, parts[1:])) # vectors of H dimensions
                    words.append(word)
        
        except:
            print("Error: failing to load the model to the file")
        return words, X


    ##
    ## @brief      Trains word embeddings and enters the interactive loop, where you can 
    ##             enter a word and get a list of k nearest neighours.
    ##        
    def train_and_persist(self):
        if os.path.exists("vectors10d.txt"):
            pass
        else:
            self.train()
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text)
            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

       
    #
    #  End of loss function, gradient descent, etc.
    #
    #-------------------------------------------------------

    #-------------------------------------------------------
    #
    #  I/O
    #

    def print_word_vectors_to_file( self, filename ) :
        with open(filename, 'w') as f:
            f.write("{} {}\n".format(len(self.id2word.keys()), self.dimension)) #
            for id in self.id2word.keys() :
                f.write('{} '.format( self.id2word[id] ))
                # Add the focus vector and the context vector for each word
                for i in list(np.add(self.w_vector[id], self.w_tilde_vector[id])) :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()
        


def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glove trainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='The files used in the training.')
    # parser.add_argument('--output', '-o', type=str, required=True, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--dimension', '-d', type=int, default='300', help='Desired vector dimension')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')

    arguments = parser.parse_args()  
    
    glove = Glove(arguments.left_window_size, arguments.right_window_size)
    glove.dimension = arguments.dimension
    glove.process_files( arguments.file )
    glove.train_and_persist()

if __name__ == '__main__' :
    main()    

