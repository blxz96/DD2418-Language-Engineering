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
    learning_rate = 0.001

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
    #


    # Initializes the necessary information for a word.

    def init_word( self, word ) :

        self.latest_new_word += 1

        # This word has never been encountered before. Init all necessary
        # data structures.
        self.id2word[self.latest_new_word] = word
        self.word2id[word] = self.latest_new_word

        # Initialize vectors with random numbers in [-0.5,0.5].
        w = [random.random()-0.5 for i in range(self.dimension)]
        self.w_vector[self.latest_new_word] = w
        w_tilde = [random.random()-0.5 for i in range(self.dimension)]
        self.w_tilde_vector[self.latest_new_word] = w_tilde
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
    def slide_window( self, id ) :
        # YOUR CODE HERE
        pass


    # Update counts based on the local context window
    def update_counts( self ) :
        # YOUR CODE HERE
        pass
    

    # Handles one token in the text
    def process_token( self, word ) :
        # YOUR CODE HERE
        pass


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
            for token in tokens:
                self.process_token(token.lower())
                self.current_token_number += 1
                if self.current_token_number % 100000 == 0 :
                    print( 'Processed ' + str(self.current_token_number) + ' tokens' )

        
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
    

    # The loss function
    def loss( self ) :
        # YOUR CODE HERE
        return 0.0    


    # Compute all gradients of both focus and context word vectors
    def computeGradients( self ) :
        # YOUR CODE HERE
        pass


    # Gradient descent
    def fit( self ) :
        # YOUR CODE HERE
        pass

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
        return [None]

    ##
    ## @brief      Trains word embeddings and enters the interactive loop, where you can 
    ##             enter a word and get a list of k nearest neighours.
    ##        
    def train_and_persist(self):
        self.fit()
        text = input('> ')
        while text != 'exit':
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
    parser.add_argument('--output', '-o', type=str, required=True, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--dimension', '-d', type=int, default='10', help='Desired vector dimension')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')

    arguments = parser.parse_args()  
    
    glove = Glove(arguments.left_window_size, arguments.right_window_size)
    glove.dimension = arguments.dimension
    glove.process_files( arguments.file )
    glove.train_and_persist()
    glove.print_word_vectors_to_file( arguments.output )

    

if __name__ == '__main__' :
    main()    

