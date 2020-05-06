import os
import argparse
import time
import string
import numpy as np
from halo import Halo
import random
from sklearn.neighbors import NearestNeighbors


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2018 by Dmytro Kalpakchi and Johan Boye.
"""


##
## @brief      Class for creating word vectors using Random Indexing technique.
## @author     Dmytro Kalpakchi <dmytroka@kth.se>
## @date       November 2018
##
class RandomIndexing(object):

    ##
    ## @brief      Object initializer Initializes the Random Indexing algorithm
    ##             with the necessary hyperparameters and the textfiles that
    ##             will serve as corpora for generating word vectors
    ##             
    ## The `self.__vocab` instance variable is initialized as a Python's set. If you're unfamiliar with sets, please
    ## follow this link to find out more: https://docs.python.org/3/tutorial/datastructures.html#sets.
    ##
    ## @param      self               The RI object itself (is omitted in the descriptions of other functions)
    ## @param      filenames          The filenames of the text files (7 Harry
    ##                                Potter books) that will serve as corpora
    ##                                for generating word vectors. Stored in an
    ##                                instance variable self.__sources.
    ## @param      dimension          The dimension of the word vectors (both
    ##                                context and random). Stored in an
    ##                                instance variable self.__dim.
    ## @param      non_zero           The number of non zero elements in a
    ##                                random word vector. Stored in an
    ##                                instance variable self.__non_zero.
    ## @param      non_zero_values    The possible values of non zero elements
    ##                                used when initializing a random word. Stored in an
    ##                                instance variable self.__non_zero_values.
    ##                                vector
    ## @param      left_window_size   The left window size. Stored in an
    ##                                instance variable self__lws.
    ## @param      right_window_size  The right window size. Stored in an
    ##                                instance variable self__rws.
    ##
    def __init__(self, filenames, dimension=300, non_zero=15, non_zero_values=list([-1, 1]), left_window_size=3, right_window_size=3):
        self.__sources = filenames
        self.__vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        # there is a list call in a non_zero_values just for Doxygen documentation purposes
        # otherwise, it gets documented as "[-1,"
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None
        

    ##
    ## @brief      A function cleaning the line from punctuation and digits
    ##
    ##             The function takes a line from the text file as a string,
    ##             removes all the punctuation and digits from it and returns
    ##             all words in the cleaned line.
    ##
    ## @param      line  The line of the text file to be cleaned
    ##
    ## @return     A list of words in a cleaned line
    ##
    def clean_line(self, line):
        cleaned_string = ''
        cleaned_line = []
        for element in line:
            if element not in string.punctuation and element not in string.digits:
                cleaned_string += element
        cleaned_line = cleaned_string.split()
        return cleaned_line


    ##
    ## @brief      A generator function providing one cleaned line at a time
    ##
    ##             This function reads every file from the source files line by
    ##             line and returns a special kind of iterator, called
    ##             generator, returning one cleaned line a time.
    ##
    ##             If you are unfamiliar with Python's generators, please read
    ##             more following these links:
    ## - https://docs.python.org/3/howto/functional.html#generators
    ## - https://wiki.python.org/moin/Generators
    ##
    ## @return     A generator yielding one cleaned line at a time
    ##
    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    ##
    ## @brief      Build vocabulary of words from the provided text files.
    ##
    ##             Goes through all the cleaned lines and adds each word of the
    ##             line to a vocabulary stored in a variable `self.__vocab`. The
    ##             words, stored in the vocabulary, should be unique.
    ##             
    ##             **Note**: this function is where the first pass through all files is made
    ##             (using the `text_gen` function)
    ##
    def build_vocabulary(self):
        # YOUR CODE HERE
        for line in self.text_gen():
            for word in line:
                if word not in self.__vocab:
                    self.__vocab.add(word)
        self.write_vocabulary()

    ##
    ## @brief      Get the size of the vocabulary
    ##
    ## @return     The size of the vocabulary
    ##
    @property
    def vocabulary_size(self):
        return len(self.__vocab)


    ##
    ## @brief      Creates word embeddings using Random Indexing.
    ## 
    ## The function stores the created word embeddings (or so called context vectors) in `self.__cv`.
    ## Random vectors used to create word embeddings are stored in `self.__rv`.
    ## 
    ## Context vectors are created by looping through each cleaned line and updating the context
    ## vectors following the Random Indexing approach, i.e. using the words in the sliding window.
    ## The size of the sliding window is governed by two instance variables `self.__lws` (left window size)
    ## and `self.__rws` (right window size).
    ## 
    ## For instance, let's consider a sentence:
    ##      I really like programming assignments.
    ## Let's assume that the left part of the sliding window has size 1 (`self.__lws` = 1) and the right
    ## part has size 2 (`self.__rws` = 2). Then, the sliding windows will be constructed as follows:
    ## \verbatim
    ##      I really like programming assignments.
    ##      ^   r      r
    ##      I really like programming assignments.
    ##      l   ^      r       r
    ##      I really like programming assignments.
    ##          l      ^       r           r
    ##      I really like programming assignments.
    ##                 l       ^           r
    ##      I really like programming assignments.
    ##                         l           ^
    ## \endverbatim
    ## where "^" denotes the word we're currently at, "l" denotes the words in the left part of the
    ## sliding window and "r" denotes the words in the right part of the sliding window.
    ## 
    ## Implementation tips:
    ## - make sure to understand how generators work! Refer to the documentation of a `text_gen` function
    ##   for more description.
    ## - the easiest way is to make `self.__cv` and `self.__rv` dictionaries with keys being words (as strings)
    ##   and values being the context vectors.
    ## 
    ## **Note**: this function is where the second pass through all files is made (using the `text_gen` function).
    ##         The first one was done when calling `build_vocabulary` function. This might not the most
    ##         efficient solution from the time perspective, but it's quite efficient from the memory
    ##         perspective, given that we are using generators, which are lazily evaluated, instead of
    ##         keeping all the cleaned lines in memory as a gigantic list.
    ##
    def create_word_vectors(self):
        # YOUR CODE HERE

        # Initialise the dictionary to store the random vectors and context vectors respectively.
        # Keys will be word, and values will be the embeddings
        self.__rv, self.__cv = {}, {}

        for line in self.text_gen():

            size = len(line)
            # if size == 1 , only focus word exists in that line
            if size == 0 or size == 1: 
                continue

            # Start from the first word in the line and iterate through the line
            current_index = 0 
            while current_index < size:
                context_words = [] 
                
                # to store the context words within the right window size
                for i in range(1, self.__rws + 1):
                    try:
                        context_words.append(line[current_index + i])
                    except IndexError: # No more word to the right.
                        break
                
                # to store the context words within the left window size
                for i in range(1, self.__lws + 1):
                    if current_index - i < 0: # No more word to the left.
                        break 
                    else:
                        context_words.append(line[current_index - i])

                # Check if current word has a context vector assigned to it.
                try:
                    context_vector = self.__cv[line[current_index]]
                except KeyError:
                    # If no context vector exists, initialize and create 0 vector for it.
                    self.__cv[line[current_index]] = np.zeros(self.__dim)

                # Add random vectors of context words to the context vector of the current word.
                for word in context_words:
                    # Check if context word already has an random vector assigned to it.
                    try:
                        random_vector = self.__rv[word]
                    except KeyError:
                        # If index vector does not already exist, create it.
                        non_zero_indices = random.sample(range(0, self.__dim), self.__non_zero) # Indices of where to place non-zero values in the index vector
                        random_vector = np.zeros(self.__dim)
                        for i in range(self.__non_zero):
                            index_of_non_zero = non_zero_indices[i] # Index of where a non_zero value in current iteration should be in the index vector
                            non_zero_value = self.__non_zero_values[random.randrange(2)] # Either a (-1) or a 1. This is the non-zero value we put in the index vector at the given index.
                            random_vector[index_of_non_zero] = non_zero_value
                        self.__rv[word] = random_vector

                    self.__cv[line[current_index]] = np.add(self.__cv[line[current_index]], self.__rv[word])

                current_index += 1

        self.__words = list(self.__cv) # List of all context words
        self.__matrix = list(self.__cv.values()) # List of all context vectors, i.e embeddings of context words


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
    
    def find_nearest(self, words, k=5, metric='cosine'):
        # YOUR CODE HERE
        all_words = []

        # self.__matrix is our X
        n = NearestNeighbors(n_neighbors=k, metric=metric).fit(self.__matrix)

        for word in words:
            context_vector = self.get_word_vector(word)
            if context_vector is None: 
                print("The word '{}' does not exist in the corpus!".format(word))
                continue 

            distance, indices_of_closest_words = n.kneighbors([context_vector])

            # Now we have indices of closest words
            closest_words = []
            for i in range(len(indices_of_closest_words[0])):
                index = indices_of_closest_words[0][i]
                dist = distance[0][i]
                w = self.__words[index]
                closest_words.append((w, dist))
            all_words.append(closest_words)


        return all_words


    ##
    ## @brief      Returns a vector for the word obtained after Random Indexing is finished
    ##
    ## @param      word  The word as a string
    ##
    ## @return     The word vector if the word exists in the vocabulary and None otherwise.
    ##
    def get_word_vector(self, word):
        # YOUR CODE HERE
        try:
            return self.__cv[word]
        except KeyError:
            return None


    ##
    ## @brief      Checks if the vocabulary is written as a text file
    ##
    ## @return     True if the vocabulary file is written and False otherwise
    ##
    def vocab_exists(self):
        return os.path.exists('vocab.txt')


    ##
    ## @brief      Reads a vocabulary from a text file having one word per line.
    ##
    ## @return     True if the vocabulary exists was read from the file and False otherwise
    ##             (note that exception handling in case the reading failes is not implemented)
    ##
    def read_vocabulary(self):
        vocab_exists = self.vocab_exists()
        if vocab_exists:
            with open('vocab.txt') as f:
                for line in f:
                    self.__vocab.add(line.strip())
        self.__i2w = list(self.__vocab)
        return vocab_exists


    ##
    ## @brief      Writes a vocabulary as a text file containing one word from the vocabulary per row. 
    ##
    def write_vocabulary(self):
        with open('vocab.txt', 'w') as f:
            for w in self.__vocab:
                f.write('{}\n'.format(w))


    ##
    ## @brief      Main function call to train word embeddings
    ## 
    ## If vocabulary file exists, it reads the vocabulary from the file (to speed up the program),
    ## otherwise, it builds a vocabulary by reading and cleaning all the Harry Potter books and
    ## storing unique words.
    ## 
    ## After the vocabulary is created/read, the word embeddings are created using Random Indexing.
    ##
    def train(self):
        spinner = Halo(spinner='arrow3')

        if self.vocab_exists():
            spinner.start(text="Reading vocabulary...")
            start = time.time()
            self.read_vocabulary()
            # Commented out below to allow Q3 to run since ri will only be initialised if _name_ = "main"
            spinner.succeed(text="Read vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        else:
            spinner.start(text="Building vocabulary...")
            start = time.time()
            self.build_vocabulary()
            # Comment out below to allow Q3 to run since ri will only be initialised if _name_ = "main"
            spinner.succeed(text="Built vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        
        spinner.start(text="Creating vectors using random indexing...")
        start = time.time()
        self.create_word_vectors()
        spinner.succeed("Created random indexing vectors in {}s.".format(round(time.time() - start, 2)))

        spinner.succeed(text="Execution is finished! Please enter words of interest (separated by space):")


    ##
    ## @brief      Trains word embeddings and enters the interactive loop, where you can 
    ##             enter a word and get a list of k nearest neighours.
    ##
    def train_and_persist(self):
        self.train()
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text)

            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    # For Question 3
    def getWords(self):
        return self.__words

    def getMatrix(self):
        return self.__matrix

    def writeWordsandMatrix(self):
        f = open("WordsandMatrix.txt", "w")
        f.write("{} {}\n".format(len(self.__words), self.__dim))
        for i in range(len(self.__matrix)):
            f.write(self.__words[i] + " " + " ".join(map(str, self.__matrix[i])) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
    parser.add_argument('-c', '--cleaning', action='store_true', default=False)
    parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
    args = parser.parse_args()

    if args.force_vocabulary:
        os.remove('vocab.txt')

    if args.cleaning:
        ri = RandomIndexing(['example.txt'])
        with open(args.cleaned_output, 'w') as f:
            for part in ri.text_gen():
                f.write("{}\n".format(" ".join(part)))
    else:
        dir_name = "data"
        filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

        # For comparison for Qn 2 w2v
        # filenames = ["data\\Harry Potter 1 - Sorcerer's Stone.txt"]


        ri = RandomIndexing(filenames)
        ri.train_and_persist()
        ri.writeWordsandMatrix()
