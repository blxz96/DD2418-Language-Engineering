import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                
                # REUSE YOUR CODE FROM BigramTester.py here
                # Algorithm:
                # For the next V lines, read in an identifier, a token and the number of times the token appears in the corpus
                # no. of V lines is represented as self.unique_words

                for i in range(self.unique_words):
                    id , token , no_times = f.readline().strip().split(' ')
                    id, noTimes = map(int, (id, no_times))
                    self.index[token] = id
                    self.word[id] = token
                    self.unigram_count[token] = no_times

                # The length of the rest of the lines - 1 = number of bigram
                # This is because the last line in the .txt file is just '-1'

                restLines = f.readlines() # will return a list
                no_bigrams = len(restLines) - 1
                for i in range(no_bigrams):
                    token1 , token2 , bigram_logProb = restLines[i].strip().split(' ')
                    token1 , token2 = map(int, (token1, token2))
                    bigram_logProb = float(bigram_logProb)
                    self.bigram_prob[(token1,token2)] = bigram_logProb

                return True


        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and following the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        # Note: With log-prob, higher log-prob = more likely.
        
        # Algorithm:
        # while < n words generated:
        # for the iteration starting with word w,
        # convert token w to to its index
        # then look at the cases where index of w is the first token in the bigram
        # Of those cases, randomly generate a bigram according to the probability distribution
        # We get the index of the 2nd token to be generated from the bigram
        # From it,we find the token to be generated and print it
        # Repeat this until the n words are generated

        # "In the rare case event where all bigrams probabilities from the last generated word are zero,
        # pick any word at random using a uniform distribution " =
        # If index of word does not appear as first token, then randomly pick any word.
         
        wordcount = 0 
        print(w, end=' ')
        wordcount += 1
        id = self.index[w]
        while wordcount < n:
            
            # cases where id represents the first token in bigram and their respective probabilites (since we take the exponential of ln)
            cases = [item for item in self.bigram_prob if item[0] == id]
            cases_prob = [math.exp(self.bigram_prob[i]) for i in cases]

            if len(cases) != 0:
                # the bigram generated according to the probability distribution
                bigram = random.choices(population = cases, weights = cases_prob)[0]
                # the next id and token to be generated
                next_id = bigram[1]
                next_token = self.word[next_id]
            
            # rare case event where all bigrams probabilities from the last generated word are zero
            else:
                next_id =  random.randint(0, self.unique_words-1)
                next_token = self.word[next_id]
            
            print(next_token,end=' ')
            wordcount += 1
            id = next_id
        #pass


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
