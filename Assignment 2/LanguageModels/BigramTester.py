#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
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

        # The identifier of the previous word processed in the test corpus. 
        # Is -1 if the last word was unknown. 2 cases:
        # 1. no previous word processed
        # 2. token in the test corpus could not be found in the training model
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                # YOUR CODE HERE
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


    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE

        # Note1: Entropy of test set: -1/N * summation of log P(wi-1 wi) from i= 1 to i = N
        # Note2: P(wi-1 wi) = λ1 * P(wi| wi-1) + λ2 * P(wi) + λ3
        
        # Alogorithm: 
        
        # When processing a certain token in the test corpus, we can only keep track of the last token we have processed before it via self.last_index
        # For the first token, since we only have one term, we cannot compute the entropy in note 2
        # Hence, we should only start computing the entropy during the 2nd token, when self.test_words_processed = 1 
        
        # Note that during the computation of the entropy during the 2nd token (or any other token apart from the 1st token), there could be 4 cases happening:
        
        # Case 1 : 1st token is unknown and 2nd token is unknown 
        # P(wi-1 wi) = λ3
        
        # Case 2 : 1st token is known but 2nd token is unknown
        # P(wi-1 wi) = λ3

        # Case 3 : 1st token is unknown but 2nd token is known
        # P(wi-1 wi) =  λ2 * P(wi) + λ3  

        # Case 4 : Both tokens are known
        # Might not necessarily have a bigram in the training model even if both tokens are known 
        # If bigram in training model: P(wi-1 wi) = λ1 * P(wi| wi-1) + λ2 * P(wi) + λ3
        # Else P(wi-1 wi) =  λ2 * P(wi) + λ3  
        
        # To get the index of tokens:
        # Previous token can be accessed via self.last_index
        # Current token : self.index[word]
    
        # Bigram Probabilities, P(wi| wi-1) could be access in self.bigram_prob and take the exponents of it
        # Unigram Probabilities, P(wi) = self.unigram_count[word]/self.total_words
        
        # After calculation of P(wi-1 wi) is done, convert to log P(wi-1 wi) with math.log
        # If self.test_words_processed = 0 , self.logProb = 0
        # If self.test_words_processed = 1 , self.logProb = -1/self.test_words_processed * log P(wi-1 wi)
        # If self.test_words_processed > 1 , self.logProb = (self.logProb * -(self.test_words_processed - 1) + log P(wi-1 wi))/ -self.test_words_processed

        # Finally, increment self.test_words_processed and update self.last_index

        
        # Processing of the first token
        if self.test_words_processed == 0 :
            id = self.index[word] if word in self.index else -1
            self.logProb = 0

        elif self.test_words_processed >= 1 :
            id = self.index[word] if word in self.index else -1
            prev_id = self.last_index

            # Case 1 and 2: As long as 2nd token is unknown, P(wi-1 wi) = λ3 irregardless of whether 1st token is known 
            if id == -1 :
                prob = self.lambda3
            # Case 3: P(wi-1 wi) =  λ2 * P(wi) + λ3  
            elif prev_id == -1 and id != -1 :  
                prob = self.lambda2 * int(self.unigram_count[word])/self.total_words + self.lambda3
            
            # Case 4: Both tokens are known
            # If bigram in training model: P(wi-1 wi) = λ1 * P(wi| wi-1) + λ2 * P(wi) + λ3
            # Else P(wi-1 wi) =  λ2 * P(wi) + λ3 
            elif prev_id != -1 and id != -1:
                if (prev_id , id) in self.bigram_prob:
                    prob = self.lambda1 * math.exp(self.bigram_prob[(prev_id,id)]) + self.lambda2 * int(self.unigram_count[word])/self.total_words + self.lambda3
                else:
                    prob = self.lambda2 * int(self.unigram_count[word])/self.total_words + self.lambda3

            if self.test_words_processed == 1:
                self.logProb = -1/self.test_words_processed * math.log(prob)
            
            elif self.test_words_processed > 1:
                self.logProb = (self.logProb * -(self.test_words_processed - 1) + math.log(prob))/ -self.test_words_processed

        self.test_words_processed += 1
        self.last_index = id 


    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) # Important that it is named self.tokens for the --check flag to work
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    if arguments.check:
        results  = bigram_tester.logProb

        payload = json.dumps({
            'model': open(arguments.file, 'r').read(),
            'tokens': bigram_tester.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_tester',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))
            print('Success! Your results are correct')
        else:
            print('Your results:')
            print('Estimated entropy: {0:.2f}'.format(bigram_tester.logProb))
            print("The server's results:\n Entropy: {0:.2f}".format(response_data['result']))

    else:
        print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
