#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try :
            self.tokens = nltk.word_tokenize(text) # Important that it is named self.tokens for the --check flag to work
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE
        
        # Algorithm for processing unigram: 
        # Check if the token is in self.index {word: identifier}
        # If so, increment self.unigram_count, self.total_words
        # Else, add entry into self.index and self.word as well as increment self.unigram_count, self.unique_words and self.total_words
        
        # Algorithm for processing bigram:
        # Pre-condition: last_index is not the default value of -1
        # Note: Bigram must be 2 token
        # Note: We will record bigram based fully on their identifier to generate out the text
        # Check if (last_index, current_index) is one of the keys of self.bigram_count. Note: last_index = last identifier, current_index = self.index[token]
        # If so, increment self.bigram_count for (last_index, current_index)
        # Else, create (last_index, current_index) as a key and increment self.bigram_count for (last_index, current_index)

        # Finally, update last_index to self.index[token], i.e the identifier matching the word/token

        if token in self.index:
            self.unigram_count[token] += 1
            self.total_words += 1
        else:
            self.index[token] = self.unique_words
            self.word[self.unique_words] = token
            self.unigram_count[token] += 1
            self.unique_words += 1
            self.total_words += 1
        
        if self.last_index != -1:
            if (self.last_index ,self.index[token]) in self.bigram_count:
                self.bigram_count[(self.last_index,self.index[token])] += 1
            else:
                self.bigram_count[(self.last_index,self.index[token])] = 1

        self.last_index = self.index[token]        


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # YOUR CODE HERE

        # The first line
        first_line = str(self.unique_words) + " " + str(self.total_words)
        rows_to_print.append(first_line)

        # The V lines
        for i in range(self.unique_words):
            line_V = str(i) + " " + str(self.word[i]) + " " + str(self.unigram_count[self.word[i]])
            rows_to_print.append(line_V)

        # Rest of the lines for non-zero bigram probability
        # Note: from small_model_correct.txt, it should be sorted
        # ln(probability of bigram i) = ln(freq of bigram i/number of first token in text)
        for k,v in sorted(self.bigram_count.items()): 
            no_token1 = self.unigram_count[self.word[k[0]]] # number of the first token
            if no_token1 != 0:
                bigram_probability_ln = format(math.log(v/no_token1),'.15f') # v = freq of bigram
            # can actually comment out else since it won't be reachable anyway as no_token1 will always be != 0    
            else:
                bigram_probability_ln = format(0,'.15f')
            rows_to_print.append(str(k[0]) + " " + str(k[1]) + " " + str(bigram_probability_ln))

        # The final line
        rows_to_print.append(str(-1))

        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')
    parser.add_argument('--check', action='store_true', help='check if your alignment is correct')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    if arguments.check:
        results  = bigram_trainer.stats()
        payload = json.dumps({
            'tokens': bigram_trainer.tokens,
            'result': results
        })
        response = requests.post(
            'https://language-engineering.herokuapp.com/lab2_trainer',
            data=payload,
            headers={'content-type': 'application/json'}
        )
        response_data = response.json()
        if response_data['correct']:
            print('Success! Your results are correct')
            for row in results: print(row)
        else:
            print('Your results:\n')
            for row in results: print(row)
            print("The server's results:\n")
            for row in response_data['result']: print(row)
    else:
        stats = bigram_trainer.stats()
        if arguments.destination:
            with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
                for row in stats: f.write(row + '\n')
        else:
            for row in stats: print(row)


if __name__ == "__main__":
    main()
