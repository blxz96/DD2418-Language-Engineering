import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        """
        #
        # YOUR CODE HERE
        #
        moves = [] # move list
        i, stack, pred_tree = 0, [], [0]*len(words) # initialize i,stack,pred_tree
        ds.add_datapoint(words, tags, i, stack, False) # Convert words and tags into features matrix
        choice = model.classify_word(ds.datapoints[0]) # Get the probs from model
        
        # Get the move that is valid and the highest possibility
        for index in range(len(choice)):
            if (choice[i] >= max(choice)) and (self.__parser.valid_movse(i, stack, pred_tree)):
                moves.append(choice[i])
            else:
                choice[i] = 0

        ds.datapoints.pop(0) # Remove the processed datapoints
        i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, choice) # make the move, upload values
        while True:
            ds.add_datapoint(words, tags, i, stack, False) # Convert words and tags into features matrix
            choice = model.classify_word(ds.datapoints[0]) # Get the next move
            for index in range(len(choice)):
                if (choice[i] >= max(choice)) and (self.__parser.valid_movse(i, stack, pred_tree)):
                    moves.append(choice[i])
                else:
                    choice[i] = 0
            ds.datapoints.pop(0) # Remove the processed datapoints
            i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, choice) # make the move, upload values
            if ds.judge_end:    # if reach the end (len(stack) == 1)
                break
        return moves


        
    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        #
        p = self.__parser
        test_ds = p.create_dataset(test_file)
        moves = []
        sentence_cnt = 0
        sentence_total = test_ds.calculateTotal()
        with open(test_file) as source:
            uascnt = 1
            for words,tags,tree,relations in p.trees(source): 
                moves = self.build(model, words, tags, ds) # call the build function 
                                                           # return moves list
                tmpcnt = 0
                flag = 1
                for item in test_ds.moves:
                    if moves[tmpcnt] == item: # if predicted correctly
                        if (item != 0): # if it's an arc (la or ra)
                            uascnt += 1 # add count
                    else: # There exist error prediction, so sentence level is wrong
                        flag = 0
                if flag == 1:
                    sentence_cnt += 1 # if all predicted correctly, sentence level + 1
        print("Sentence-level acc: " + str(sentence_cnt / sentence_total) + "%")
        print("UAS: " + str(uascnt / test_ds.get_all_cnt()) + "%")

        

if __name__ == '__main__':
    #
    # TODO:
    # 1) Replace the `create_dataset` function from dep_parser_fix.py to your dep_parser.py file
    # 2) Replace parse_dataset.py with the given new version
    #

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)
    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())
    
    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev.conllu', ds)            