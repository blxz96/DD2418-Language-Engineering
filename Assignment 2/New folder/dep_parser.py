from pathlib import Path
from parse_dataset import Dataset
import argparse

class Parser: 
    SH, LA, RA = 0,1,2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()    # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():    # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations


    def step_by_step(self,string) :
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0]*len(w) # Input configuration
        while True :
            print( "----------------" )
            print( "Buffer: ", w[i:] )
            print( "Stack: ", [w[s] for s in stack] )
            print( "Predicted tree: ", pred_tree )
            try :
                m = int(input( "Move (SH=0, LA=1, RA=2): " ))
                if m not in self.valid_moves(i,stack,pred_tree) :
                    print( "Illegal move" )
                    continue
            except :
                print( "Illegal move" )
                continue
            i, stack, pred_tree = self.move(i,stack,pred_tree,m)
            if i == len(w) and stack == [0] :
                # Terminal configuration
                print( "----------------" )
                print( "Final predicted tree: ", pred_tree )
                return

    #
    # REPLACE THE ORIGINAL create_dataset FUNCTION WITH THIS ONE
    #
    def create_dataset(self, source, train=False) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(source) as f:
            for w,tags,tree,relations in self.trees(f): 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds

   


    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []

        # As long as there are unprocessed word, SH is a valid move
        if i < len(pred_tree):
            moves.append(self.SH)
        
        # There need to be a second topmost(which can be ROOT) and
        # a topmost, so len(stack) >= 2
        if i >= 2 and len(stack) >= 2:
            moves.append(self.RA)
        
        # There need to be a second topmost(which cannot be ROOT, otherwise 
        # would remove ROOT) and a topmost, so len(stack) >= 3     
        if i >= 3 and len(stack) >= 3:
            moves.append(self.LA)        
        
        return moves

        
    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # SH will add the first unprocessed word currently on the buffer to stack
        if move == self.SH:
            stack.append(i)
            i += 1
        # LA will create an arc from the topmost word to the second topmost word on the stack
        # then remove the second topmost word from stack
        elif move == self.LA:
            topmost_word = stack[-1]
            second_topmost_word = position_update_index = stack.pop(-2) 
            pred_tree[position_update_index] = topmost_word
        # RA will create an arc from the second topmost word to the topmost word on the stack
        # then remove the topmost word from stack
        elif move == self.RA:
            second_topmost_word = stack[-2]
            topmost_word = position_update_index = stack.pop(-1) 
            pred_tree[position_update_index] = second_topmost_word

        return i, stack, pred_tree


    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser 
        must perform in order to produce the input tree.
        """
        i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
        moves = []
        m = self.compute_correct_move(i,stack,pred_tree,tree)
        while m != None :
            moves.append(m)
            i,stack,pred_tree = self.move(i,stack,pred_tree,m)
            m = self.compute_correct_move(i,stack,pred_tree,tree)
        return moves


    def compute_correct_move(self, i,stack,pred_tree,correct_tree) :
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook, chapter 15, page 11. 
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        assert len(pred_tree) == len(correct_tree)

        if i <= len(pred_tree):
            # SH is the only valid choice here
            if i < 2 and len(stack) <2:
                return self.SH
            else:
                numMatchingPairs = sum(a == b for a,b in zip(pred_tree, correct_tree))

                if len(stack) >= 2: 
                    topmost_word = stack[-1]
                    second_topmost_word = stack[-2]
                    
                    # Check if new number of matching pairs increase by 1, if so choose 'LA'
                    pred_tree_copy = pred_tree[:]
                    pred_tree_copy[second_topmost_word] = topmost_word
                    new_numMatchingPairs = sum(a == b for a,b in zip(pred_tree_copy, correct_tree))
                    if  new_numMatchingPairs == numMatchingPairs + 1 :
                        return self.LA
                    
                    # Check if new number of matching pairs increase by 1 and
                    # all of the dependents of topmost word has been assigned
                    # If so, choose 'RA'
                    pred_tree_copy = pred_tree[:]
                    pred_tree_copy[topmost_word] = second_topmost_word
                    new_numMatchingPairs = sum(a == b for a,b in zip(pred_tree_copy, correct_tree))
                    if (new_numMatchingPairs == numMatchingPairs + 1) and (pred_tree_copy.count(topmost_word) == correct_tree.count(topmost_word)):
                        return self.RA

                    # If pred_tree = correct_tree, draw a right arc from ROOT to the only word left in the stack
                    if pred_tree == correct_tree:
                       return self.RA

                    # else SH
                    return self.SH

        return None
   

  
filename = "en-ud-train-projective.conllu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename, help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step( args.step_by_step )

    elif args.compute_correct_moves:
        outputFile = open("output.txt", "w")
        with open(filename) as source:
            for w,tags,tree,relations in p.trees(source) :
                pmoves = p.compute_correct_moves(tree)
                print(pmoves)
                outputFile.write("[")
                cnt = 0
                for item in pmoves:
                    if cnt == 0:
                        outputFile.write(str(item))
                        cnt += 1
                    else:
                        outputFile.write("," + str(item))
                outputFile.write("]\n")





