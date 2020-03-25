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

    def create_dataset(self, source) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(filename) as source:
            for w,tags,tree,relations in self.trees(source) : 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w,tags,i,stack,m)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds.dataset2arrays()
   


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
                configuration
        """
        moves = []

        # YOUR CODE HERE

        # From step_by_step() , we note that i starts from 0.
        # The inital length of the buffer is also the length of the predicted tree
        # Since i starts from 0, as long as i < len(pred_tree), SH is a valid move
        # When i >= len(pred_tree), there are no more unprocessed word in the buffer, hence SH will become an invalid move

        if i < len(pred_tree):
            moves.append(self.SH)

        # The definition of RA is to create an arc from the second topmost word to the topmost word on the stack,
        # then remove the topmost word from the stack
        # Since the stack should contains root in the final configuration, no removal of root is allowed
        # Hence, the length of the stack must be >= 2

        if i>=2 and len(stack) >=2:
            moves.append(self.RA)

        # The definition of LA is to create an arc from the topmost word to the second topmost word on the stack,
        # then remove the second topmost word from the stack
        # Since the stack should contains root in the final configuration, no removal of root is allowed
        # Hence, the length of the stack must be >= 3

        if i>=3 and len(stack) >=3:
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

        # YOUR CODE HERE

        # SH will add the first unprocessed word currently on the buffer to stack
        # After that, It will also increment and update the index of the first unprocessed word on the buffer  
        if move == self.SH:
            stack.append(i)
            i += 1

        # LA will create an arc from the topmost word to the second topmost word on the stack
        # then remove the second topmost word from stack
        # See slide 75 to 76 in lecture 2
        elif move == self.LA:
            topmost_word = stack[-1]
            # remove the 2nd topmost word from stack which also serves as the position in which we update the tree
            second_topmost_word = position_update_index = stack.pop(-2) 
            pred_tree[position_update_index] = topmost_word

        # RA will create an arc from the second topmost word to the topmost word on the stack
        # then remove the topmost word from stack
        # See slide 84 to 85 in lecture 2
        elif move == self.RA:
            second_topmost_word = stack[-2]
            # remove the topmost word from stack which also serves as the position in which we update the tree
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


        # YOUR CODE HERE

        # Intuition from Page 11 for Chapter 15 of the textbook
        # Basic algorithm to choose the correct move involves finding the number of matches between pred_tree and correct_tree

        # If i<2 and len(stack)<2, only one choice: SH
        # Else, check LA -> RA -> SH

        # For choosing LA, just need one condition:
        # 1. If after executing LA, the number of correct matches + 1

        # There are 2 different cases for choosing RA:
        # For the 1st case, 2 conditions must be met: 
        # i.  After the execution of RA, umber of correct matches + 1 AND
        # ii. All of the dependents of topmost word has been assigned, i.e. number of occurrence of topmost word in pred_tree = correct_tree 
        # For the second case, just need to fulfil one condition
        # pred_tree == correct_tree
        # Note: RA is needed here to draw a right arc from ROOT to the only word left in the stack

        # If LA and RA both do not fit the criteria, then choose SH
        

        # If i< 2 and len(stack) < 2, choose SH
        if i <= len(pred_tree): 
            if i<2 and len(stack) <2:
                return self.SH
            else:
                numMatchingPairs = sum(a == b for a,b in zip(pred_tree, correct_tree))
                # Note: We do not want to modify any of the tree here, 
                # so it is best to make a copy of the pred_tree for testing in each scenario
                if len(stack) >= 2: 
                    topmost_word = stack[-1]
                    second_topmost_word = stack[-2] #only available if len(stack)>=2, therefore the precondition
                    
                    # Suppose 'LA' is executed,
                    # Check if new number of matching pairs increase by 1, if so choose 'LA'
                    pred_tree_copy = pred_tree.copy()
                    pred_tree_copy[second_topmost_word] = topmost_word
                    new_numMatchingPairs = sum(a == b for a,b in zip(pred_tree_copy, correct_tree))
                    if  new_numMatchingPairs == numMatchingPairs + 1 :
                        return self.LA
                    
                    # If not, suppose 'RA' is executed,
                    # Check if new number of matching pairs increase by 1 and
                    # all of the dependents of topmost word has been assigned
                    # If so, choose 'RA'
                    pred_tree_copy = pred_tree.copy()
                    pred_tree_copy[topmost_word] = second_topmost_word
                    new_numMatchingPairs = sum(a == b for a,b in zip(pred_tree_copy, correct_tree))
                    if (new_numMatchingPairs == numMatchingPairs + 1) and (pred_tree_copy.count(topmost_word) == correct_tree.count(topmost_word)):
                        return self.RA
                    # If pred_tree = correct_tree, draw a right arc from ROOT to the only word left in the stack
                    # Note that at this point, pred_tree == correct_tree since all index are initialised to 0
                    if pred_tree == correct_tree:
                       return self.RA

                    # else execute SH
                    return self.SH

        return None
   

  
filename = Path("en-ud-train-projective.conllu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename, help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step( args.step_by_step )

    elif args.compute_correct_moves:
        with open(filename) as source:
            for w,tags,tree,relations in p.trees(source) :
                print( p.compute_correct_moves(tree) )





