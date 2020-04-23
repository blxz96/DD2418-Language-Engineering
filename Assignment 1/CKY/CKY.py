from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class CKY :

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []


    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file) :
        stream = open( grammar_file, mode='r', encoding='utf8' )
        for line in stream :
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2 :
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules :
                    first_rules = self.binary_rules[first]
                else :
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules :
                    second_rules = first_rules[second]
                    if left not in second_rules :
                        second_rules.append[left]
                else :
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1 :
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules :
                    word_rules = self.unary_rules[word]
                    if left not in word_rules :
                        word_rules.append( left )
                else :
                    word_rules = [left]
                    self.unary_rules[word] = word_rules


    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s) :
        self.words = s.split()        
        
        #  YOUR CODE HERE
        
        n = len(self.words)

        for i in range(n): # create a list with nested lists
            self.table.append([])
            for j in range(n):
                self.table[i].append("")

        for j in range(n):
            for i in range(n-1, -1, -1):
                if j < i:
                    continue
                elif j == i:
                    curword = self.words[i]
                    self.table[i][j] = self.unary_rules[curword]
                    for every in self.unary_rules[curword]:
                        tmptup = (every, i, j, curword, -1, -1, curword, -1, -1)
                        self.backptr.append(tmptup)
                    
                else:
                    self.binary_find(n, i, j)

    # Find in binary_rules
    def binary_find(self, n, r, c):
        # print("----------")
        # print(r, c)
        tmplist = []
        for j in range(c):
            leftBox = self.table[r][j]
            i = j + 1
            belowBox = self.table[i][c]

            if (leftBox == "") or (belowBox == ""):
                continue
            for leftBoxItems in leftBox:
                for belowBoxItems in belowBox:
                    tmpdic = {}
                    if leftBoxItems in self.binary_rules:
                        tmpdic = self.binary_rules[leftBoxItems]
                        if belowBoxItems in tmpdic:
                            tmplist.append(tmpdic[belowBoxItems][0])
                            tmptup = (tmpdic[belowBoxItems][0], r, c, leftBoxItems, r, j, belowBoxItems, i, c)
                            self.backptr.append(tmptup)
        self.table[r][c] = tmplist

    # Prints the parse table
    def print_table( self ) :
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print( t.table )
        t = AsciiTable(self.backptr)
        t.inner_heading_row_border = False
        print( t.table )


    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    def print_trees( self, row, column, symbol ) :
        for everytup in self.backptr:

            if (everytup[1] == row) and (everytup[2] == column) and (everytup[0] == symbol):
                if (everytup[4] == -1):
                    print(symbol + " ( " + everytup[3] + " ),"),
                    return
                else:
                    print(symbol + "("),
                    self.print_trees(everytup[4], everytup[5], everytup[3])
                    self.print_trees(everytup[7], everytup[8], everytup[6])



def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str,  required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY( arguments.grammar )
    cky.parse( arguments.input_sentence )
    if arguments.print_parsetable :
        cky.print_table()
    if arguments.print_trees :
        cky.print_trees( 0, len(cky.words)-1, arguments.symbol )
    

if __name__ == '__main__' :
    main()    


                        
                        
                        
                    
                
                    

                
        
    
