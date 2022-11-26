#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = None
        self._load_rules_from_file(grammar_file)
    
    

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """
        """
        
        1.first we need to read each line of the grammar.gr, skip empty lines and lines start with #
        2,split each line with 	, get count, key, value, then split the value by space, get the value list, such as NP VP
        3.if the key is not in rules, rules[key]=[[value_list],[count]]
        else:
            rules[key][0].append(value_list)
            rules[key][1].append(count)


        we need a dictionary/hashmap: rules={key:[options,weight for each option]}

        """
        self.rules = {}
        with open(grammar_file) as f:
            lines = f.read().splitlines()
            for line in lines:
                if not line.strip() or line.startswith("#"):
                    continue
                else:
                    #print(line)
                    line.splitlines() 
                    count,key,values = line.split("\t")
                    value_list = values.split(" ") 
                    #for case 1	ROOT	is it true that S ?     # mixing terminals and nonterminals is ok
                    for i in range(len(value_list)):
                        if value_list[i]==''or value_list[i]=='#':
                            value_list = value_list[:i]
                            break
                            
                    
                    if key not in self.rules:
                        self.rules[key]=[[value_list],[float(count)]]
                    else:
                        self.rules[key][0].append(value_list)
                        self.rules[key][1].append(float(count))
        #comb Nonterminal items in self.rule
        self.combNonterminals()
        #return self.rules
        #raise NotImplementedError
        
    #helper function,combine nonterminal for each list
    #['is', 'it', 'true', 'that', 'S', '?'] --> ['is it true that', 'S', '?'], count stay unchanged
    def comb(self,value_list):
        
        comb_value_list = [value_list[0]]
        flag = True if value_list[0] not in self.rules else False
        
        for i in range(1,len(value_list)):
            if value_list[i] not in self.rules:
                #
                if not flag:
                    flag=True
                    comb_value_list.append(value_list[i])
                else:
                    #comb
                    comb_value_list[-1] = comb_value_list[-1]+' '+ value_list[i]
            else:
                flag=False
                comb_value_list.append(value_list[i])
            
        return comb_value_list
    
    #comb nonterminals for the all hashtable
    def combNonterminals(self):
        """
        iterate through each item of self.rules, 
        
        """
        for key in self.rules:
            values=self.rules[key][0] #count remain unchanged
            #self.rules[key][0][i] = []
            for i in range(len(values)):
                value_list = values[i]
                comb_value_list = self.comb(value_list)
                self.rules[key][0][i] = comb_value_list
                
    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        self.output=[]
        self.expansion = 1 # global variables
        self.derivation_tree = derivation_tree
        self.max_expansions = max_expansions
       
        self.dfs(start_symbol)
        #return self.output
        return " ".join(self.output)
    
    def dfs(self,symbol):
        #print(self.expansion,symbol)
        

        #base case
        if symbol not in self.rules:
            self.output.append(symbol)
            return 
        
        #if exceed the maximal number of nonterminals
        if self.expansion > self.max_expansions:
            self.output.append("...")
            return
        
        #otherwise
        if self.derivation_tree:
            self.output.append("(" +symbol)
            
        children = random.choices(self.rules[symbol][0], weights=self.rules[symbol][1], k=1)[0]
        self.expansion+= 1
        for child in children:
            #self.expansion+= 1
            self.dfs(child)
            
        if self.derivation_tree:
            self.output.append(")")
            

####################
# ## Main Program
# ###################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
