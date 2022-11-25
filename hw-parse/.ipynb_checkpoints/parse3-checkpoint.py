#!/usr/bin/env python3
"""
Determine whether sentences are grammatical under a CFG, using Earley's algorithm.
(Starting from this basic recognizer, you should write a probabilistic parser
that reconstructs the highest-probability parse of each given sentence.)
"""

# Recognizer code by Arya McCarthy, Alexandra DeLucia, Jason Eisner, 2020-10, 2021-10.
# This code is hereby released to the public domain.

from __future__ import annotations
import argparse
import logging
import math
import tqdm
from dataclasses import dataclass
from pathlib import Path
from collections import Counter,defaultdict
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG'"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol", 
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--progress", 
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    args = parser.parse_args()
    return args


class EarleyChart:
    """A chart for Earley's algorithm."""
    
    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.  
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self.left_ancestor = defaultdict(set)
        self.tree =[]
        self._run_earley()    # run Earley's algorithm to construct self.cols
    
        
    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    return True
        return False   # we didn't find any appropriate item
    def best_weight(self) ->List[float]:
        """ find all the possible parse if exist, otherwise return empty
        """
        weight =0 
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    weight = self.cols[-1].get_weight(item)
                    
        return weight
    def print_tree(self) ->List[float]:
        """ find all the possible parse if exist, otherwise return empty
        """
        
        for item in self.cols[-1].all():    # the last column
            if (item.rule.lhs == self.grammar.start_symbol   # a ROOT item in this column
                and item.next_symbol() is None               # that is complete 
                and item.start_position == 0):               # and started back at position 0
                    weight = self.cols[-1].get_weight(item)
                    #minimum_parse.append(weight)
                    idx = self.cols[-1]._index[item]
                    coordinate = (-1,idx) #col=-1 and index=idx
                    self.recursive_print_tree(coordinate) #an empty tree
        if not self.tree:
            return None
        else:
            self.tree.append(")")
            return " ".join(self.tree)
    def recursive_print_tree(self,coordinate):
        #base case coordinate is tuple of token
        if len(coordinate) == 1:
            #self.tree.append(coordinate[0])
            return
        #otherwise, we assume that coordinate is (col, idx)
        #we first find the item that coordinate pointers to 
        col = coordinate[0]
        idx = coordinate[1]
        log.debug(f"\tcolumn {col} index {idx}")
        agenda = self.cols[col]
        item = agenda._items[idx]
        rhs = item.rule.rhs
        backpointer = agenda._backpointer[idx]
        #we assert that the length of rhs equals to the length of backpointer
        assert (len(rhs) == len(backpointer))
        
        #add root
        if len(self.tree) == 0:
            self.tree.append("("+item.rule.lhs)
        
        for i in range(len(rhs)):
            if self.grammar.is_nonterminal(rhs[i]):
                self.tree.append("(")
            self.tree.append(rhs[i])
            self.recursive_print_tree(backpointer[i])
            if self.grammar.is_nonterminal(rhs[i]):
                self.tree.append(")")
    def recursive_build_left_ancestor(self,y):
        #self.grammar._expansions
        #self.grammar._leftparent
        #outpout self.left_ancestor 
        for parent in self.grammar._leftparent[y]:
            if y not in self.left_ancestor[parent]:
                self.left_ancestor[parent].add(y)
                #use parent to search grandparent 
                self.recursive_build_left_ancestor(parent)
        
    def _run_earley(self) -> None:
        """Fill in the Earley chart"""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]
        
        
        
        # Start looking for ROOT at position 0
        #self._predict(self.grammar.start_symbol, 0)

        # We'll go column by column, and within each column row by row.
        # Processing earlier entries in the column may extend the column
        # with later entries, which will be processed as well.
        # 
        # The iterator over numbered columns is `enumerate(self.cols)`.  
        # Wrapping this iterator in the `tqdm` call provides a progress bar.
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            predicted = set() #keep track of nonterminals that have already been predict for the current column
            self.left_ancestor = defaultdict(set)
            log.debug(f"self.grammar._leftparent: {self.grammar._leftparent}")
            if i <len(self.tokens):
                self.recursive_build_left_ancestor(self.tokens[i])
            log.debug(f"self.left_ancestor: {self.left_ancestor}")
            
            
            # Start looking for ROOT at position 0
            if i==0:
                self._predict(self.grammar.start_symbol, 0)
            
            
            while column:    # while agenda isn't empty
                item,weight = column.pop()   # dequeue the next unprocessed item
                next = item.next_symbol();
                if next is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i,weight)   
                elif self.grammar.is_nonterminal(next):
                    # Predict the nonterminal after the dot
                    # if we've already predicted the nonterminal, then skip 
                    if next not in predicted:
                        log.debug(f"{item} => PREDICT")
                        self._predict(next, i)
                        predicted.add(next)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i,weight)                      

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            #add exactly the rules in R(A,B) for each B in S_j(A)
            log.debug(f"\tcheck: {rule.rhs[0]} in {self.left_ancestor[nonterminal]}")
            if rule.rhs[0] in self.left_ancestor[nonterminal]:
                new_item = Item(rule, dot_position=0, start_position=position)
                new_weight = rule.weight
                backpointer = [None]*len(rule.rhs)#backpointers is None for new item
                self.cols[position].push(new_item,new_weight,backpointer) 
                log.debug(f"\tPredicted: {new_item} in column {position}")
                self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int,weight:float) -> None:
        """Attach the next word to this item that ends at position, 
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_item = item.with_dot_advanced()
            #the customer backpointer first None will be replaced by the item's coordinate
            _agenda = self.cols[position]
            idx = _agenda._index[item]
            old_backpointer = _agenda._backpointer[idx] #list
            coordinate = (self.tokens[position],) #the scaned token: string 
            new_backpointer =  _agenda.pointer_advanced(old_backpointer,coordinate) #tuple

            self.cols[position + 1].push(new_item,weight,new_backpointer)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            log.debug(f"\tbackpointer is {self.cols[position+1]._backpointer[-1]}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int,weight:float) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column.  (This operation is sometimes
        called "complete," but actually it attaches an item that was already complete.)
        """
        mid = item.start_position   # start position of this item = end position of item to its left
        _agenda = self.cols[mid]
        _next_symbol = _agenda._next_symbol
        log.debug(f"\t print the _next_symbol: {_next_symbol} in column {mid}")
        log.debug(f"\t print the lhs: {item.rule.lhs} in column {position}")
        if item.rule.lhs in _next_symbol:
            for idx in _next_symbol[item.rule.lhs]:
                customer = _agenda._items[idx] #customer is the item
        
        #for customer in self.cols[mid].all():  # could you eliminate this inefficient linear search?
            #if customer.next_symbol() == item.rule.lhs:
                new_item = customer.with_dot_advanced()
                #get the customer's weight, self.cols[mid] is a agenda where customer exist
                customer_weight = _agenda.get_weight(customer)
                
                #the customer backpointer first None will be replaced by the item's coordinate
                customer_backpointer = _agenda._backpointer[idx] #list
                coordinate = (position,self.cols[position]._index[item]) #coordinate of item col and index
                log.debug(f"\tcoordinate is {coordinate}")
                new_backpointer =  _agenda.pointer_advanced(customer_backpointer,coordinate) #list
                
                self.cols[position].push(new_item,weight+customer_weight,new_backpointer)
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                log.debug(f"\tbackpointer is {self.cols[position]._backpointer[-1]}")
                self.profile["ATTACH"] += 1


class Agenda:
    """An agenda of items that need to be processed.  Newly built items 
    may be enqueued for processing by `push()`, and should eventually be 
    dequeued by `pop()`.

    This implementation of an agenda also remembers which items have
    been pushed before, even if they have subsequently been popped.
    This is because already popped items must still be found by
    duplicate detection and as customers for attach.  

    (In general, AI algorithms often maintain a "closed list" (or
    "chart") of items that have already been popped, in addition to
    the "open list" (or "agenda") of items that are still waiting to pop.)

    In Earley's algorithm, each end position has its own agenda -- a column
    in the parse chart.  (This contrasts with agenda-based parsing, which uses
    a single agenda for all items.)

    Standardly, each column's agenda is implemented as a FIFO queue
    with duplicate detection, and that is what is implemented here.
    However, other implementations are possible -- and could be useful
    when dealing with weights, backpointers, and optimizations.

    >>> a = Agenda()
    >>> a.push(3)
    >>> a.push(5)
    >>> a.push(3)   # duplicate ignored
    >>> a
    Agenda([]; [3, 5])
    >>> a.pop()
    3
    >>> a
    Agenda([3]; [5])
    >>> a.push(3)   # duplicate ignored
    >>> a.push(7)
    >>> a
    Agenda([3]; [5, 7])
    >>> while a:    # that is, while len(a) != 0
    ...    print(a.pop())
    5
    7

    """

    def __init__(self) -> None:
        self._items: List[Item] = []       # list of all items that were *ever* pushed
        self._weights: List[Float] =[]
        self._next = 0                     # index of first item that has not yet been popped
        self._reprocess = []               # buffer: store index of item that need to be reprocessing
        self._index: Dict[Item, int] = {}  # stores index of an item if it has been pushed before
        self._next_symbol =defaultdict(list)#:Dict[Str,List[int] #store next symbol of item and its list of index
        self._backpointer:List[List[Tuple]] =[]  # list of backpointer for each items, backpointer is  list  of 
                                          # coordinate, coordinate is tuple (col,index)
        # Note: There are other possible designs.  For example, self._index doesn't really
        # have to store the index; it could be changed from a dictionary to a set.  
        # 
        # However, we provided this design because there are multiple reasonable ways to extend
        # this design to store weights and backpointers.  That additional information could be
        # stored either in self._items or in self._index.

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped.
        Enables `len(my_agenda)`."""
        return len(self._items) - self._next

    def push(self, item: Item, weight:Float,new_backpointer:List) -> None:
        """Add (enqueue) the item, unless it was previously added."""
        """if the item already exist in the agenda, compare their weight, and update the weight as the less one"""
        if item not in self._index:    # O(1) lookup in hash table
            self._items.append(item)
            self._weights.append(weight)
            self._index[item] = len(self._items) - 1
            #next_symbol (only for nonterminal):index
            if item.next_symbol(): # and not self.grammar.is_nonterminal(item.next_symbol):
                _next =item.next_symbol()
                self._next_symbol[_next].append(len(self._items) - 1)
            #append new_backpointer
            self._backpointer.append(new_backpointer)
            
        if item in self._index:
            idx = self._index[item]
            #if the new item has less weight, then replace the old item's weight and back pointer
            if weight <self._weights[idx]:
                self._weights[idx] = weight
                self._backpointer[idx] = new_backpointer
                #reprocessing, append the updated item into reprocess buffer, so we can reprocess it again
                self._reprocess.append(idx)
                
            
            
    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued).
        Raises IndexError if there are no items waiting."""
        """if the reprocess buffer in not empty, we first pop item in repr"""
        if len(self)==0:
            raise IndexError
        if len(self._reprocess) !=0:
            idx = self._reprocess.pop()
            item = self._items[idx]
            weight = self._weights[idx]
        else:
            item = self._items[self._next]
            weight = self._weights[self._next]
            self._next += 1
        return item,weight

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed, even if 
        they've already been popped."""
        return self._items
    def get_weight(self,item):
        """ used in attach, find the weight of customer:Item """
        idx = self._index[item]
        return self._weights[idx]
    def pointer_advanced(self,old_backpointer:List, coordinate:Tuple)-> List:
        """build a new pointer copy old pointer, but replace the first None as coordinate"""
        new_backpointer = []
        count =0
        for e in old_backpointer:
            if e==None and count==0:
                new_backpointer.append(coordinate)
                count+=1
            else:
                new_backpointer.append(e)
        return new_backpointer
    
    def __repr__(self):
        """Provide a REPResentation of the instance for printing."""
        next = self._next
        return f"{self.__class__.__name__}({self._items[:next]}; {self._items[next:]})"

class Grammar:
    """Represents a weighted context-free grammar."""
    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol, 
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = {}    # maps each LHS to the list of rules that expand it
        self._leftparent = defaultdict(set)   # maps from left child to its parent
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited linfore of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())  
                rule = Rule(lhs=lhs, rhs=rhs, weight=-math.log2(prob))
                if lhs not in self._expansions:
                    self._expansions[lhs] = []
                    
                self._expansions[lhs].append(rule)
                self._leftparent[rhs[0]].add(lhs)
                
    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


# A dataclass is a class that provides some useful defaults for you. If you define
# the data that the class should hold, it will automatically make things like an
# initializer and an equality function.  This is just a shortcut.  
# More info here: https://docs.python.org/3/library/dataclasses.html
# Using a dataclass here lets us specify that instances are "frozen" (immutable),
# and therefore can be hashed and used as keys in a dictionary.
@dataclass(frozen=True)
class Rule:
    """
    Convenient abstraction for a grammar rule. 
    A rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"


# We particularly want items to be immutable, since they will be hashed and 
# used as keys in a dictionary (for duplicate detection).  
@dataclass(frozen=True)
class Item:
    """An item in the Earley parse table, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    
    # We don't store the end_position, which corresponds to the column
    # that the item is in, although you could store it redundantly for 
    # debugging purposes if you wanted.

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def with_dot_advanced(self) -> Item:
        if self.next_symbol() is None:
            raise IndexError("Can't advance the dot past the end of the rule")
        return Item(rule=self.rule, dot_position=self.dot_position + 1, 
                    start_position=self.start_position)

    def __repr__(self) -> str:
        """Complete string used to show this item at the command line"""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule})"  # matches notation on slides


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.verbose)  # Set logging level appropriately

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f.readlines():
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                chart = EarleyChart(sentence.split(), grammar, progress=args.progress)
                # print the result
                #print(
                #    f"'{sentence}' is {'accepted' if chart.accepted() else 'rejected'} by {args.grammar}"
                #)
                #print(
                #    f"'{sentence}' has parse_tree {chart.print_tree()} by {args.grammar}"
                #)
                print(chart.print_tree())
                print(chart.best_weight())
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest
    #doctest.testmod(verbose=False)   # run tests
    main()
