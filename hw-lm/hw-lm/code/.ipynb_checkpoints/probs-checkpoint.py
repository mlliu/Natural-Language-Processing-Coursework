#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import sys
import random
import tqdm
from SGD_convergent import ConvergentSGD
import numpy as np

from pathlib import Path

import torch
from torch import nn
from torch import optim
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Counter
from collections import Counter

patch_typeguard()   # makes @typechecked work with torchtyping

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Set[Wordtype]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


# #### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process_the_token(token)
    # Whenever the `for` loop needs another token, read_tokens picks up where it
    # left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

# #### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return vocab

#for each word in vocab, build a embedding vector for it
#return a tensor for embedding vactor and a hashtable word2id
def read_lexicon(lexicon_file: Path,vocab:Set):
    #first read each line of lexicon, if the word in vocab,then append word and vector to corresponding list
    #build word2 id
    #iterate through vocab, for those word that not exist in words2id, append it to word2id, with id = oovid

    #iterate through lexicons, build a hashmap mapping from words(in vocab) to vectors
    word2vector = {}
    with open(lexicon_file,'rt') as f:
        first_line = next(f)  # Peel off the special first line.
        for line in f:  # All of the other lines are regular.
            line = line.split('\t') 
            word = line[0]
            if word in vocab or word == "OOL":
                word2vector[word] = [float(line[i]) for i in range(1,len(line))]


    
    #iterate through vocab, for those word that not exist in embeddings, append it to embeddings, 
    #with vector == ool_vector, this can make sure that each word in the vocab has a corresponing vector
    OOL_vector = word2vector["OOL"]
    for v in vocab:
        if v not in word2vector:
            word2vector[v] = OOL_vector
    #add BOS to word2id
    word2vector[BOS] = OOL_vector
    
    #tensor
    
    #embeddings = nn.Tensor(vectors)
    return word2vector


#this is the earlystopper. the evaluation metric is the cross entropy on 
class EarlyStopper:
    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_cross_entropy = np.inf

    def early_stop(self, cross_entropy):
        if cross_entropy < self.min_cross_entropy:
            self.min_cross_entropy = cross_entropy
            self.counter = 0
        elif cross_entropy > (self.min_cross_entropy + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# #### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle  # for loading/saving Python objects
        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            log.info(f"Loaded model from {source}")
            return pickle.load(f)

    def save(self, destination: Path) -> None:
        import pickle
        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


# #### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )
        
    def sample(self,max_length):
        output = [] #store the generated sentence
        x, y = BOS, BOS
        for i in range(max_length):
            candidates = list(self.vocab)
            weights = [ self.prob(x,y,each) for each in candidates]
            #random.choices(self.rules[symbol][0], weights=self.rules[symbol][1], k=1)[0]
            z = random.choices(candidates, weights = weights,k=1)[0]
            
            output.append(z)
            #reach the end?
            if z == EOS:
                break
                
            #otherwise, update x and y
            x = y
            y = z
            
        
        #cheack if the sentence has a EOS or not
        if output[-1]!=EOS:
            output.append("...")
        return ' '.join(output)

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)

        if lambda_ < 0:
            raise ValueError("negative lambda argument of {lambda_}")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)
        
        
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # TODO: Reimplement me so that I do backoff
        #self.total_count = 
        #print(self.event_count[(z,)])
        #print(self.total_count)
        prob_z = (self.event_count[(z,)] +self.lambda_)/(self.event_count[()]+self.lambda_ * self.vocab_size) #p^_z == p_z
        prob_yz = ((self.event_count[(y, z)] + self.lambda_ * self.vocab_size*prob_z)/
                (self.context_count[(y,)] + self.lambda_ * self.vocab_size))
        
        prob_xyz = ((self.event_count[(x, y, z)] + self.lambda_ * self.vocab_size*prob_yz)/
                (self.context_count[(x, y)] + self.lambda_ * self.vocab_size))
        return prob_xyz
        #return super().prob(x, y, z)
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float) -> None:
        super().__init__(vocab)
        if l2 < 0:
            log.error(f"l2 regularization strength value was {l2}")
            raise ValueError("You must include a non-negative regularization value")
        self.l2: float = l2
        
        # TODO: READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        #vocab need to include EOS and OOV
        self.vocab  = vocab
        self.vocab.add(EOS)
        self.vocab.add(OOV)
        #read lexicons file, and build a hashtbale mapping from word to vectors
        word2vector=read_lexicon(lexicon_file,self.vocab)
        self.word2vector = word2vector #hashmap
        
        #self.embeddings = torch.stack([self.word2vector[word] for word in self.vocab])
        self.embeddings = torch.Tensor([self.word2vector[word] for word in self.vocab])
        self.dim = self.embeddings.shape[1]  # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS
        self.embeddings = torch.transpose(self.embeddings,0,1)
        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TensorType[()]:
        """Return the same value as log_prob, but stored as a tensor."""
        
        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)

        # TODO: IMPLEMENT ME!
        # This method should call the logits helper method.
        # You are free to define other helper methods too.
        #
        # Be sure to use vectorization over the vocabulary to
        # compute the normalization constant Z, or this method
        # will be very slow. Some useful functions of pytorch that could
        # be useful are torch.logsumexp and torch.log_softmax.
        #raise NotImplementedError("Implement me!")
        x_embedding = torch.Tensor(self.word2vector[x])
        y_embedding = torch.Tensor(self.word2vector[y])
        #z_embedding = self.embeddings[self.word2id[z]]
        numerator = self.logits(x,y,z)
        denominator = torch.logsumexp(x_embedding @ self.X @ self.embeddings + y_embedding @ self.Y @ self.embeddings,0)
        #print(x_embedding.shape)
        #print(x_embedding.shape)
        #print(self.embeddings.shape)
        #print(numerator,denominator)
        return numerator - denominator
    
    def logits(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        """Return a vector of the logs of the unnormalized probabilities, f(xyz) * θ.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        # TODO: IMPLEMENT ME!
        # Don't forget that you can create additional methods
        # that you think are useful, if you'd like.
        # It's cleaner than making this function massive.
        #
        # The operator `@` is a nice way to write matrix multiplication:
        # you can write J @ K as shorthand for torch.mul(J, K).
        # J @ K looks more like the usual math notation.
        #
        # The return type, TensorType[()], represents a torch.Tensor scalar.
        # See Question 7 in INSTRUCTIONS.md for more info about fine-grained 
        # type annotations for Tensors.
        #raise NotImplementedError("Implement me!")
        
        #check if the word in the self.vocab
        #for word in (x,y,z):
        #    if word not in self.vocab:
        #        word = OOV
        x_embedding = torch.Tensor(self.word2vector[x]) if x in self.vocab else torch.Tensor(self.word2vector[OOV])
        y_embedding = torch.Tensor(self.word2vector[y]) if y in self.vocab else torch.Tensor(self.word2vector[OOV])
        z_embedding = torch.Tensor(self.word2vector[z]) if z in self.vocab else torch.Tensor(self.word2vector[OOV])
        return x_embedding @ self.X @ z_embedding + y_embedding @ self.Y @z_embedding

    def train(self, file: Path):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Optimization hyperparameters.
        gamma0 = 0.01 # initial learning rate

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=gamma0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        log.info("learning rate {gamma0}")
        log.info("Start optimizing on {N} training tokens...")

        #####################
        # TODO: Implement your SGD here by taking gradient steps on a sequence
        # of training examples.  Here's how to use PyTorch to make it easy:
        #
        # To get the training examples, you can use the `read_trigrams` function
        # we provided, which will iterate over all N trigrams in the training
        # corpus.
        #
        # For each successive training example i, compute the stochastic
        # objective F_i(θ).  This is called the "forward" computation. Don't
        # forget to include the regularization term. Part of F_i(θ) will be the
        # log probability of training example i, which the helper method
        # log_prob_tensor computes.  It is important to use log_prob_tensor
        # (as opposed to log_prob which returns float) because torch.Tensor
        # is an object with additional bookkeeping that tracks e.g. the gradient
        # function for backpropagation as well as accumulated gradient values
        # from backpropagation.
        #
        # To get the gradient of this objective (∇F_i(θ)), call the `backward`
        # method on the number you computed at the previous step.  This invokes
        # back-propagation to get the gradient of this number with respect to
        # the parameters θ.  This should be easier than implementing the
        # gradient method from the handout.
        #
        # Finally, update the parameters in the direction of the gradient, as
        # shown in Algorithm 1 in the reading handout.  You can do this `+=`
        # yourself, or you can call the `step` method of the `optimizer` object
        # we created above.  See the reading handout for more details on this.
        #
        # For the EmbeddingLogLinearLanguageModel, you should run SGD
        # optimization for 10 epochs and then stop.  You might want to print
        # progress dots using the `show_progress` method defined above.  Even
        # better, you could show a graphical progress bar using the tqdm module --
        # simply iterate over
        #     tqdm.tqdm(read_trigrams(file), total=10*N)
        # instead of iterating over
        #     read_trigrams(file)
        #####################
        epochs = 10
        for i in range(epochs):
            F=0 #keep track of the average of F_I over N
            #for trigram in read_trigrams(file, self.vocab):
            for trigram in tqdm.tqdm(read_trigrams(file,self.vocab), total=N):
                x, y,z = trigram
                regularizer = (self.l2 / N )* torch.sum(self.X*self.X +self.Y*self.Y) #l2 regularization bit-wise
                #print("regularizer",regularizer)
                #regularizer = 0
                F_i = self.log_prob_tensor(x,y,z)-regularizer
                F += F_i
                (-F_i).backward()
                optimizer.step()
                optimizer.zero_grad()
            print("epoch: ",i,"average F = ",F/N)
        
        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, devfiles: Path) -> None:
        super().__init__(vocab,lexicon_file,l2)
        if l2 < 0:
            log.error(f"l2 regularization strength value was {l2}")
            raise ValueError("You must include a non-negative regularization value")
        self.l2: float = l2
        self.devfiles = devfiles
        
        # TODO: READ THE LEXICON OF WORD VECTORS AND STORE IT IN A USEFUL FORMAT.
        #vocab need to include EOS and OOV
        self.vocab  = vocab
        self.vocab.add(EOS)
        self.vocab.add(OOV)
        #read lexicons file, and build a hashtbale mapping from word to vectors
        word2vector=read_lexicon(lexicon_file,self.vocab)
        self.word2vector = word2vector #hashmap
        
        #self.embeddings = torch.stack([self.word2vector[word] for word in self.vocab])
        self.embeddings = torch.Tensor([self.word2vector[word] for word in self.vocab])
        self.dim = self.embeddings.shape[1]  # TODO: SET THIS TO THE DIMENSIONALITY OF THE VECTORS
        self.embeddings = torch.transpose(self.embeddings,0,1)
        
        self.unigram_count = torch.Tensor([ self.event_count[word]+1 for word in self.vocab])
        
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Xoov = nn.Parameter(torch.zeros((self.dim)), requires_grad=True)
        self.Yoov = nn.Parameter(torch.zeros((self.dim)), requires_grad=True)
        self.Unigram = nn.Parameter(torch.zeros((1)), requires_grad=True)
        
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TensorType[()]:
        """Return the same value as log_prob, but stored as a tensor."""

        x_embedding = torch.Tensor(self.word2vector[x]) if x in self.vocab else torch.Tensor(self.word2vector[OOV])
        y_embedding = torch.Tensor(self.word2vector[y]) if y in self.vocab else torch.Tensor(self.word2vector[OOV])
        #z_embedding = self.embeddings[self.word2id[z]]
        numerator = self.logits(x,y,z)
        denominator = torch.logsumexp(x_embedding @ self.X @ self.embeddings +\
                                       y_embedding @ self.Y @ self.embeddings +\
                                      torch.log(self.unigram_count) * self.Unigram,0)
        
        add_oov_denominator = x_embedding @ self.Xoov + y_embedding @ self.Yoov
        
        
        denominator = torch.logsumexp(denominator+add_oov_denominator,0)
        #print(numerator,denominator)
        return numerator - denominator
    
    def logits(self, x: Wordtype, y: Wordtype, z: Wordtype) -> torch.Tensor:
        """Return a vector of the logs of the unnormalized probabilities, f(xyz) * θ.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        
        
        x_embedding = torch.Tensor(self.word2vector[x]) if x in self.vocab else torch.Tensor(self.word2vector[OOV])
        y_embedding = torch.Tensor(self.word2vector[y]) if y in self.vocab else torch.Tensor(self.word2vector[OOV])
        z_embedding = torch.Tensor(self.word2vector[z]) if z in self.vocab else torch.Tensor(self.word2vector[OOV])
        
        add_oov = 0 if z == OOV else x_embedding @ self.Xoov + y_embedding @ self.Yoov
        add_unigram = torch.log(torch.Tensor([self.event_count[z]+1])) * self.Unigram #log(c(z)+1)
        #print("event_count",torch.log(torch.Tensor([self.event_count[z]+1])))
        #print("self.Unigram",self.Unigram)
        #print("add_unigram",add_unigram)
        return x_embedding @ self.X @ z_embedding + y_embedding @ self.Y @z_embedding + add_oov+add_unigram
    
  
    #compute the cross entropy for development file, which can be used for early stopping
    def dev_cross_entropy(self, devfiles:Path):
        total_log_prob = 0.0
        total_tokens = 0
        for file in devfiles:
            for (x, y, z) in read_trigrams(file, self.vocab):
                total_log_prob += self.log_prob_tensor(x,y,z)  # log p(z | xy), no regularization 
                total_tokens+=1
        
        bits = -total_log_prob / math.log(2)   # convert to bits of surprisal
        return bits/total_tokens
        
    def train(self, file: Path):    # type: ignore
        
        #first collect the event_count
        self.event_count   = Counter()
        #self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
 
        
        # Optimization hyperparameters.
        gamma0 = 0.1  # initial learning rate
        lambda_ = 0.01 #decrease rate

        #optimizer = optim.SGD(self.parameters(), lr=gamma0)
        optimizer = ConvergentSGD(self.parameters(), gamma0, lambda_)
        
        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        log.info("learning rate {gamma0}")
        log.info("Start optimizing on {N} training tokens...")

            
        early_stopper = EarlyStopper(patience=2, min_delta=0)
        epochs = 50
        i=0
        for i in range(epochs):
            F=0 #keep track of the average of F_I over N
            #for trigram in read_trigrams(file, self.vocab):
            for trigram in tqdm.tqdm(draw_trigrams_forever(file,self.vocab,True), total=N):
                #because the draw_trigram model is forever, so we need to add a check
                i+=1
                if i >N:
                    break
                    
                x, y,z = trigram
                regularizer = (self.l2 / N )* torch.sum(self.X*self.X +self.Y*self.Y) #l2 regularization bit-wise
                #print("regularizer",regularizer)
                #regularizer = 0
                F_i = self.log_prob_tensor(x,y,z)-regularizer
                F += F_i
                (-F_i).backward()
                optimizer.step()
                optimizer.zero_grad()
            print("epoch: ",i,"average F = ",F/N)
            cross_entropy = self.dev_cross_entropy(self.devfiles)
            print("epoch: ",i,"CE per tokens = ",cross_entropy)
            #if early_stopper.early_stop(cross_entropy):             
            #    break

    log.info("done optimizing.")
    pass
