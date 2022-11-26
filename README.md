# Natural-Language-Processing-Coursework

## hw-grammar

• Understand how CFGs work and how they can be used to describe natural language
• Realize that describing natural language is hard
• Understand how parsers use probability to disambiguate sentences

## hw-lm

• estimating conditional probabilities from supervised data
– direct estimation of probabilities (with simple or backoff smoothing)
– conditional log-linear modeling (including feature engineering using external information such as lexicons)
– subtleties of language modeling (tokenization, EOS, OOV, OOL)
– subtleties of training (logarithms, autodiff, SGD, regularization)
• evaluating language models via sampling, perplexity, and multiple tasks, using a train/dev/test split
• tuning hyperparameters by hand to improve a formal evaluation metric
• implementing these methods cleanly in Python
– using basic facilities of PyTorch

## hw-parse

This is the primary algorithms homework for this class. After completing it,
you should be comfortable designing and implementing dynamic programming algorithms. You
should have a strong understanding of
• how to maintain a record of the items (partial constituents) that have been built so far
• how to maintain each item's optimal derivation and its weight
• how to index these data structures for quick lookup when building new items
• how to organize the computation so that all possible items will eventually get built
• how to speed up the algorithm by skipping unnecessary work (pruning/prioritization/ ltering), avoiding duplicate work (merging items), and improving
the data structures (indexing)
• why the algorithm runs in O(n3), and what changes would put that at risk

## hw-sem

This short homework is the main homework that deals with formal
representations of meaning. After doing it, you should be comfortable
• manipulating  -calculus expressions formally
• building them up compositionally using semantic attachments to syntactic rules
• considering whether they appropriately capture the meaning of a phrase

