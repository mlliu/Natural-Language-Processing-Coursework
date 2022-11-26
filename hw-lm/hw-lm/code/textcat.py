#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to the trained model_1",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to the trained model_2",
    )
    parser.add_argument(
        "prior_1",
        type=float,
        help="prior probability of model_1",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
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

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1)
    lm_2 = LanguageModel.load(args.model_2)
    
    #check if lm_1 and lm_2 use the same vocabuary
    try:
        assert(lm_1.vocab == lm_2.vocab)
    except:
        log.error("this two language should share the same vocab set")
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("Per-file post-probabilities:")
    #total_log_prob = 0.0
    count_1 =0
    for file in args.test_files:
        log_prob_1: float = file_log_prob(file, lm_1)
        log_prob_2: float = file_log_prob(file, lm_2)
        #print(log_prob_1,log_prob_2)
       
        #unnomilized postrerior probability for lm1 and lm2 
        unP_lm1_text: float = log_prob_1 +math.log(args.prior_1)
        unP_lm2_text: float = log_prob_2 +math.log(1-args.prior_1)
        
        # choose the class with the maximum posteriori
        
        if unP_lm1_text >= unP_lm2_text:
            print(f"{args.model_1}\t{file}")
            count_1 += 1
        else:
            print(f"{args.model_2}\t{file}")
        #total_log_prob += log_prob
    #print the total information
    count_2 = len(args.test_files)-count_1
    print(f"{count_1} files were more probably {args.model_1} ({count_1/len(args.test_files):.0%})")
    print(f"{count_2} files were more probably {args.model_2} ({count_2/len(args.test_files):.0%})")
    
    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    #bits = -total_log_prob / math.log(2)   # convert to bits of surprisal
    #tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    #print(f"Overall cross-entropy:\t{bits / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()