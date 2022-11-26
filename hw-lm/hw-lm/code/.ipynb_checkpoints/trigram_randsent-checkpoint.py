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


#./trigram_randsent.py [mymodel] 10 --max_length 20
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "num_sentence",
        type=int,
        help ="number of generated sentence"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help ="maximum of length of generated sentence"
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


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    log.info("generate sentence:")
    for i in range (args.num_sentence):
        sent = lm.sample(args.max_length)
        print(sent)


if __name__ == "__main__":
    main()
