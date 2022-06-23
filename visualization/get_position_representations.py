"""
Get representations for different token position indices.
Outputs a pickled dictionary for each position, mapping languages to
representations in that language for the given position. Representations must
have already been saved in RAW_REPS_DIR from corresponding tokenized examples in
TOKENIZED_EXAMPLES_DIR (see readme and extract_representations.py).
Please ensure that the directory locations are correct in directories.py.
Sample usage:

python3 get_position_representations.py \
--max_seq_length 512 \
--n_per_language=256 --layer=8

"""

import os
import argparse
import numpy as np
import pickle
from collections import defaultdict

from visualization_constants import RAW_REPS_DIR, TOKENIZED_SUBSETS_DIR, CONSIDERED_LANGS, POSITION_REPS_DIR, POSITIONS


def create_parser():
    parser = argparse.ArgumentParser()
    # Number of representations per language and per position.
    parser.add_argument('--n_per_language', type=int)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--max_seq_length', type=int, default=512)
    return parser


def main(args):
    # For each position, map languages to representations.
    position_reps_dict = defaultdict(dict)
    for lang in CONSIDERED_LANGS:
        # Load the representations for the language.
        lang_reps = np.load(os.path.join(RAW_REPS_DIR, "{0}_layer{1}_reps.npy".format(lang, args.layer)), allow_pickle=False)
        print("Loaded {0} reps for {1}.".format(lang_reps.shape[0], lang))
        # Load the corresponding tokenized examples.
        pickled_subsets_file = os.path.join(TOKENIZED_SUBSETS_DIR, "{}.pickle".format(lang))
        with open(pickled_subsets_file, 'rb') as handle:
            examples_subsets = pickle.load(handle)
        print("{0} example subsets for language {1}.".format(len(examples_subsets), lang))
        examples = examples_subsets[0] # Assume representations were extracted from the first subset of examples.
        for example_i in range(len(examples)): # Truncate examples.
            if len(examples[example_i]) > args.max_seq_length:
                examples[example_i] = examples[example_i][:args.max_seq_length]
        # Get masks corresponding to each position.
        position_masks = np.zeros((len(POSITIONS), lang_reps.shape[0]), dtype=bool)
        # Find the original token corresponding to each representation.
        example_i = 0 # Example index.
        example_token_i = 0 # Token index within example.
        for overall_token_i in range(lang_reps.shape[0]):
            for position_i, position_idx in enumerate(POSITIONS):
                position_masks[position_i, overall_token_i] = example_token_i == position_idx # Keep if correct position.
            # Increment.
            example_token_i += 1
            if example_token_i >= len(examples[example_i]):
                example_i += 1
                example_token_i = 0
        # Get the representation subset for each position.
        for position_i, position_idx in enumerate(POSITIONS):
            position_reps = lang_reps[position_masks[position_i]]
            # Select a random subset of those representations.
            print("Position {0} reps: {1}".format(position_idx, position_reps.shape[0]), end="")
            if position_reps.shape[0] > args.n_per_language:
                to_keep = np.random.choice(position_reps.shape[0], size=args.n_per_language, replace=False)
                to_keep_bool = np.zeros(position_reps.shape[0], dtype=bool)
                to_keep_bool[to_keep] = True
                position_reps = position_reps[to_keep_bool]
                print(" -> {} reps".format(position_reps.shape[0]))
            position_reps_dict[position_idx][lang] = position_reps.copy()
        del lang_reps
        print("Finished {}.".format(lang))
    if not os.path.isdir(POSITION_REPS_DIR):
        os.mkdir(POSITION_REPS_DIR)
    for position_idx in POSITIONS:
        with open(os.path.join(POSITION_REPS_DIR, "position{0}_layer{1}_reps.pickle".format(position_idx, args.layer)), 'wb') as handle:
            pickle.dump(position_reps_dict[position_idx], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
