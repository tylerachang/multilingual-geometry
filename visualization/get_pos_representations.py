"""
Get representations for different parts-of-speech (POS).
Outputs a pickled dictionary for each POS, mapping languages to representations
in that language for the given POS. Representations must have already been
saved in RAW_REPS_DIR from corresponding tokenized examples in
TOKENIZED_EXAMPLES_DIR (see readme and extract_representations.py).
Please ensure that the directory locations are correct in directories.py.
Sample usage:

python3 get_pos_representations.py \
--model_name_or_path="xlm-roberta-base" --max_seq_length 512 \
--n_per_language=1024 --layer=8 \
--pos ADJ ADP ADV AUX DET NOUN VERB PRON PROPN

"""

import os
import argparse
import numpy as np
import codecs
import pickle
from collections import defaultdict
from transformers import AutoTokenizer

from visualization_constants import RAW_REPS_DIR, TOKENIZED_SUBSETS_DIR, UD_POS_DICT_PATH, CONSIDERED_LANGS, POS_REPS_DIR


def create_parser():
    parser = argparse.ArgumentParser()
    # Number of representations per language and per POS.
    parser.add_argument('--n_per_language', type=int)
    parser.add_argument('--layer', type=int)
    parser.add_argument('--pos', nargs='+', type=str)
    # Used to identify the correct tokenizer, to map token ids to POS.
    parser.add_argument('--model_name_or_path', type=str, default="xlm-roberta-base")
    parser.add_argument('--do_lower_case', type=bool, default=False)
    parser.add_argument('--max_seq_length', type=int, default=512)
    return parser

# Gets the POS tags for each token index.
# Returns a list of lists (one for each token index):
# e.g. pos_list[10] = ["ADJ", "VERB"].
# Uses the universal tagset (https://universaldependencies.org/u/pos/).
# Takes as input the path to a tsv dictionary mapping words to space-separated
# POS tags (generated from the Universal Dependencies dataset).
def get_pos_list(tokenizer, ud_dict_file):
    # Default to unknown POS.
    pos_list = [['Unknown'] for _ in range(len(tokenizer))] # Include special tokens.
    # Read ud_dict.
    word_to_pos = dict()
    infile = codecs.open(ud_dict_file, 'rb', encoding='utf-8')
    for line in infile:
        word, pos = tuple(line.strip().split('\t'))
        word_to_pos[word] = pos.split()
    infile.close()
    # Map tokens through word_to_pos to POS.
    for id in range(tokenizer.vocab_size):
        token = tokenizer.decode(id)
        token = token.replace("\u2581", "").strip().lower()
        if token in word_to_pos:
            pos_list[id] = word_to_pos[token] # Replace the POS list at token_id.
    return pos_list


def main(args):
    tokenizer_cache_dir = "tokenizer_cache"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=tokenizer_cache_dir,
    )
    pos_list = get_pos_list(tokenizer, UD_POS_DICT_PATH)
    def is_pos(token_id, pos):
        return pos in pos_list[token_id]

    # For each POS, map languages to representations.
    pos_reps_dict = defaultdict(dict)
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
        # Get masks corresponding to each POS.
        pos_masks = np.zeros((len(args.pos), lang_reps.shape[0]), dtype=bool)
        # Find the original token corresponding to each representation.
        example_i = 0 # Example index.
        example_token_i = 0 # Token index within example.
        for overall_token_i in range(lang_reps.shape[0]):
            token_id = examples[example_i][example_token_i]
            for pos_i, pos in enumerate(args.pos):
                pos_masks[pos_i, overall_token_i] = is_pos(token_id, pos)
            # Increment.
            example_token_i += 1
            if example_token_i >= len(examples[example_i]):
                example_i += 1
                example_token_i = 0
        # Get the representation subset for each POS.
        for pos_i, pos in enumerate(args.pos):
            pos_reps = lang_reps[pos_masks[pos_i]]
            # Select a random subset of those representations.
            print("{0} reps: {1}".format(pos, pos_reps.shape[0]), end="")
            if pos_reps.shape[0] > args.n_per_language:
                to_keep = np.random.choice(pos_reps.shape[0], size=args.n_per_language, replace=False)
                to_keep_bool = np.zeros(pos_reps.shape[0], dtype=bool)
                to_keep_bool[to_keep] = True
                pos_reps = pos_reps[to_keep_bool]
                print(" -> {} reps".format(pos_reps.shape[0]))
            pos_reps_dict[pos][lang] = pos_reps.copy()
        del lang_reps
        print("Finished {}.".format(lang))
    if not os.path.isdir(POS_REPS_DIR):
        os.mkdir(POS_REPS_DIR)
    for pos in args.pos:
        with open(os.path.join(POS_REPS_DIR, "{0}_layer{1}_reps.pickle".format(pos, args.layer)), 'wb') as handle:
            pickle.dump(pos_reps_dict[pos], handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
