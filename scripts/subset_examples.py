"""
Collects subsets of tokenized sequences from each text file in a directory.
Each line in each text file should be a space-separated list of token ids.
Outputs a pickle file for each text file, with a list of lists (each representing
a subset) of tokenized sequences (integer lists).
Sample usage:

python3 subset_examples.py --input_dir="../../oscar_xlmr_tokenized" \
--output_dir="../../oscar_xlmr_tokenized_subsets/512_examples" \
--max_examples=512 --num_subsets=1

"""

import argparse
import os
import codecs
import numpy as np
import math
import pickle

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--max_examples', type=int, default=512)
    # Number of random subsets per input file.
    parser.add_argument('--num_subsets', type=int, default=1)
    return parser

def get_examples_subsets(inpath, max_examples, infile_len, num_subsets):
    examples = [[] for _ in range(num_subsets)]
    # Get an iterable of example indices for each subset.
    if max_examples > infile_len:
        print("WARNING: max_examples > infile_length. Using all examples.")
        indices = [iter(np.arange(infile_len)) for _ in range(num_subsets)]
    else:
        np.random.seed()
        indices = None
        if indices is None: # Note: not necessarily disjoint subsets.
            indices = [np.random.choice(infile_len, max_examples, replace=False) for _ in range(num_subsets)]
        indices = [iter(np.sort(subset_indices)) for subset_indices in indices]
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    # Maintain a list of target indices to search for.
    target_indices = [next(subset_indices) for subset_indices in indices]
    for line_i, tokenized_example in enumerate(infile):
        if line_i % 100000 == 0:
            print("Read example {}".format(line_i))
        for subset_i, target_index in enumerate(target_indices):
            if line_i == target_index:
                # Add example to the target subset and update the target indices.
                example = [int(token_id) for token_id in tokenized_example.split(" ")]
                examples[subset_i].append(example)
                target_indices[subset_i] = next(indices[subset_i], -1)
    infile.close()
    for subset_i in range(num_subsets):
        if not math.isinf(max_examples):
            assert (len(examples[subset_i]) == max_examples or
                    len(examples[subset_i]) == infile_len), "Examples not pulled correctly."
    return examples

def main(args):
    max_examples = math.inf if args.max_examples == -1 else args.max_examples
    os.mkdir(args.output_dir)
    infile_lengths = dict()
    for root, dirs, files in os.walk(args.input_dir):
        for fname in files:
            inpath = os.path.join(root, fname)
            print("Reading file: {}".format(inpath))
            fprefix = fname.replace(".txt", "")
            outpath = os.path.join(args.output_dir, "{}.pickle".format(fprefix))
            if os.path.isfile(outpath):
                print("File already exists: {}".format(outpath))
                continue
            # Get length of input file (total number of examples available).
            if inpath not in infile_lengths:
                print("Counting examples: {}".format(inpath))
                infile = codecs.open(inpath, 'rb', encoding='utf-8')
                line_count = 0
                for line in infile:
                    line_count += 1
                    if line_count % 100000 == 0:
                        print("Counted {} examples...".format(line_count))
                infile.close()
                print("{} examples.".format(line_count))
                infile_lengths[inpath] = line_count
            infile_len = infile_lengths[inpath]
            examples = get_examples_subsets(inpath, max_examples, infile_len, args.num_subsets)
            with open(outpath, 'wb') as handle:
                pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
