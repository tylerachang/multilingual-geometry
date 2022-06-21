"""
Counts tokens in tokenized text files in a directory.
For each input file, outputs an npy file containing a tensor of per-1000 token frequencies.
Sample usage:

python3 count_tokens.py --vocab_size=250002 \
--input_dir="../../oscar_xlmr_tokenized" \
--output_dir="../../oscar_xlmr_token_counts" \
--max_seq_len=512

"""

import argparse
import os
import codecs
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--vocab_size', type=int, default=250002)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--max_tokens', type=int, default=1000000000)
    return parser


def tokenize_file(input_path, output_path, vocab_size, max_seq_len, max_tokens):
    print("Counting file: {}".format(input_path))
    if os.path.isfile(output_path):
        print("File already exists: {}".format(output_path))
        return
    infile = codecs.open(input_path, 'rb', encoding='utf-8')
    example_count = 0
    total_token_count = 0
    token_counts = np.zeros(vocab_size, dtype=np.int64)
    for line_i, line in enumerate(infile):
        example_count += 1
        example = [int(token_id) for token_id in line.strip().split(" ")]
        example = example[:max_seq_len]
        for token_id in example:
            token_counts[token_id] += 1
        total_token_count += len(example)
        if total_token_count >= max_tokens:
            break
        if line_i % 100000 == 0:
            print("Read {} examples...".format(line_i))
    infile.close()
    # Per 1000 token frequencies.
    token_counts = (token_counts / float(total_token_count)) * 1000.0
    np.save(output_path, token_counts, allow_pickle=False)
    print("Finished counting: {} tokens.".format(total_token_count))
    return


def main(args):
    if os.path.isdir(args.output_dir):
        print("WARNING: existing output directory.")
    else:
        os.mkdir(args.output_dir)
    for root, dirs, files in os.walk(args.input_dir):
        for fname in files:
            inpath = os.path.join(root, fname)
            outname = fname.replace(".txt", ".npy")
            outpath = os.path.join(args.output_dir, outname)
            tokenize_file(inpath, outpath, args.vocab_size, args.max_seq_len, args.max_tokens)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
