"""
Deduplicates lines in text files in a directory.
Sample usage:

python3 deduplicate_examples.py \
--input_dir="../../xnli_eval" --output_dir="../../xnli_eval_deduplicated"

"""

import argparse
import os
import codecs

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')
    return parser

def deduplicate_file(input_path, output_path):
    print("Deduplicating file: {}".format(input_path))
    if os.path.isfile(output_path):
        print("File already exists: {}".format(output_path))
        return
    infile = codecs.open(input_path, 'rb', encoding='utf-8')
    deduplicated_lines = set(infile.readlines())
    infile.close()
    outfile = codecs.open(output_path, 'wb', encoding='utf-8')
    for line in deduplicated_lines:
        outfile.write(line)
    outfile.close()
    return

def main(args):
    for root, dirs, files in os.walk(args.input_dir):
        for fname in files:
            inpath = os.path.join(root, fname)
            outpath = os.path.join(args.output_dir, fname)
            deduplicate_file(inpath, outpath)
    print("Done.")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
