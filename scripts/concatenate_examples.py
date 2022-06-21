"""
Concatenates lines in text files in a directory.
Sample usage:

python3 concatenate_examples.py \
--input_dir="text_files" --output_file="concatenated_text_file.txt"

"""

import argparse
import os
import codecs

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output_file')
    return parser

def main(args):
    if os.path.isfile(args.output_file):
        print("File already exists: {}".format(args.output_file))
        return
    outfile = codecs.open(args.output_file, 'wb', encoding='utf-8')
    for root, dirs, files in os.walk(args.input_dir):
        for fname in files:
            inpath = os.path.join(root, fname)
            print("Reading file: {}".format(inpath))
            infile = codecs.open(inpath, 'rb', encoding='utf-8')
            for line in infile:
                stripped_line = line.strip()
                if len(stripped_line) == 0:
                    continue
                outfile.write(stripped_line)
                outfile.write('\n')
            infile.close()
    outfile.close()
    print("Done.")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
