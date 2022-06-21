"""
Pulls examples from OSCAR or XNLI.
Note: pulls from the data in-order.
Sample usage:

python3 get_text_data.py --dataset="oscar" --output_dir="../../oscar_data" \
--max_examples=128000000

"""

from datasets import load_dataset
import codecs
import argparse
import math
import os
import re
from collections import Counter

# OSCAR notes:
#
# Hopefully the code cleans the following:
# Warning: ceb somewhat unclean.
# Warning: kv has some lines of random characters.
# Warning: new has some lines of random characters.
# Warning: oc has lots of random z and Z characters.
# Warning: sah has large blocks of black box characters (\u2588).
# Warning: sh has lots of ... characters.
#
# Note: some languages have characters that show up as red dots in some text editors:
# \u200b (zero width space) (ceb, cv, da, en, fi, fy, ga, gd, gu, hi, hr, it,
#                            km, lo, ml, my, nl, no, or, pa, sd, ta, tg, ur, yi)
# \u200c (zero width non-joiner) (as, azb, bn, ckb, fa, gom, kn, ml, my, mzn,
#                                 new, or, ps, sa, ta, te, ur)
# \u200d (‍zero width joiner) (azb, bn, gom, gu, hi, ky, ml, mr, ne, new,
#                             or, sa, si, ta, te, ug)
# \u200e (left-to-right mark) (ba, da, he, hi, mrj, mzn, new, os, sd, tg, xmf)
# \u200f (right-to-left mark) (ba, he, pnb, sd, ug)
# \ufeff (zero width no-break space) (az, be, ceb, gom, hi, is, pt, ro, th)
# \xad (soft hyphen) (az, ba, cv, fr, hsb, is, kk, krc, lez, mhr,
#                     mn, os, sk, tr, tt, ug)
# \x08 (backspace) (la)
# \x80 (control character) (en)
# \x85 (next line) (ce)
# \x94 (cancel) (no)
# \x98 (start of string) (fr)
# \x0... (vi, zh)
# \x1... (zh)
# \x9... (gn)
#
# All languages below have <2048 lines of data.
# Remove 'bar' because it has almost no data, and it is mostly random characters.
# Remove 'bs' because it has little data, and many gibberish lines.
# Remove 'cbk' because it has only two words.
# Remove 'diq' because it has only one line.
# Remove 'eml' because it has little data, and somewhat unclean.
# Remove 'frr' because it has almost no data, and it is mostly random characters.
# Remove 'gv' because it has almost no data, and very unclean.
# Remove 'ht' because it has almost no data, and very unclean.
# Remove 'ia' because it has very little data, and mostly unclean.
# Remove 'ie' because it has almost no data, and very unclean.
# Remove 'kw' because it has very little data, and mostly unclean.
# Remove 'li' because it has very little data, and several lines of random characters.
# Remove 'lrc' because it has only one line.
# Remove 'min' because it has little data, and mostly unknown characters.
# Remove 'mwl' because it has very little data, and somewhat unclean.
# Remove 'nap' because it has almost no data, and somewhat unclean.
# Remove 'rue' because it has only one line.
# Remove 'sco' because it has almost no data, and it is mostly random characters.
# Remove 'so' because it has almost no data, and it is mostly random characters.
# Remove 'vls' because it has only one line.
# Remove 'wuu' because it has very little data, and much of it seems redundant.
# Total: 146 languages.
LANGS_DICT = {'oscar': ['af', 'als', 'am', 'an', 'ar', 'arz', 'as', 'ast', 'av', 'az',
                    'azb', 'ba', 'be', 'bg', 'bh', 'bn', 'bo', 'bpy', 'br',
                    'bxr', 'ca', 'ce', 'ceb', 'ckb', 'cs', 'cv', 'cy',
                    'da', 'de', 'dsb', 'dv', 'el', 'en', 'eo', 'es',
                    'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl',
                    'gn', 'gom', 'gu', 'he', 'hi', 'hr', 'hsb', 'hu',
                    'hy', 'id', 'ilo', 'io', 'is', 'it', 'ja', 'jbo',
                    'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'krc', 'ku', 'kv',
                    'ky', 'la', 'lb', 'lez', 'lmo', 'lo', 'lt', 'lv',
                    'mai', 'mg', 'mhr', 'mk', 'ml', 'mn', 'mr', 'mrj', 'ms',
                    'mt', 'my', 'myv', 'mzn', 'nah', 'nds', 'ne', 'new',
                    'nl', 'nn', 'no', 'oc', 'or', 'os', 'pa', 'pam', 'pl', 'pms',
                    'pnb', 'ps', 'pt', 'qu', 'rm', 'ro', 'ru', 'sa', 'sah',
                    'scn', 'sd', 'sh', 'si', 'sk', 'sl', 'sq', 'sr',
                    'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr',
                    'tt', 'tyv', 'ug', 'uk', 'ur', 'uz', 'vec', 'vi', 'vo',
                    'wa', 'war', 'xal', 'xmf', 'yi', 'yo', 'zh'],
              'oscar_xnli': ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw',
                    'th', 'tr', 'ur', 'vi', 'zh'],
              'xnli': ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw',
                    'th', 'tr', 'ur', 'vi', 'zh']}

# Additional fixes for some languages with <1GB of data.
ADDITIONAL_CLEANING = True

def create_parser():
    parser = argparse.ArgumentParser()
    # The dataset to pull from: oscar or xnli.
    # Use oscar_xnli to pull data from OSCAR for the XNLI languages.
    parser.add_argument('--dataset', default="oscar")
    parser.add_argument('--output_dir')
    # Only supported for OSCAR. Skip examples before this index.
    parser.add_argument('--first_example', type=int, default=1)
    # Max examples per language.
    parser.add_argument('--max_examples', type=int, default=128000000)
    return parser


def main(args):
    if args.dataset == 'oscar' or args.dataset == 'oscar_xnli':
        hf_dataset = "oscar-corpus/OSCAR-2109"
        hf_dataset_split = 'train'
    elif args.dataset == 'xnli':
        hf_dataset = "xnli"
        hf_dataset_split = 'validation'
    else:
        print("Unrecognized dataset: {}".format(args.dataset))
        return
    langs = LANGS_DICT[args.dataset]
    hf_datasets_cache = "hf_datasets_cache"
    max_examples = math.inf if args.max_examples == -1 else args.max_examples

    os.mkdir(args.output_dir)
    for lang in langs:
        print("Running language: {}".format(lang))
        outpath = os.path.join(args.output_dir, "{}.txt".format(lang))
        if os.path.isfile(outpath):
            print("File already exists for language: {}".format(lang))
            continue
        dataset_subset = lang
        if args.dataset == 'oscar' or args.dataset == 'oscar_xnli':
            dataset_subset = "deduplicated_{}".format(lang)
        # Note: login with "transformers-cli login" first (must have installed
        # the transformers library, and activated the OSCAR 2109 dataset).
        dataset = load_dataset(hf_dataset, dataset_subset, streaming=True, split=hf_dataset_split,
                               use_auth_token=True, cache_dir=hf_datasets_cache)
        # Compile examples.
        outfile = codecs.open(outpath, 'wb', encoding='utf-8')
        example_count = 0
        if args.dataset == 'oscar' or args.dataset == 'oscar_xnli':
            # Remove lines where there are more than ten single characters separated by spaces.
            # This removes lines such as:
            # ö Ü Á Á Á Á Á Á É ö ü Á Á Á ö Á Í É Á Á ö ü ő ú ő ü ö ü ő ö ü...
            # which apear in some languages.
            reject_pattern1 = re.compile(".*\S \S \S \S \S \S \S \S \S \S \S \S \S.*")
            # Remove lines where there are at least ten ... separated by spaces.
            # Found in Serbian (Latin; sh).
            reject_pattern2 = re.compile(".* \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. \.\.\. .*")
            for example in dataset:
                # Split on newlines.
                split_example = example['text'].strip().replace("\\n", "\n").split('\n')
                for segment in split_example:
                    cleaned_segment = segment.strip() # No lowercase for now.
                    if len(cleaned_segment) == 0:
                        continue
                    cleaned_segment = cleaned_segment.replace("\\t", "\t")
                    cleaned_segment = cleaned_segment.replace("\t", " ")
                    if ADDITIONAL_CLEANING:
                        if "\u0648\u0648\u0648\u0648\u0648\u0648\u0648\u0648\u0648\u0648" in cleaned_segment:
                            # Skip some strange lines in Aragonese (an).
                            continue
                        if "zzzzzzzzzzzzzzzz" in cleaned_segment or "ZZZZZZZZZZZZZZZZ" in cleaned_segment:
                            # Skip some strange lines in Occitan (oc).
                            continue
                        if "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588" in cleaned_segment:
                            # Skip strange lines in Sakha (sah).
                            continue
                        if (reject_pattern1.match(cleaned_segment) is not None
                                or reject_pattern2.match(cleaned_segment) is not None):
                            continue
                    example_count += 1
                    if example_count >= args.first_example and example_count <= max_examples:
                        outfile.write(cleaned_segment)
                        outfile.write('\n')
                if example_count % 100000 == 0 and example_count != 0:
                    print("Read up to {} examples...".format(example_count))
                if example_count >= max_examples:
                    break
        elif args.dataset == 'xnli':
            for example in dataset:
                outfile.write(example['premise'].strip()) # No lowercase for now.
                outfile.write('\n')
                outfile.write(example['hypothesis'].strip()) # No lowercase for now.
                outfile.write('\n')
                example_count += 2
                if example_count >= max_examples:
                    break
        outfile.close()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
