"""
Gets the overall mean representation across languages given the language-specific means.
This weights each language equally, regardless of the number of
tokens in each language.
Sample usage:

python3 get_global_mean.py --layer=8 \
--input_dir="../../oscar_xlmr_eval/subspace_cache" \
--output="../../oscar_xlmr_eval/global_layer8_mean.npy"

"""

import numpy as np
import os
import argparse
from utils import OSCAR_XLMR_LANGS
CONSIDERED_LANGS = XLMR_OSCAR_LANGS


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir')
    parser.add_argument('--output')
    parser.add_argument('--layer', type=int, default=8)
    return parser


def main(args):
    print("Reading language means...")
    means = []
    for lang in CONSIDERED_LANGS:
        # Shape: (dim_size).
        mean = np.load(os.path.join(args.input_dir, "{0}_layer{1}_mean.npy".format(lang, args.layer)))
        means.append(mean)
    means_tensor = np.stack(means, axis=0) # Shape: (n_langs, dim_size).
    global_mean = np.mean(means_tensor, axis=0)
    np.save(args.output, global_mean, allow_pickle=False)
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
