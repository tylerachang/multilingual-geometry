"""
Get raw distances between (mean-centered) language subspaces.
Note: subspace distances are dependent on the representation variances.
To compute these variances from the cached singular values \sigma, we need to
know the number of representations originally used to compute each language
subspace. These values should be included manually in the N_REPS variable.
This only affects distance values if different languages used different numbers
of representations.
Sample usage:

python3 compute_distances.py \
--subspace_cache="../oscar_xlmr_eval/subspace_cache" \
--output_dir="../oscar_xlmr_eval/subspace_distances" \
--total_dims=768 --layers 8

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from itertools import combinations
from collections import defaultdict

from src.distances import subspace_distance
from src.utils import OSCAR_XLMR_LANGS, XNLI_LANGS
CONSIDERED_LANGS = XNLI_LANGS
DISTANCE_METRIC = "riemann_full_distance"

# The number of representations used to compute the subspace for each
# considered language.
N_REPS = defaultdict(lambda: 512 * 512) # Most languages had 512*512 reps.
N_REPS["su"] = 512 * 137
N_REPS["jv"] = 512 * 377


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subspace_cache')
    parser.add_argument('--output_dir')
    parser.add_argument('--total_dims', type=int, default=768)
    parser.add_argument('--layers', nargs='+', type=int)
    return parser

def main(args):
    n_langs = len(CONSIDERED_LANGS)
    for layer in args.layers:
        distances = np.zeros((n_langs, n_langs))
        subspaces_v = np.zeros((n_langs, args.total_dims, args.total_dims))
        subspaces_s = np.zeros((n_langs, args.total_dims))
        for lang_i, lang in tqdm(enumerate(CONSIDERED_LANGS)):
            vh = np.load(os.path.join(args.subspace_cache, "{0}_layer{1}_vh.npy".format(lang, layer)))
            if vh.shape[0] < args.total_dims:
                # There can be fewer basis vectors than the original representation dimensionality if the
                # subspace was computed from n tokens, where n < original_dimensionality.
                # This should not happen in practice.
                print("WARNING: skipping language ({0}) because not enough basis vectors ({1}).".format(lang, vh.shape[0]))
                distances[:, lang_i] = -1.0
                distances[lang_i, :] = -1.0
                continue
            subspaces_v[lang_i, :, :] = np.transpose(vh)
            s = np.load(os.path.join(args.subspace_cache, "{0}_layer{1}_s.npy".format(lang, layer)))
            # Scale s down by sqrt(n_reps-1). Then, the squared value will be the variance along the corresponding axis.
            s = s / np.sqrt(N_REPS[lang]-1)
            subspaces_s[lang_i, :] = s
        # Compute distances.
        subspace_pairs = list(combinations(range(n_langs), 2))
        print("Computing distances...")
        for lang_i, lang_j in tqdm(subspace_pairs):
            if distances[lang_i, lang_j] == -1.0 or distances[lang_j, lang_i] == -1.0:
                continue
            distance = subspace_distance(subspaces_v[lang_i], subspaces_v[lang_j],
                                         metric=DISTANCE_METRIC, s=(subspaces_s[lang_i], subspaces_s[lang_j]),
                                         dim_a=args.total_dims, dim_b=args.total_dims)
            distances[lang_i, lang_j] = distance
            distances[lang_j, lang_i] = distance
        distances_file = os.path.join(args.output_dir, "oscar_xlmr_distances_layer{0}.npy".format(layer))
        np.save(distances_file, distances, allow_pickle=False)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
