"""
Gets distances between subspaces scaled by some proportion in each dimension.
For each layer, outputs a pickled dictionary mapping scaling multipliers to
distance values (for different languages and different random scalings).
Sample usage:

python3 get_scaled_subspace_distances.py \
--subspace_cache="../oscar_xlmr_eval/subspace_cache" \
--output_dir="../oscar_xlmr_eval/subspace_distances/scalings"

"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

from src.utils import XLMR_OSCAR_LANGS
from src.distances import subspace_distance, random_scaling

# Choose scales and number of random scalings to evaluate.
SCALES = np.linspace(1.00, 4.00, num=301, endpoint=True) # Scale multipliers.
LAYERS = range(0, 13)
N_SCALINGS = 16 # Random scalings per scale. Default 16.
DISTANCE_METRIC = "riemann_full_distance"
DIM_SIZE = 768


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subspace_cache')
    parser.add_argument('--output_dir')
    return parser


def main(args):
    distances = dict()
    n_langs = len(XLMR_OSCAR_LANGS)
    for layer in LAYERS:
        # Load subspaces.
        subspaces = np.zeros((n_langs, DIM_SIZE, DIM_SIZE))
        s_vectors = np.zeros((n_langs, DIM_SIZE))
        for lang_i, lang in tqdm(enumerate(XLMR_OSCAR_LANGS)):
            vh = np.load(os.path.join(args.subspace_cache, "{0}_layer{1}_vh.npy".format(lang, layer)))
            subspaces[lang_i, :, :] = np.transpose(vh)
            s = np.load(os.path.join(args.subspace_cache, "{0}_layer{1}_s.npy".format(lang, layer)))
            # Don't need to scale s_vectors by sqrt(n_reps-1) because we are only comparing within languages (same n_reps),
            # and the distance metric is invariant to scaling.
            s_vectors[lang_i, :] = s
        # Compute distances.
        for scale in SCALES:
            scale_distances = np.zeros((n_langs, N_SCALINGS))
            for scaling_i in range(N_SCALINGS):
                print("Running scaling {0} for scale {1} (layer {2}).".format(scaling_i, scale, layer))
                scaling = random_scaling(DIM_SIZE, scale=scale)
                for lang_i in tqdm(range(n_langs)):
                    # Can apply square root to scaling to scale the variance (proportional to s^2) directly.
                    # Without the square root (by default), the variance is scaled by the square of the scaling.
                    # This is equivalent to if the original points were scaled along the corresponding
                    # dimensions (which scales the variance by the square of the scaling).
                    distance = subspace_distance(subspaces[lang_i], subspaces[lang_i],
                                          metric=DISTANCE_METRIC, s=(s_vectors[lang_i], np.multiply(scaling, s_vectors[lang_i])),
                                          dim_a=DIM_SIZE, dim_b=DIM_SIZE)
                    scale_distances[lang_i, scaling_i] = distance
                print("Mean distance: {}".format(np.mean(scale_distances[:, scaling_i])))
            distances[scale] = scale_distances.copy()
        distances_file = os.path.join(args.output_dir, "scaling_distances_layer{0}.pickle".format(layer))
        with open(distances_file, 'wb') as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
