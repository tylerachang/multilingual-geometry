"""
Gets distances between subspaces rotated by n degrees.
For each layer, outputs a pickled dictionary mapping angles to distance values
(for different languages and different random rotations).
Sample usage:

python3 get_rotated_subspace_distances.py \
--subspace_cache="../oscar_xlmr_eval/subspace_cache" \
--output_dir="../oscar_xlmr_eval/subspace_distances/rotations"

"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

from src.utils import XLMR_OSCAR_LANGS
from src.distances import subspace_distance, random_rotation_matrix

# Choose angles and number of random rotations to evaluate.
ANGLES = range(0, 91) # Angles in degrees.
LAYERS = range(0, 13)
N_ROTATIONS = 16 # Random rotations per angle.
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
        for angle in ANGLES:
            angle_distances = np.zeros((n_langs, N_ROTATIONS))
            for rotation_i in range(N_ROTATIONS):
                print("Running rotation {0} for angle {1} (layer {2}).".format(rotation_i, angle, layer))
                rotation, _ = random_rotation_matrix(DIM_SIZE, angle=angle)
                for lang_i in tqdm(range(n_langs)):
                    distance = subspace_distance(subspaces[lang_i], np.matmul(rotation, subspaces[lang_i]),
                                          metric=DISTANCE_METRIC, s=(s_vectors[lang_i], s_vectors[lang_i]),
                                          dim_a=DIM_SIZE, dim_b=DIM_SIZE)
                    angle_distances[lang_i, rotation_i] = distance
                print("Mean distance: {}".format(np.mean(angle_distances[:, rotation_i])))
            distances[angle] = angle_distances.copy()
        distances_file = os.path.join(args.output_dir, "rotation_distances_layer{0}.pickle".format(layer))
        with open(distances_file, 'wb') as handle:
            pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
