"""
Utilities for visualization code.
"""

import os
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from visualization_constants import (RAW_REPS_DIR, TOKENIZED_SUBSETS_DIR, POS_REPS_DIR,
    POSITION_REPS_DIR, CACHED_VECTORS_DIR, SUBSPACE_CACHE_DIR, CONSIDERED_LANGS, POSITIONS)

# Get orthonormal component of a vector relative to an existing basis.
# Vector (d) and axis_directions (d x k).
def get_orthonormal(vector, axis_directions):
    proj_matrix = np.linalg.inv(np.matmul(axis_directions.T, axis_directions))
    proj_matrix = np.matmul(np.matmul(axis_directions, proj_matrix), axis_directions.T)
    projected_component = np.matmul(proj_matrix, vector)
    orthogonal_component = vector - projected_component
    return orthogonal_component / np.linalg.norm(orthogonal_component)

# Input: axis directions in the original space (d x k; orthogonalized),
# points to project from the original space (n x d), and
# the subspace origin in the original space (d).
# The subspace origin only shifts the projected points by a constant vector.
# Returns the projected points (n x k).
def project_reps(axis_directions, points, subspace_m):
    subspace_k = axis_directions.shape[1]
    inner_products = np.dot(axis_directions.T, axis_directions) # k x k
    if not np.allclose(inner_products, np.identity(subspace_k)):
        print("WARNING: basis not orthonormal.")
    projected_points = np.matmul(axis_directions.T, (points - subspace_m.reshape(1, -1)).T).T
    return projected_points

# Get randomly sampled representations with corresponding token information (original
# input sequence, token position idx) for a language. The token_info_tuples are
# a list of tuples, where each tuple contains information about the representation
# at the corresponding index.
def get_reps_with_token_info(lang, layer, n_reps, max_seq_length=512):
    # Load representations.
    reps = np.load(os.path.join(RAW_REPS_DIR, "{0}_layer{1}_reps.npy".format(lang, layer)), allow_pickle=False)
    to_keep = np.random.choice(reps.shape[0], size=n_reps, replace=False)
    to_keep_bool = np.zeros(reps.shape[0], dtype=bool)
    to_keep_bool[to_keep] = True
    reps_to_return = reps[to_keep_bool].copy()
    del reps
    # Each tuple contains the original sequence (list of token ids), and the token position.
    token_info_tuples = []
    pickled_subsets_file = os.path.join(TOKENIZED_SUBSETS_DIR, "{}.pickle".format(lang))
    with open(pickled_subsets_file, 'rb') as handle:
        examples_subsets = pickle.load(handle)
    examples = examples_subsets[0] # Assume representations were extracted from the first subset of examples.
    for example_i in range(len(examples)): # Truncate examples.
        if len(examples[example_i]) > max_seq_length:
            examples[example_i] = examples[example_i][:max_seq_length]
    # Find the original token corresponding to each representation.
    example_i = 0 # Example index.
    example_token_i = 0 # Token index within example.
    for kept in to_keep_bool:
        if kept:
            token_info_tuples.append(tuple([examples[example_i], example_token_i]))
        # Increment.
        example_token_i += 1
        if example_token_i >= len(examples[example_i]):
            example_i += 1
            example_token_i = 0
    assert reps_to_return.shape[0] == len(token_info_tuples), "ERROR: token_info_tuples should correspond to representations."
    return reps_to_return, token_info_tuples

# Get global mean (across languages) if possible.
def get_global_mean(layer):
    global_mean_file = os.path.join(CACHED_VECTORS_DIR, "global_layer{0}_mean.npy".format(layer))
    if os.path.isfile(global_mean_file):
        return np.load(global_mean_file).reshape(-1)
    else:
        print("Unable to find global representation mean in CACHED_VECTORS_DIR; using default origin.")
        return np.zeros(1)

# Get LDA directions separating parts of speech.
# Input: list of parts-of-speech, layer index.
# Must have run get_pos_representations.py first.
def get_pos_lda(parts_of_speech, layer, n_per_pos=8192):
    all_pos_reps = []
    all_pos_labels = []
    for pos_i, pos in enumerate(parts_of_speech):
        # Get reps for one POS.
        with open(os.path.join(POS_REPS_DIR, "{0}_layer{1}_reps.pickle".format(pos, layer)), 'rb') as handle:
            pos_reps = pickle.load(handle)
        pos_reps = np.concatenate(list(pos_reps.values()), axis=0) # Concatenate across languages.
        if n_per_pos < pos_reps.shape[0]:
            to_keep = np.random.choice(pos_reps.shape[0], size=n_per_pos, replace=False)
            pos_reps = pos_reps[to_keep]
        all_pos_reps.append(pos_reps)
        all_pos_labels.append(np.ones(pos_reps.shape[0], dtype=np.int32) * pos_i)
    all_pos_reps = np.concatenate(all_pos_reps, axis=0) # Concatenate across parts-of-speech.
    all_pos_labels = np.concatenate(all_pos_labels, axis=0)
    lda = LinearDiscriminantAnalysis()
    lda.fit(all_pos_reps, all_pos_labels)
    # Shape: (n_dims, n_classes-1)
    lda_axes = lda.scalings_
    return lda_axes

# Get LDA directions separating position indices.
# Must have run get_position_representations.py first.
def get_position_lda(positions, layer, n_per_position=8192):
    all_position_reps = []
    all_position_labels = []
    for position_i, position_idx in enumerate(positions):
        with open(os.path.join(POSITION_REPS_DIR, "position{0}_layer{1}_reps.pickle".format(position_idx, layer)), 'rb') as handle:
            position_reps = pickle.load(handle)
        position_reps = np.concatenate(list(position_reps.values()), axis=0) # Concatenate across languages.
        if n_per_position < position_reps.shape[0]:
            to_keep = np.random.choice(position_reps.shape[0], size=n_per_position, replace=False)
            position_reps = position_reps[to_keep]
        all_position_reps.append(position_reps)
        all_position_labels.append(np.ones(position_reps.shape[0], dtype=np.int32) * position_i)
    all_position_reps = np.concatenate(all_position_reps, axis=0) # Concatenate across positions.
    all_position_labels = np.concatenate(all_position_labels, axis=0)
    lda = LinearDiscriminantAnalysis()
    lda.fit(all_position_reps, all_position_labels)
    # Shape: (n_dims, n_classes-1)
    lda_axes = lda.scalings_
    if not os.path.isdir(CACHED_VECTORS_DIR):
        os.mkdir(CACHED_VECTORS_DIR)
    np.save(os.path.join(CACHED_VECTORS_DIR, "position_lda_layer{}.npy".format(layer)), lda_axes, allow_pickle=False)
    return lda_axes

# Get LDA directions separating languages.
def get_lang_lda(langs, layer, n_per_lang=4096):
    all_lang_reps = []
    all_lang_labels = []
    for lang_i, lang in enumerate(langs):
        lang_reps = np.load(os.path.join(RAW_REPS_DIR, "{0}_layer{1}_reps.npy".format(lang, layer)), allow_pickle=False)
        if n_per_lang < lang_reps.shape[0]:
            to_keep = np.random.choice(lang_reps.shape[0], size=n_per_lang, replace=False)
            lang_reps = lang_reps[to_keep]
        all_lang_reps.append(lang_reps)
        all_lang_labels.append(np.ones(lang_reps.shape[0], dtype=np.int32) * lang_i)
    all_lang_reps = np.concatenate(all_lang_reps, axis=0) # Concatenate across languages.
    all_lang_labels = np.concatenate(all_lang_labels, axis=0)
    lda = LinearDiscriminantAnalysis()
    lda.fit(all_lang_reps, all_lang_labels)
    # Shape: (n_dims, n_classes-1)
    lda_axes = lda.scalings_
    if not os.path.isdir(CACHED_VECTORS_DIR):
        os.mkdir(CACHED_VECTORS_DIR)
    np.save(os.path.join(CACHED_VECTORS_DIR, "lang_lda_layer{}.npy".format(layer)), lda_axes, allow_pickle=False)
    return lda_axes

# Get an axis direction from the inputs described in visualize_representations.py.
def get_direction(axis_code, layer):
    if isinstance(axis_code, np.ndarray):
        return axis_code # Axis is already computed.
    # Otherwise, assume axis_code is a string.
    axis_type, suffix = tuple(axis_code.split(":"))
    if axis_type == "langsvd":
        # SVD directions for a given language.
        # Format: langsvd:[lang]-[dim_i]
        lang, dim_i = tuple(suffix.split("-"))
        dim_i = int(dim_i)
        subspace_vh_file = os.path.join(SUBSPACE_CACHE_DIR, "{0}_layer{1}_vh.npy".format(lang, layer))
        if not os.path.isfile(subspace_vh_file):
            print("ERROR: subspace has not been computed yet, or is not cached in SUBSPACE_CACHE_DIR; "
                  "compute subspaces using eval_perplexity.py (see readme).")
            return None
        vh = np.load(subspace_vh_file)
        return vh[dim_i, :]
    elif axis_type == "lang":
        # Language LDA axes. Format: lang:[dim_i]
        lang_lda_file = os.path.join(CACHED_VECTORS_DIR, "lang_lda_layer{}.npy".format(layer))
        if os.path.isfile(lang_lda_file):
            lang_lda = np.load(lang_lda_file)
        else:
            print("Computing language LDA. Representations must be saved in RAW_REPS_DIR.")
            lang_lda = get_lang_lda(CONSIDERED_LANGS, layer)
        dim_i = int(suffix)
        return lang_lda[:, dim_i]
    elif axis_type == "position":
        # Position LDA axes. Format: position:[dim_i]
        position_lda_file = os.path.join(CACHED_VECTORS_DIR, "position_lda_layer{}.npy".format(layer))
        if os.path.isfile(position_lda_file):
            position_lda = np.load(position_lda_file)
        else:
            print("Computing position LDA. Must have run get_position_representations.py.")
            position_lda = get_position_lda(POSITIONS, layer)
        dim_i = int(suffix)
        return position_lda[:, dim_i]
    print("Unsupported axis type: {}".format(axis_type))
    return None
