"""
Constants and directories to store representations and data for visualizations.
"""

UD_POS_DICT_PATH = "multilingual-geometry/visualization/ud_pos_dict.txt"
RAW_REPS_DIR = "oscar_xlmr_reps"
TOKENIZED_SUBSETS_DIR = "oscar_xlmr_tokenized_subsets/for_visualizations"
POS_REPS_DIR = "oscar_xlmr_reps/pos"
POSITION_REPS_DIR = "oscar_xlmr_reps/position"
VISUALIZATION_OUTPUTS = "oscar_xlmr_visualizations"
CONSIDERED_LANGS = ["ar", "en", "es", "zh", "ru"]
ALL_POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
           "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
CACHED_VECTORS_DIR = "oscar_xlmr_cached_vectors"
SUBSPACE_CACHE_DIR = "oscar_xlmr_eval/subspace_cache"
POSITIONS = range(1, 512, 32)
