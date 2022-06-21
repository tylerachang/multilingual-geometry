"""
Evaluates the perplexity of a model for different languages, optionally projected
into language subspaces. Computes and saves language subspaces in the process.
Sample usage:

python3 eval_perplexity.py --model_name_or_path="xlm-roberta-base" \
--per_device_eval_batch_size=8 --max_seq_length 512 --cache_dir="../hf_cache" \
--pickled_subsets_dir="../oscar_xlmr_tokenized_subsets/512_examples" \
--output_dir="../oscar_xlmr_eval" \
--output_filename="perplexity_eval_results.tsv" \
--subspace_cache_name="subspace_cache" \
--projection_layers=8 \
--lang_tokens_dir="../oscar_xlmr_token_counts"

"""

import os
import argparse
import math
import numpy as np
import torch
import codecs
import pickle
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    HfArgumentParser,
    TrainingArguments,
)

from src.utils import run_eval, project_xlmr_model, ProjectionEvalArguments
from src.utils import OSCAR_XLMR_LANGS, XNLI_LANGS
from src.xlmr_modeling import ModifiedRobertaEncoder

# Choose evaluations to run.
# Each tuple should be eval_lang, proj_lang, mean_a_lang, mean_b_lang, variance_accounted.
# When evaluating on eval, the projection will apply the formula:
# V_{proj}V_{proj}^T(x - \mu_{a}) + \mu_{b}
# where the dimensionality of the proj subspace is based on variance_accounted.
EVAL_TUPLES = []
CONSIDERED_LANGS = XNLI_LANGS # Use OSCAR_XLMR_LANGS for additional languages.

# Perplexities without projections.
for lang in CONSIDERED_LANGS:
    EVAL_TUPLES.append((lang, None, None, None, None))
# Default projections (a subset of the projections below).
for lang in CONSIDERED_LANGS:
    EVAL_TUPLES.append((lang, lang, lang, lang, 0.90))

# Projections not maintaining eval language A representation means.
# Projects onto the closest point on subspace B.
# for eval_lang in CONSIDERED_LANGS:
#     for proj_lang in CONSIDERED_LANGS:
#         EVAL_TUPLES.append((eval_lang, proj_lang, proj_lang, proj_lang, 0.90))

# Projections roughly maintaining language A representation means.
# Projects onto the closest point on subspace B shifted to contain mean A.
# for eval_lang in CONSIDERED_LANGS:
#     for proj_lang in CONSIDERED_LANGS:
#         EVAL_TUPLES.append((eval_lang, proj_lang, eval_lang, eval_lang, 0.90))

# Only shifting language means from A to B (no projection).
# for eval_lang in CONSIDERED_LANGS:
#     for proj_lang in CONSIDERED_LANGS:
#         EVAL_TUPLES.append((eval_lang, None, eval_lang, proj_lang, None))

# Projections shifting the means from A to B then projecting onto B.
# Projects the mean-shifted point (shifted A to B) onto the closest point in subspace B.
# for eval_lang in CONSIDERED_LANGS:
#     for proj_lang in CONSIDERED_LANGS:
#         EVAL_TUPLES.append((eval_lang, proj_lang, eval_lang, proj_lang, 0.90))


def main():
    parser = HfArgumentParser(ProjectionEvalArguments)
    eval_args = parser.parse_args_into_dataclasses()[0]

    # Load model.
    config = AutoConfig.from_pretrained(
        eval_args.model_name_or_path,
        cache_dir=eval_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        eval_args.model_name_or_path,
        do_lower_case=eval_args.do_lower_case,
        cache_dir=eval_args.cache_dir,
    )
    model_class = AutoModelForMaskedLM
    model = model_class.from_pretrained(
        eval_args.model_name_or_path,
        config=config,
        cache_dir=eval_args.cache_dir,
    )
    original_encoder = model.roberta.encoder
    # Use modified encoder to allow projections.
    # Can now use the function model.roberta.encoder.set_transformations() or
    # project_xlmr_model(model).
    model.roberta.encoder = ModifiedRobertaEncoder(original_encoder=original_encoder)

    # Prepare to write outputs.
    if not os.path.isdir(eval_args.output_dir):
        os.mkdir(eval_args.output_dir)
    outpath = os.path.join(eval_args.output_dir, eval_args.output_filename)
    if os.path.isfile(outpath):
        outfile = codecs.open(outpath, 'ab', encoding='utf-8')
    else:
        outfile = codecs.open(outpath, 'wb', encoding='utf-8')
        outfile.write("EvalLang\tProjectionLang\tMeanA\tMeanB\t")
        outfile.write("VarianceAccounted\tProjectionLayers\tPerplexity\tAccuracy\tInEvalTokens\tLangB\tInLangBTokens\tCommonTokensProportion\n")
    def write_row(eval_perplexity, eval_accuracy, in_eval_proportion=-1.0, langb=None, in_langb_proportion=-1.0, common_tokens_proportion=-1.0):
        nonlocal outfile
        nonlocal eval_args
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\n".format(eval_args.eval_language,
                eval_args.projection_lang, eval_args.mean_a, eval_args.mean_b, eval_args.variance_accounted,
                ",".join([str(layer) for layer in eval_args.projection_layers]),
                eval_perplexity, eval_accuracy, in_eval_proportion, langb, in_langb_proportion, common_tokens_proportion))

    # Run perplexity evaluations.
    for eval_i, eval_tuple in enumerate(EVAL_TUPLES):
        eval_lang, proj_lang, mean_a_lang, mean_b_lang, variance_accounted = eval_tuple
        print("Running {0} projected into {1} ({2}/{3}).".format(eval_lang, proj_lang,
                                                                 eval_i+1, len(EVAL_TUPLES)))
        # Load examples.
        eval_args.eval_language = eval_lang # For correct logging and outputs.
        eval_args.projection_lang = proj_lang
        eval_args.mean_a = mean_a_lang
        eval_args.mean_b = mean_b_lang
        eval_args.variance_accounted = variance_accounted
        pickled_subsets_file = os.path.join(eval_args.pickled_subsets_dir, "{}.pickle".format(eval_lang))
        if os.path.isfile(pickled_subsets_file):
            print("Loading example subsets from file: {}".format(pickled_subsets_file))
            with open(pickled_subsets_file, 'rb') as handle:
                examples_subsets = pickle.load(handle)
            examples = examples_subsets[-1] # By default, evaluate on the last subset.
            for example_i in range(len(examples)): # Truncate examples.
                if len(examples[example_i]) > eval_args.max_seq_length:
                    examples[example_i] = examples[example_i][:eval_args.max_seq_length]
        else:
            print("Missing pickled examples file ({}). Skipping.".format(eval_lang))
            continue

        # Project the model.
        # Note that subspaces are computed from examples in the first subset in the pickled subsets.
        # Evaluation is performed on the last subset.
        # Reset the model from any previous projections.
        n_layers = model.roberta.config.num_hidden_layers + 1 # Including embedding layer.
        model.roberta.encoder.set_transformations(projections=[None] * n_layers)
        # Computes subspaces if not already computed.
        total_batch_size = eval_args.per_device_eval_batch_size
        if torch.cuda.is_available():
            total_batch_size = eval_args.per_device_eval_batch_size * torch.cuda.device_count()
        subspace_cache_dir = os.path.join(eval_args.output_dir, eval_args.subspace_cache_name)
        if not os.path.isdir(subspace_cache_dir):
            os.mkdir(subspace_cache_dir)
        # This will automatically compute the subspaces if not already cached.
        model = project_xlmr_model(model, tokenizer, eval_args.projection_layers,
                                   eval_args.projection_lang, eval_args.mean_a, eval_args.mean_b,
                                   eval_args.variance_accounted, subspace_cache_dir,
                                   eval_args.pickled_subsets_dir,
                                   eval_args.max_seq_length, total_batch_size)

        # Optionally, get the proportion of predicted tokens in each language during evaluation.
        # In lang_tokens_dir, must have saved:
        # (1) a binary npy array (common_tokens.npy, shape: vocab_size) with True at the
        # indices of any common tokens across languages. These tokens will not be included
        # in the in-language token counts.
        # (2) an array for each language ([lang].npy, shape: vocab_size) of per-1000
        # token frequencies.
        additional_metrics = []
        if eval_args.lang_tokens_dir != "":
            # Tokens to exclude:
            common_tokens = np.load(os.path.join(eval_args.lang_tokens_dir, "common_tokens.npy"), allow_pickle=False)
            # Consider a token in the language if at least 0.001 occurrences every 1000 tokens (one every 1M tokens).
            cutoff = 0.001
            langs_to_evaluate_proportions = None
            if (proj_lang is None and mean_a_lang is None and mean_b_lang is None) or proj_lang == eval_lang:
                # If no projection, or if projecting into the evaluation language, consider all other possible languages.
                langs_to_evaluate_proportions = CONSIDERED_LANGS
            else:
                # Otherwise, to save time, only consider the evaluation and projection languages.
                eval_lang_b = mean_b_lang if proj_lang is None else proj_lang
                langs_to_evaluate_proportions = [eval_lang, eval_lang_b]
            for lang in langs_to_evaluate_proportions:
                # Per-1000 token frequencies.
                lang_tokens = np.load(os.path.join(eval_args.lang_tokens_dir, "{}.npy".format(lang)), allow_pickle=False)
                lang_tokens = lang_tokens >= cutoff
                lang_tokens = np.logical_and(lang_tokens, np.logical_not(common_tokens))
                lang_tokens = np.nonzero(lang_tokens)[0] # Get the indices of valid tokens.
                additional_metrics.append(lang_tokens)
            # Finally, evaluate for common tokens.
            additional_metrics.append(np.nonzero(common_tokens)[0])

        # Run eval.
        eval_results = run_eval(model, tokenizer, examples, total_batch_size, additional_metrics=additional_metrics)
        perplexity = torch.exp(eval_results["loss"])
        print("Perplexity: {}".format(perplexity))
        print("Accuracy: {}".format(eval_results["accuracy"]))

        if eval_args.lang_tokens_dir == "":
            # Not computing in-language token proportions.
            write_row(perplexity, eval_results["accuracy"])
            continue
        # Compute in-language token proportions.
        if (proj_lang is None and mean_a_lang is None and mean_b_lang is None) or proj_lang == eval_lang:
            # If no projection, or if projecting into the evaluation language:
            # Write row for every other language.
            # Only the langb proportion will change.
            eval_lang_idx = CONSIDERED_LANGS.index(eval_lang)
            in_eval_proportion = eval_results["in_langs"][eval_lang_idx]
            common_tokens_proportion = eval_results["in_langs"][-1]
            # For all possible projection languages.
            for lang_i, langb in enumerate(CONSIDERED_LANGS):
                in_langb_proportion = eval_results["in_langs"][lang_i]
                write_row(perplexity, eval_results["accuracy"], in_eval_proportion, langb, in_langb_proportion, common_tokens_proportion)
        else:
            # To save time, only consider in-language proportions for the eval and projection language.
            in_eval_proportion = eval_results["in_langs"][0]
            in_proj_proportion = eval_results["in_langs"][1]
            common_tokens_proportion = eval_results["in_langs"][2]
            print("In eval/proj language: {0}, {1}".format(in_eval_proportion, in_proj_proportion))
            print("Common tokens proportion: {0}".format(common_tokens_proportion))
            write_row(perplexity, eval_results["accuracy"], in_eval_proportion, langb=eval_lang_b, in_langb_proportion=in_proj_proportion, common_tokens_proportion=common_tokens_proportion)
    outfile.close()
    print("Done.")


if __name__ == "__main__":
    main()
