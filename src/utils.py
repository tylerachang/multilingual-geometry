"""
Utilities and constants for multilingual language model code.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Optional, List
from tqdm import tqdm
import numpy as np
import codecs
import os
import pickle
from scipy.linalg import svd
from transformers import AutoModelForMaskedLM
from collections import Counter

from src.xlmr_modeling import ModifiedRobertaEncoder

OSCAR_XLMR_LANGS = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'ca',
                    'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu',
                    'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'he', 'hi',
                    'hr', 'hu', 'hy', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk',
                    'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lo', 'lt', 'lv', 'mg',
                    'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no', 'or',
                    'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sa', 'sd', 'si', 'sk',
                    'sl', 'sq', 'sr', 'su', 'sv', 'sw', 'ta', 'te', 'th', 'tl',
                    'tr', 'ug', 'uk', 'ur', 'uz', 'vi', 'yi', 'zh']
XNLI_LANGS = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw',
              'th', 'tr', 'ur', 'vi', 'zh']


# Load one model, given the directory (or pre-trained model name) and config.
def load_single_model(single_model_dir, config, tokenizer, cache_dir=None):
    print("Loading from directory: {}".format(single_model_dir))
    model = AutoModelForMaskedLM.from_pretrained(
        single_model_dir,
        config=config,
        cache_dir=cache_dir
    )
    model.resize_token_embeddings(len(tokenizer))
    # Load onto GPU.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


# Convert a list of integer token_id lists into input_ids and attention_mask.
# Inputs should already include CLS and SEP tokens. Because this function
# does not know the maximum sequence length, examples should already be truncated.
# All sequences will be padded to the length of the longest example, so this
# should be called per batch.
# Labels are set to None.
def prepare_tokenized_examples(tokenized_examples, tokenizer):
    # Convert into a tensor.
    tensor_examples = [torch.tensor(e, dtype=torch.long) for e in tokenized_examples]
    input_ids = pad_sequence(tensor_examples, batch_first=True,
                             padding_value=tokenizer.pad_token_id)
    attention_mask = input_ids != tokenizer.pad_token_id
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": None}
    if torch.cuda.is_available():
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
    return inputs


# Convert a list of integer token_id lists into input_ids, attention_mask, and labels.
# 15% of tokens are replaced with [MASK] (80%), a random token (10%), or the original token (10%).
# Labels are -100 (no loss computed) for the remaining 85% of tokens.
def prepare_tokenized_examples_masked(tokenized_examples, tokenizer, mlm_probability=0.15):
    inputs = prepare_tokenized_examples(tokenized_examples, tokenizer)
    if tokenizer.mask_token is None:
        print("ERROR: this tokenizer has no mask token.")
        return inputs
    labels = inputs["input_ids"].clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability
    # mlm_probability which defaults to 0.15 in Bert/RoBERTa).
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    padding_mask = labels.eq(tokenizer.pad_token_id).cpu()
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens.
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK]).
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs["input_ids"][indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word.
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    if torch.cuda.is_available():
        random_words = random_words.cuda()
    inputs["input_ids"][indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged.
    # Add the labels to the inputs.
    inputs["labels"] = labels
    if torch.cuda.is_available():
        inputs["labels"] = inputs["labels"].cuda()
    return inputs


# Output the hidden states given examples (lists of token_ids).
# Outputs tensor of shape n_tokens x hidden_size.
# Handles batching and example tensorizing.
def get_hidden_states(model, examples, batch_size, tokenizer, layer):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        batches.append(examples[i:])
    # Run evaluation.
    model.eval()
    with torch.no_grad():
        eval_hidden_states = []
        for batch_i in tqdm(range(len(batches))):
            inputs = prepare_tokenized_examples(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            output_hidden_states=True, return_dict=True)
            hidden_states = outputs["hidden_states"][layer].detach()
            del outputs # Delete before the next batch runs.
            hidden_size = hidden_states.shape[-1]
            hidden_states = hidden_states.reshape(-1, hidden_size)
            # Remove pad tokens.
            # Shape: n_tokens x hidden_size
            hidden_states = hidden_states[inputs["attention_mask"].flatten(), :]
            eval_hidden_states.append(hidden_states.detach().cpu()) # Send to CPU so not all need to be held on GPU.
        all_eval_hidden_states = np.zeros((0, hidden_size))
        while len(eval_hidden_states) > 0:
            states = eval_hidden_states.pop(0)
            all_eval_hidden_states = np.concatenate((all_eval_hidden_states, states), axis=0)
    return all_eval_hidden_states


# Note: examples should already be truncated.
# Outputs (average) loss. Handles batching and example tensorizing.
# Each additional metric can be:
# an array of the valid token ids in a language to count the proportion of
# output tokens in that language.
def run_eval(model, tokenizer, examples, batch_size, additional_metrics=[]):
    # Create batches.
    batches = []
    i = 0
    while i+batch_size <= len(examples):
        batches.append(examples[i:i+batch_size])
        i += batch_size
    if len(examples) % batch_size != 0:
        # Note: the final loss weights each batch equally, so ideally each batch
        # has the same size (i.e. batch_size divides num_examples).
        batches.append(examples[i:])
    # Run evaluation.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    total_loss = 0.0
    additional_counts = Counter()
    # To count the proportion of tokens in each language.
    lang_tokens = [] # List of ndarrays.
    for metric in additional_metrics:
        if isinstance(metric, np.ndarray):
            lang_tokens.append(torch.tensor(metric))
        else:
            print("WARNING: expected ndarray.")
    model.eval()
    with torch.no_grad():
        for batch_i in tqdm(range(len(batches))):
            inputs = prepare_tokenized_examples_masked(batches[batch_i], tokenizer)
            # Run model.
            outputs = model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            labels=inputs["labels"],
                            output_hidden_states=False, return_dict=True)
            total_loss += outputs["loss"].detach().mean() # Mean over GPUs.
            masked = inputs["labels"] >= 0
            logits = outputs["logits"].detach()[masked, :]
            preds = torch.argmax(logits, axis=-1)
            additional_counts["total_preds"] += torch.sum(masked)
            additional_counts["correct"] += torch.sum(inputs["labels"][masked] == preds)
            for lang_i, valid_tokens in enumerate(lang_tokens):
                additional_counts["in_lang{}".format(lang_i)] += torch.sum(torch.isin(preds.cpu(), valid_tokens))
            # To print the predictions, to check that they're not complete gibberish.
            # for token_id in preds:
            #     print(tokenizer.decode(token_id), end="\t")
            # print("")
            del logits
            del outputs
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    loss = total_loss / len(batches)
    accuracy = float(additional_counts["correct"]) / additional_counts["total_preds"]
    proportion_in_langs = []
    for lang_i, valid_tokens in enumerate(lang_tokens):
        proportion_in_lang = float(additional_counts["in_lang{}".format(lang_i)]) / additional_counts["total_preds"]
        proportion_in_langs.append(proportion_in_lang.item())
    return {"loss": loss, "accuracy": accuracy, "in_langs": proportion_in_langs}


@dataclass
class ProjectionEvalArguments:
    projection_layers: List[int] = field(
        default_factory=list, metadata={"help": "Which layers to project. Defaults to no projections."}
    )
    # Tokenized example subsets to use when computing subspaces and running evaluations.
    # The first subset will be used for subspace computation.
    # The last subset will be used for evaluation.
    pickled_subsets_dir: str = field(
        default=None, metadata={"help": "Path to the pickled subsets generated by subset_examples.py"}
    )
    output_dir: str = field(default=None, metadata={"help": "Output directory containing results file and subspace cache."})
    output_filename: str = field(
        default="eval_results.tsv", metadata={"help": "Output tsv of perplexity results."}
    )
    subspace_cache_name: str = field(default="subspace_cache")

    # A directory containing the token vocabularies for each language. Optional.
    # See eval_perplexity.py script for details.
    lang_tokens_dir: str = field(default="")

    # Arguments about the model to use.
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated"
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."})
    per_device_eval_batch_size: Optional[int] = field(default=8)

    # Arguments overridden by the EVAL_TUPLES in eval_perplexity.py.
    eval_language: str = field(default=None, metadata={"help": "Evaluation language."})
    projection_lang: str = field(default=None, metadata={"help": "Language for projection."})
    mean_a: str = field(default=None, metadata={"help": "Vector subtracted before projection."})
    mean_b: str = field(default=None, metadata={"help": "Vector added after projection."})
    variance_accounted: float = field(default=None, metadata={"help": "The proportion of variance each subspace should account for."})


# Note: examples should already be truncated (but need not be padded).
# Computes and saves a single subspace. Saves the singular values s (shape: dim_size),
# corresponding orthonormal basis vh (shape: dim_size x dim_size), and representation
# mean (shape: dim_size, subtracted before SVD). The rows of vh correspond to basis
# vectors. If the number of token representations is less than dim_size (which should
# not happen in practice), then vh has shape (n_tokens, dim_size).
# Optionally saves the raw token representations, by setting save_representations to
# "true" or "only" (to save the representations without computing any subspace).
def get_subspace(model, tokenizer, examples, output_dir, lang, subspace_name,
                 batch_size, layer, save_representations="false"):
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
        # get_subspace doesn't automatically send to DataParallel.
        model = torch.nn.DataParallel(model)
    # Get hidden states with shape: num_tokens x hidden_size.
    print("Running model...")
    hidden_states = get_hidden_states(model, examples, batch_size, tokenizer, layer)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    rep_mean = np.mean(hidden_states, axis=0)
    if save_representations in ["true", "only"]:
        rep_outpath = os.path.join(output_dir, "{0}_reps.npy".format(subspace_name))
        np.save(rep_outpath, hidden_states, allow_pickle=False)
        print("Saved token representations.")
        if save_representations == "only":
            print("Not computing subspace.")
            return None, None, None
    print("Running SVD on {} token representations...".format(hidden_states.shape[0]))
    hidden_states = hidden_states - rep_mean.reshape(1, -1)
    # Assume num_tokens > dim_size.
    # Shapes: (num_tokens x num_tokens), (dim_size), (dim_size x dim_size).
    # Note: rows of vh form the orthonormal basis, corresponding to the singular
    # values in s.
    # Note: do not output full matrices because u would have shape (num_tokens x num_tokens)
    # instead of num_tokens x dim_size (much smaller in memory).
    u, s, vh = svd(hidden_states, full_matrices=False, compute_uv=True, overwrite_a=True)
    s_outpath = os.path.join(output_dir, "{0}_s.npy".format(subspace_name))
    vh_outpath = os.path.join(output_dir, "{0}_vh.npy".format(subspace_name))
    mean_outpath = os.path.join(output_dir, "{0}_mean.npy".format(subspace_name))
    np.save(s_outpath, s, allow_pickle=False)
    np.save(vh_outpath, vh, allow_pickle=False)
    np.save(mean_outpath, rep_mean, allow_pickle=False)
    print("Saved computed subspace.")
    return s, vh, rep_mean


# Convert the XLM-R model to a model that projects representations from one
# language into another in given layers. Assumes an input model type of XLMRobertaForMaskedLM.
# Note that simply shifting representations according to language means (no projection)
# can be done by setting projection_lang to None.
# Previous projections and means will be overridden in the model.
# If not already computed and cached in subspace_dir, the subspaces and means are
# computed from the example subsets in pickled_subsets_dir (see scripts/subset_examples.py).
def project_xlmr_model(model, tokenizer, projection_layers, projection_lang, mean_a_lang, mean_b_lang,
                       variance_accounted, subspace_dir, pickled_subsets_dir="",
                       max_seq_length=512, total_batch_size=8, save_representations="false"):
    print("Projecting XLM-R model into {0} (means {1} to {2}, variance {3})...".format(projection_lang, mean_a_lang, mean_b_lang, variance_accounted))
    # Helper function to get a subspace for representations in a language.
    # Subtracts the representation mean before computing the subspace with SVD.
    # subspace_name is usually [subspace_language]_layer[layer]
    def get_subspace_for_lang(lang, layer, subspace_name=""):
        nonlocal model
        nonlocal tokenizer
        nonlocal pickled_subsets_dir
        nonlocal subspace_dir
        nonlocal max_seq_length
        nonlocal total_batch_size
        nonlocal save_representations
        # If the subspace has already been saved, load it.
        s_outpath = os.path.join(subspace_dir, "{0}_s.npy".format(subspace_name))
        vh_outpath = os.path.join(subspace_dir, "{0}_vh.npy".format(subspace_name))
        mean_outpath = os.path.join(subspace_dir, "{0}_mean.npy".format(subspace_name))
        if os.path.isfile(s_outpath) and os.path.isfile(vh_outpath) and os.path.isfile(mean_outpath):
            print("Loading saved subspace.")
            s = np.load(s_outpath, allow_pickle=False)
            vh = np.load(vh_outpath, allow_pickle=False)
            m = np.load(mean_outpath, allow_pickle=False)
            return s.reshape(-1), vh, m.reshape(-1)
        # Otherwise, compute the subspace.
        pickled_subsets_file = os.path.join(pickled_subsets_dir, "{}.pickle".format(lang))
        with open(pickled_subsets_file, 'rb') as handle:
            examples_subsets = pickle.load(handle)
        examples = examples_subsets[0] # Default to use the first subset of examples when computing subspaces.
        for example_i in range(len(examples)): # Truncate examples.
            if len(examples[example_i]) > max_seq_length:
                examples[example_i] = examples[example_i][:max_seq_length]
        s, vh, subspace_m = get_subspace(model, tokenizer, examples, subspace_dir, lang, subspace_name,
                                       total_batch_size, layer, save_representations=save_representations)
        return s, vh, subspace_m
    # To project representations:
    # Subtract mean_a, project into projection_language, add mean_b.
    projections = []
    means_a = []
    means_b = []
    subspace_dims = []
    num_layers = model.roberta.config.num_hidden_layers
    dim_size = model.roberta.config.hidden_size
    for layer_i in range(num_layers+1): # Embeddings plus each hidden layer.
        if layer_i in projection_layers:
            print("Project layer {}".format(layer_i))
            # Get mean_a.
            # Because the representations are computed anyways, compute the
            # language subspaces when the means are computed.
            # Loads the cached means and subspaces if possible.
            mean_a = np.zeros(dim_size)
            if mean_a_lang is not None and len(mean_a_lang) != 0:
                subspace_name = "{0}_layer{1}".format(mean_a_lang, layer_i)
                _, _, mean_a = get_subspace_for_lang(mean_a_lang, layer_i, subspace_name=subspace_name)
            # Get mean_b.
            mean_b = np.zeros(dim_size)
            if mean_b_lang is not None and len(mean_b_lang) != 0:
                subspace_name = "{0}_layer{1}".format(mean_b_lang, layer_i)
                _, _, mean_b = get_subspace_for_lang(mean_b_lang, layer_i, subspace_name=subspace_name)
            means_a.append(mean_a)
            means_b.append(mean_b)
            # Get the language subspace projection.
            if projection_lang is None or len(projection_lang) == 0 or variance_accounted == 1.0:
                # No projection.
                subspace_dim = dim_size
                projection_matrix = np.identity(dim_size)
            else:
                subspace_name = "{0}_layer{1}".format(projection_lang, layer_i)
                s, vh, subspace_m = get_subspace_for_lang(projection_lang, layer_i, subspace_name=subspace_name)
                v = np.transpose(vh) # Columns of V form the desired orthonormal basis.
                # Determine how many dimensions to use.
                subspace_dim = 0
                s_squared = np.square(s)
                total_variance = np.sum(s_squared) # Proportional to total variance.
                cutoff_variance = variance_accounted * total_variance
                curr_variance = 0.0
                for i in range(s.shape[-1]):
                  curr_variance += s_squared[i]
                  if curr_variance >= cutoff_variance:
                    subspace_dim = i+1
                    break
                # Projection matrix: convert into basis (excluding some dimensions), then
                # convert back into standard basis.
                v = v[:, :subspace_dim]
                projection_matrix = np.matmul(v, np.transpose(v))
            projections.append(projection_matrix)
            subspace_dims.append(subspace_dim)
        else: # Do not project this layer.
            projections.append(None)
            means_a.append(None)
            means_b.append(None)
            subspace_dims.append(dim_size)
    # Project the encoder.
    if not isinstance(model.roberta.encoder, ModifiedRobertaEncoder):
        original_encoder = model.roberta.encoder
        model.roberta.encoder = ModifiedRobertaEncoder(original_encoder=original_encoder)
    model.roberta.encoder.set_transformations(projections=projections, means_a=means_a, means_b=means_b)
    subspace_dims = ",".join([str(dim) for dim in subspace_dims])
    print("Subspace dims: {}".format(subspace_dims))
    return model
