# multilingual-language-models
Code for the paper [The Geometry of Multilingual Language Model Representations](https://arxiv.org/abs/2205.10964) (2022).
Includes code for identifying affine language subspaces in multilingual language models, evaluating perplexities when representations are projected onto language subspaces, computing subspace distances, and visualizing representations projected onto various axes and subspaces.

For more detailed usage, see comments in the individual Python scripts.
Visualizations require only steps 0 and 3 below.
Run on Python 3.7.9 and Pytorch 1.7.1 (see requirements.txt).
Sample usage is provided online on Google Colab [here](https://colab.research.google.com/drive/1DxEi6_gg3WLaUWwz61-JQ7iwykyN1hud?usp=sharing).

## 0. Get tokenized sequences from the OSCAR corpus.
This step is necessary for any of the sections below.
First, pull raw text data from the OSCAR corpus.
This outputs a text file for each language.
To access the data, you may have to create a Hugging Face account, activate the dataset [here](https://huggingface.co/datasets/oscar-corpus/OSCAR-2109), install the transformers library, and log in with the command "huggingface-cli login" first.
<pre>
python3 scripts/get_text_data.py --dataset="oscar" --output_dir="../oscar_data" \
--max_examples=128000000
</pre>
Tokenize the examples.
This outputs a text file for each language, where each line is a space-separated list of token ids.
<pre>
python3 scripts/tokenize_examples.py --tokenizer="xlm-roberta-base" \
--input_dir="../oscar_data" --output_dir="../oscar_xlmr_tokenized" \
--max_examples=-1 --max_segments=-1 --max_seq_len=512
</pre>

## 1. Compute language subspaces and evaluate projected language modeling perplexities.
For each language, create two subsets of 512 examples (one for subspace computation, and one for language modeling evaluation).
This outputs a pickle file for each language.
<pre>
python3 scripts/subset_examples.py --input_dir="../oscar_xlmr_tokenized" \
--output_dir="../oscar_xlmr_tokenized_subsets/512_examples" \
--max_examples=512 --num_subsets=2
</pre>
Run perplexity evaluations when the model is projected into each language subspace.
This automatically computes and caches the language subspaces.
By default (along with unprojected perplexities), it runs evaluation for each language A projecting into the language A subspace using the formula V_{A}V_{A}^T(x - \mu_{A}) + \mu_{A}.
Other projections can be selected manually by updating EVAL_TUPLES in eval_perplexity.py.
<pre>
python3 eval_perplexity.py --model_name_or_path="xlm-roberta-base" \
--per_device_eval_batch_size=8 --max_seq_length 512 --cache_dir="../hf_cache" \
--pickled_subsets_dir="../oscar_xlmr_tokenized_subsets/512_examples" \
--output_dir="../oscar_xlmr_eval" \
--output_filename="perplexity_eval_results.tsv" \
--subspace_cache_name="subspace_cache" \
--projection_layers=8
</pre>

## 2. Compute distances between subspaces.
Step 1 saves the cached language subspaces in the output directory.
We can compute raw distances between the (mean-centered) subspaces using the metric from Bonnabel and Sepulchre (2009).
<pre>
python3 compute_distances.py \
--subspace_cache="../oscar_xlmr_eval/subspace_cache" \
--output_dir="../oscar_xlmr_eval/subspace_distances" \
--total_dims=768 --layers 8
</pre>

## 3. Visualize representations projected onto different axes.
This does not require running steps 1 and 2 above.
Code to be added.

## Citation.
<pre>
@article{chang-etal-2022-geometry,
  title={The Geometry of Multilingual Language Model Representations},
  author={Tyler Chang and Zhuowen Tu and Benjamin Bergen},
  journal={arXiv preprint},
  year={2022},
}
</pre>