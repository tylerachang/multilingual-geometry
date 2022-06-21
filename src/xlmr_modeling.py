"""
Modified version of the XLM-R encoder from Huggingface.
"""

import torch
import torch.nn as nn

from transformers.models.roberta.modeling_roberta import RobertaLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


# Modified RoBERTa encoder.
# Copied from transformers.models.roberta.modeling_roberta.RobertaEncoder with
# optional transformations after each layer. Initialize as usual, then class
# variables can be modified to add projections, using the set_transformations()
# function.
class ModifiedRobertaEncoder(torch.nn.Module):
    # Either config or original_encoder should be set.
    def __init__(self, config=None, original_encoder=None):
        super().__init__()
        if original_encoder is None:
            self.config = config
            self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
            self.gradient_checkpointing = False
        else: # Initialize from original encoder.
            self.config = original_encoder.config
            self.layer = original_encoder.layer
            self.gradient_checkpointing = original_encoder.gradient_checkpointing

        # Optionally apply batch norm (independently for each time step, as in Zhao et al., 2021).
        # Note that this is a stronger condition than batch norm across all time steps;
        # instead of the entire set of representation means being zero, the means are zero
        # for every time step.
        self.use_batch_norm_mean = [False] * (self.config.num_hidden_layers+1)
        # Optionally apply normalization to the variances.
        self.use_batch_norm_stdev = [False] * (self.config.num_hidden_layers+1)

        self.projections = [None] * (self.config.num_hidden_layers+1)
        self.projection_shifts = [None] * (self.config.num_hidden_layers+1)
        # Fixed shifts will occur after any projections.
        # Shift formula: r + shift
        self.fixed_shifts = [None] * (self.config.num_hidden_layers+1)

    # Call this method to set transformations within the model.
    # Anything set to None will not modify the existing field in the model.
    #
    # Optionally project into a subspace. Subtracts mean_a before projection and adds
    # mean_b after projection. The goal of mean_a is to roughly mean-center the representations.
    # Requires projection matrices (a list of dim_size x dim_size tensors, one for the embeddings
    # plus each hidden layer, potentially including None values for layers with no projection), the
    # mean_a vectors (a list of dim_size tensors), and the mean_b vectors.
    # Projection formula: W(v-m_a)+m_b = W v - W m_a + m_b
    def set_transformations(self, projections=None, means_a=None, means_b=None, fixed_shifts=None,
                            use_batch_norm_mean=None, use_batch_norm_stdev=None):
        dim_size = self.config.hidden_size
        for layer_i in range(self.config.num_hidden_layers+1): # Embedding plus each hidden layer.
            # Update batch norming if provided.
            if use_batch_norm_mean is not None:
                self.use_batch_norm_mean[layer_i] = use_batch_norm_mean[layer_i]
            if use_batch_norm_stdev is not None:
                self.use_batch_norm_stdev[layer_i] = use_batch_norm_stdev[layer_i]
            # Add any projections from A to B.
            if projections is None:
                pass # Leave projections unchanged.
            elif projections[layer_i] is None:
                # Setting a projection to None will also remove the means.
                self.projection_shifts[layer_i] = None
                self.projections[layer_i] = None
            else:
                projection = torch.tensor(projections[layer_i]).float()
                mean_a = torch.tensor(means_a[layer_i]).float()
                mean_b = torch.tensor(means_b[layer_i]).float()
                projection_shift = mean_b - torch.matmul(projection, mean_a)
                projection_shift = projection_shift.reshape(1, 1, dim_size)
                self.projection_shifts[layer_i] = projection_shift
                self.projections[layer_i] = projection
            # Add any fixed shifts.
            if fixed_shifts is None:
                pass # Leave unchanged.
            elif fixed_shifts[layer_i] is None:
                self.fixed_shifts[layer_i] = None
            else:
                fixed_shift = torch.tensor(fixed_shifts[layer_i]).float()
                fixed_shift = fixed_shift.reshape(1, 1, dim_size)
                self.fixed_shifts[layer_i] = fixed_shift

    def transform_embeddings(self, inputs, layer):
        # Layer in [0, num_hidden_layers].
        # Input shape (batch_size, seq_len, dim_size).
        # Apply batch norm first.
        if self.use_batch_norm_mean[layer]:
            mean = inputs.mean(0, keepdim=True)
            inputs = inputs - mean
        if self.use_batch_norm_stdev[layer]:
            var = inputs.var(0, unbiased=False, keepdim=True)
            inputs = inputs / torch.sqrt(var + 1e-9)
        if self.projections[layer] is not None:
            projection = self.projections[layer].to(inputs.device)
            projection_shift = self.projection_shifts[layer].to(inputs.device)
            inputs = torch.einsum('ij,bsj->bsi', projection, inputs) + projection_shift
        if self.fixed_shifts[layer] is not None:
            fixed_shift = self.fixed_shifts[layer].to(inputs.device)
            inputs = inputs + fixed_shift
        return inputs

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        # Transform layer 0 (embeddings).
        hidden_states = self.transform_embeddings(hidden_states, 0)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # Transform layer i.
            hidden_states = self.transform_embeddings(hidden_states, i+1)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
