#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains custom `TransformerEncoder` class
that accepts different mask for each layer.
"""

import torch
import torch.nn as nn
import copy
from torch import Tensor
from typing import Optional
from models.talesumm.custom_transformer import TransformerEncoderLayer

def _get_activation(activation: str) -> nn.Module:
    r"""
    Returns the activation function given the name.
    ----------
    Args:
        - activation: activation function name.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "celu":
        return nn.CELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "softshrink":
        return nn.Softshrink()
    elif activation == "mish":
        return nn.Mish()
    else:
        raise ValueError(f"Enter correct activation function name. Got {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerEncoder(nn.Module):
    r"""CustomTransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``False`` (disabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, 
                 encoder_layer, 
                 num_layers, 
                 norm=None,
                 per_layer_src_mask:bool=False,
                 enable_nested_tensor=False):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.per_layer_src_mask = per_layer_src_mask
        self.enable_nested_tensor = enable_nested_tensor

    def forward(self,
                src: Tensor,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None
               ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            per_layer_src_mask: if True, different mask will be applied to each layer. Default: `False`.
            mask: the mask for the src sequence (optional). Shape: [S x S] or [N.M x S x S].
                If `per_layer_src_mask` is `True`, shape should be [L x S x S] or [L x N.M x S x S],
                where `L` is the number of layers, `N` is the batch size, `M` is the number of heads,
                and `S` is the sequence length.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        if self.per_layer_src_mask:
            assert len(self.layers) == mask.size(0), "Number of layers must match mask size"
            
        if isinstance(first_layer, TransformerEncoderLayer):
            if (not first_layer.norm_first and not first_layer.training and
                    first_layer.self_attn.batch_first and
                    first_layer.self_attn._qkv_same_embed_dim and first_layer.activation_relu_or_gelu and
                    first_layer.norm1.eps == first_layer.norm2.eps and
                    src.dim() == 3 and self.enable_nested_tensor):
                if src_key_padding_mask is not None and not output.is_nested and mask is None:
                    tensor_args = (
                        src,
                        first_layer.self_attn.in_proj_weight,
                        first_layer.self_attn.in_proj_bias,
                        first_layer.self_attn.out_proj.weight,
                        first_layer.self_attn.out_proj.bias,
                        first_layer.norm1.weight,
                        first_layer.norm1.bias,
                        first_layer.norm2.weight,
                        first_layer.norm2.bias,
                        first_layer.linear1.weight,
                        first_layer.linear1.bias,
                        first_layer.linear2.weight,
                        first_layer.linear2.bias,
                    )
                    if not self.per_layer_src_mask and not torch.overrides.has_torch_function(tensor_args):
                        if not torch.is_grad_enabled() or all([not x.requires_grad for x in tensor_args]):
                            if output.is_cuda or 'cpu' in str(output.device):
                                convert_to_nested = True
                                output = torch._nested_tensor_from_mask(
                                    output, src_key_padding_mask.logical_not())

        for i, mod in enumerate(self.layers):
            if convert_to_nested:
                output = mod(output, src_mask=mask)
            else:
                if self.per_layer_src_mask:
                    output = mod(output, src_mask=mask[i],
                                 src_key_padding_mask=src_key_padding_mask)
                else:
                    output = mod(output, src_mask=mask,
                                 src_key_padding_mask=src_key_padding_mask)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output
