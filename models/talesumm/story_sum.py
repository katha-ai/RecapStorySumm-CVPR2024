#!/usr/bin/env python

"""story_sum.py: define summarization model solely based on transformer.
Note: The self-attention mechanism is used to capture the dependencies
between every other token against every other token in the sequence.
"""

import torch
import torch.nn as nn

from typing import List, Optional, Dict
from models.talesumm.custom_encoder import _get_activation
from models.talesumm.encoder import *
from models.talesumm.decoder import decoder

class StorySum(nn.Module):
    r"""
    Tries to merge the `encoder` and `decoder` class to form a complete end-to-end model.
    This takes episode as input and tries to generate `logits` which then passed to `sigmoid`
    in loss function to generate binary prediction labels (i.e., selecting shots as well as
    dialogue to form the short summary).
    -------------------------------------------------------------------------------------------
    GOOD PART: We can partially turn on/off the encoder as well as decoder, based on design choice
    we want. E.g., For `video-baseline` models (MLP / TRM), we need only the `encoder`, while for the
    `dialogue-baseline` models (TRM), we need only the `decoder` ( or the second-level transformer) or both
    (`encoder` and `decoder`).
    """
    def __init__(self,
                 modality: str,
                 which_features: List[str],
                 which_dia_model: str,
                 vid_concatenate: bool,
                 feat_fusion_style: str,
                 withGROUP: bool,
                 enable_encoder: bool = True,
                 encoder_type: str='trm',
                 enable_dia_encoder: bool = False,
                 dia_encoder_type: str='trm',
                 pool_location: int=0,
                 enable_decoder: bool = True,
                 attention_type: str = 'full',
                 differential_attention: bool = False,
                 differential_attention_type: str = 'basic',
                 max_groups: int = 100,
                 max_pos_enc_len: int = 4000,
                 d_model: int = 512,
                 ffn_ratio: int = 4,
                 enc_layers: int = 6, dec_layers: int = 6,
                 enc_num_heads: int = 8, dec_num_heads: int = 8,
                 drop_proj: float = 0.1, drop_trm: float = 0.2, drop_fc: float = 0.4,
                 activation_trm: str = 'gelu', activation_mlp: str = 'gelu', activation_clf: str = 'relu',
                 mlp_hidden_sizes: List[int] = [],
                 init_weights: bool = True
                ) -> None:
        r"""
        Tries to merge the `encoder` and `decoder` class to form a complete end-to-end model.
        This takes episode as input and tries to generate `logits` which then passed to `sigmoid`
        to generate binary prediction labels.
        -------------------------------------------------------------------------------------------
        Args:
            - which_features: List of features to be used for video encoder. Available: `imagenet`,
              `mvit`, `clip`.
            - which_dia_model: Which dialogue model to use. Available: `roberta-large`,
              `sentence-transformers/all-mpnet-base-v2`.
            - vid_concatenate: Whether to concatenate video features or not.
            - feat_fusion_style: How to concatenate video features. Available: `concat`, `stack`, `simple`.
              Default: `concat`. Should be used only when `vid_concatenate=True`.
            - withGROUP: Whether to use `GROUP` token for separating segments.
            - encoder_type: Whether to use `trm` (for Transformer), `max` (for
              max-pooling across frames) `avg` (for averaging across frames). `default='trm'`.
            - enable_encoder: Whether to enable video encoder or not.
            - enable_dia_encoder: Whether to enable dialogue encoder (`CLS` pooling from words) or not.
            - enable_decoder: Whether to enable higher level decoder that looks at all modality.
            - attention_type: Type of attention to use. Available: `full`, `sparse`. `default=full`
            - max_groups: Maximum number of groups to consider. `default=100`
            - d_model: The dimensions of token embedding. `default=512`
            - enc_layers: number of Transformer ENCODER layers.
            - dec_layers: number of Transformer DECODER layers.
            - enc_num_heads: number of heads in Transformer ENCODER.
            - dec_num_heads: number of heads in Transformer DECODER.
            - ffn_ratio: The ratio of inner hidden size of feed-forward layer to `d_model`.
                `default=4`
            - drop_vid/dia/trm/fc: Dropout to be applied to video/dialogue/transformer/fc layers.
                `default=0.1/0.1/0.2/0.4`
            - activation_trm/mlp: Activation function to be used in Transformer/MLP layers. Default: `gelu`.
            - mlp_hidden_sizes: MLP hidden-layer sizes in form of a list. This maps the output vector
              in `n-dim` from `decoder` to `one` dimensional. `default=[]`
            - init_weights: Whether to initialize weights with Truncated Normal or not. `default=True`
        """
        super(StorySum, self).__init__()
        self.feat_dim_dict = {'imagenet':1664, 'mvit':768, 'clip':512,
                              'googlenet':1024, 'i3d_flow':1024, 'i3d_rgb':1024}
        dia_feat_dim_dict = {'fb-roberta-large':1024, 'roberta-large':1024, 'all-mpnet-base-v2': 768,
                             'all-MiniLM-L6-v2': 384, 'pegasus-large': 1024, 'mpnet-base': 768}
        # Encoder
        if enable_encoder:
            if vid_concatenate:
                feat_dim = sum([self.feat_dim_dict[feat] for feat in which_features])
                if encoder_type == 'trm':
                    if feat_fusion_style == 'simple':
                        self.encoder = simple_encoder(feature_size=feat_dim,
                                                    num_layers=enc_layers,
                                                    num_heads=enc_num_heads,
                                                    d_model=d_model,
                                                    ffn_ratio=ffn_ratio,
                                                    drop_trm=drop_trm,
                                                    drop_proj=drop_proj,
                                                    activation_trm=activation_trm,
                                                    activation_mlp=activation_mlp)
                    elif feat_fusion_style in ['concat', 'stack']:
                        self.encoder = fusion_encoder(which_features=which_features,
                                                    num_layers=enc_layers,
                                                    num_heads=enc_num_heads,
                                                    d_model=d_model,
                                                    ffn_ratio=ffn_ratio,
                                                    drop_trm=drop_trm,
                                                    drop_vid=drop_proj,
                                                    activation_trm=activation_trm,
                                                    activation_mlp=activation_mlp,
                                                    feat_fusion_style=feat_fusion_style)
                    else:
                        raise ValueError(f"Expected feat_fusion_style to be 'simple', 'concat', or 'stack'. Got {feat_fusion_style}.")
                elif encoder_type in ['max', 'avg']:
                    self.encoder = mlp_enc(input_size=feat_dim,
                                           d_model=d_model,
                                           dropout=drop_proj,
                                           activation=activation_mlp,
                                           pool_type=encoder_type,
                                           pool_location=pool_location)
                else:
                    raise ValueError(f"Expected encoder_type to be 'max', 'trm', or 'avg'. \
                        Got {encoder_type}.")
            else:
                self.encoder = vis_encoder_nc(which_features=which_features,
                                            num_layers=enc_layers,
                                            num_heads=enc_num_heads,
                                            d_model=d_model,
                                            ffn_ratio=ffn_ratio,
                                            drop_vid=drop_proj,
                                            drop_trm=drop_trm,
                                            activation_trm=activation_trm,
                                            activation_mlp=activation_mlp)
        # Dialogue Encoder
        if enable_dia_encoder:
            feat_dim = dia_feat_dim_dict[which_dia_model]
            if dia_encoder_type == 'trm':
                self.encoder_dia = simple_encoder(feature_size=feat_dim, 
                                                  num_layers=enc_layers,
                                                  num_heads=enc_num_heads,
                                                  d_model=d_model,
                                                  ffn_ratio=ffn_ratio,
                                                  drop_trm=drop_trm,
                                                  drop_proj=drop_proj,
                                                  activation_trm=activation_trm,
                                                  activation_mlp=activation_mlp)
            elif dia_encoder_type in ['max', 'avg']:
                self.encoder_dia = mlp_enc(input_size=feat_dim,
                                           d_model=d_model,
                                           dropout=drop_proj,
                                           activation=activation_mlp,
                                           pool_type=dia_encoder_type,
                                           pool_location=pool_location)
            else:
                raise ValueError(f"Expected encoder_type to be 'max', 'trm', or 'avg'. \
                    Got {encoder_type}.")

        # Decoder
        if enable_decoder:
            vid_feat_dim = sum([self.feat_dim_dict[feat] for feat in which_features]) \
                            if not enable_encoder else d_model
            dia_feat_dim = dia_feat_dim_dict[which_dia_model] \
                            if not enable_dia_encoder else d_model
            self.decoder = decoder(vid_feat_dim=vid_feat_dim,
                                   dia_feat_dim=dia_feat_dim,
                                   d_model=d_model,
                                   num_heads=dec_num_heads,
                                   ffn_ratio=ffn_ratio,
                                   modality=modality,
                                   withGROUP=withGROUP,
                                   attention_type=attention_type,
                                   differential_attention=differential_attention,
                                   differential_attention_type=differential_attention_type,
                                   max_groups=max_groups,
                                   max_pos_enc_len=max_pos_enc_len,
                                   num_layers=dec_layers,
                                   drop_vid=drop_proj,
                                   drop_dia=drop_proj,
                                   drop_trm=drop_trm,
                                   drop_fc=drop_fc,
                                   activation_trm=activation_trm,
                                   activation_mlp=activation_mlp,
                                   activation_clf=activation_clf,
                                   hidden_sizes=mlp_hidden_sizes)
        else:
            linear_layers = []
            if modality == 'vid':
                old_size = d_model
            elif modality == 'dia':
                if enable_dia_encoder:
                    old_size = d_model
                else:
                    old_size = dia_feat_dim_dict[which_dia_model]
            for size in mlp_hidden_sizes:
                linear_layers.extend([nn.Linear(old_size, size),
                                      _get_activation(activation_clf),
                                      nn.Dropout(drop_fc)])
                old_size = size
            linear_layers.append(nn.Linear(old_size, 1))
            self.mlp = nn.Sequential(*linear_layers)

        # Declare remaining attributes
        self.modality = modality
        self.d_model = d_model
        self.which_features = which_features
        self.vid_concatenate = vid_concatenate
        self.withGROUP = withGROUP
        self.attention_type = attention_type
        self.max_groups = max_groups
        self.enable_encoder = enable_encoder
        self.encoder_type = encoder_type
        self.enable_dia_encoder = enable_dia_encoder
        self.dia_encoder_type = dia_encoder_type
        self.enable_decoder = enable_decoder

        # Initialize weights
        if init_weights:
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self,
                vid_feats: Optional[Dict[str, torch.Tensor]],
                dia_feats: Optional[Dict[str, torch.Tensor]],
                time_idx: torch.Tensor,
                token_type_ids: torch.Tensor,
                group_ids: torch.Tensor,
                src_mask: torch.Tensor,
                subgroup_lens: torch.Tensor
               ) -> torch.Tensor:
        r"""
        Args:
            NOTE: `n-1` convention as we are generating summary for `n`th episode.
            - vid_feats:
                - If `concatenate=True`: Dictionary of video features for `n-1`th episode.
                It contains `vid_enc` and `vid_mask`.
                    - `vid_enc` size: `(b=batch of episodes, m=number of shots, n=number of frames, 2994)`.
                    This `2994` can change depending on the which all model's extracted features are used
                    to concatenate.
                    - `vid_mask`: Mask for video features from `n-1`th episode. Size `(b, m, n)`.
                    - May contan video features for `n`th episode depending upon the `self.condition_on_current`.
                        - Size `(b, m1, n, 2994)`.
                        - Size `(b, m1, n)`.
                - `Else`: A dictionary containing all the video features and mask for `n-1`th episode.
                  Consists of following keys (in OR fashion):
                    - imagenet_enc: Dense features of the shots of `n-1`th EPISODE. A tensor of shape
                    `(b=batch of episodes, m=no. of shots, n=no. of frames, dense_feat_dim)`.
                    - mvit_enc: MViT features of the shots of `n-1`th EPISODE. A tensor of shape
                    `(b, m, n1, mvit_feat_dim)`.
                    - clip_enc: CLIP features of the shots of `n-1`th EPISODE. A tensor of shape
                    `(b, m, n, clip_feat_dim)`.
                    - imagenet_mask: Mask for dense features. A tensor of shape `(b, m, n)`.
                    - mvit_mask: Mask for MViT features. A tensor of shape `(b, m, n1)`.
                    - clip_mask: Mask for CLIP features. A tensor of shape `(b, m, n)`.
                    - imagenet_idx: Index of the shots of `n-1`th EPISODE. A tensor of shape `(b, m, n)`.
                    - mvit_idx: Index of the shots of `n-1`th EPISODE. A tensor of shape `(b, m, n1)`.
                    - clip_idx: Index of the shots of `n-1`th EPISODE. A tensor of shape `(b, m, n)`.
                    - May contain video features for `n`th episode depending upon the `self.condition_on_current`.
                - Same keys repeat with only change that they got `cond_` as prefix.
            - dia_feats: Dictionary of dialogue features. Keys: `freezed roberta-large`, `finetuned pegasus-large`
              `freezed sentence-transformers/all-mpnet-base-v2`.
                - A dictionary containing all the dialogue features and mask for `n-1`th episode.
                  It contains following keys:
                    - `dia_enc`: A tensor of size `(b=batch of episodes, l=number of dialogues, 512)` if 
                      sentence-level encoding is used or `(b, l, k, 1024)` if word-level encoding is used
                      (with `pegasus-large`), where `k` is the number of words in the longest dialogue.
                    - `dia_mask`: Mask for dialogue features from `n-1`th episode. Size `(b, l)` or `(b, l, k)`.
                    - May contan dialogue features for `n`th episode depending upon the `self.condition_on_current`.
                        - Size `(b, l1, 512)` or `(b, l1, k1, 1024)`.
                        - Size `(b, l1)` or `(b, l1, k1)`.
                - Same keys repeat with only change that they got `cond_` as prefix.
            - time_idx: Time index of each token in the sequence.
            - token_type_ids: Token type ids to distinguish between dialogue and video tokens.
            - group_ids: Group ids to distinguish between different segments.
            - src_mask: Mask to distinguish between `PAD` and `non-PAD` tokens.
            - subgroup_lens: Length of each segment.

        Returns:
            - logits: Output logits from the decoder.
        """
        # pass through encoder
        if self.enable_encoder:
            if self.vid_concatenate:
                # pass video features through encoder & get the [CLS] embedding.
                if self.encoder_type == 'trm':
                    vid_feats = self.encoder(vid_feats['vid_enc'],
                                             vid_feats['vid_mask'],
                                             time_idx = vid_feats['vid_idx'])
                else:
                    vid_feats = self.encoder(vid_feats['vid_enc'],
                                             vid_feats['vid_mask'])
            else:
                vid_feats = self.encoder(vid_feats)

        # pass through dialogue encoder
        if self.enable_dia_encoder:
            if self.dia_encoder_type == 'trm':
                dia_feats = self.encoder_dia(dia_feats['dia_enc'],
                                             dia_feats['word_mask'],
                                             time_idx = None)
            else:
                dia_feats = self.encoder_dia(dia_feats['dia_enc'],
                                             dia_feats['word_mask'])
        
        # pass through multi-modal transformer decoder
        if self.modality != "vid" and not self.enable_dia_encoder:
            dia_feats = dia_feats['dia_enc']
        if self.enable_decoder:
            out = self.decoder(vid_feats,
                               dia_feats,
                               src_mask,
                               time_idx,
                               token_type_ids,
                               group_ids,
                               subgroup_lens)
        else:
            if self.modality == 'vid':
                out = self.mlp(vid_feats).squeeze(dim=-1)
            elif self.modality == 'dia':
                out = self.mlp(dia_feats).squeeze(dim=-1)
            else:
                raise ValueError(f"Modality {self.modality} not supported.")
        return out



if __name__ == '__main__':
    import ipdb
    model = StorySum(modality='vid',
                     which_features=['imagenet', 'mvit', 'clip'],
                     which_dia_model='pegasus-large',
                     vid_concatenate=True,
                     withGROUP=True,
                     enable_encoder=True,
                     encoder_type='trm',
                     enable_dia_encoder=True,
                     dia_encoder_type='trm',
                     pool_location=0,
                     enable_decoder=True,
                     attention_type='sparse',
                     differential_attention=True,
                     differential_attention_type='basic',
                     max_groups=100,
                     d_model=512,
                     enc_layers=6,
                     dec_layers=6,
                     mlp_hidden_sizes=[])
    # test model
    vid_feats = {'vid_enc': torch.randn(2, 10, 10, 2944),
                 'vid_mask': torch.ones(2, 10, 10),
                 'vid_idx': torch.ones(2, 10, 10)}
    dia_feats = {'dia_enc': torch.randn(2, 10, 5, 1024),
                 'word_mask': torch.ones(2, 10, 5)}
    time_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
    token_type_ids = torch.zeros(2, 22)
    group_ids = torch.tensor([[0]*10+[2]+[1]*10+[2], [0]*10+[2]+[1]*10+[2]])
    src_mask = torch.ones(2, 22)
    subgroup_lens = torch.tensor([[[5, 5], [5, 5]], [[5, 5], [5, 5]]])
    logits = model(vid_feats, dia_feats, time_idx, token_type_ids, group_ids, src_mask, subgroup_lens)
    ipdb.set_trace()

# Some mask generating links
# https://discuss.pytorch.org/t/memory-mask-in-nn-transformer/55230/5
# https://discuss.pytorch.org/t/how-to-get-memory-mask-for-nn-transformerdecoder/60414/4
# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
