#!/usr/bin/env python
"""
decoder.py: define Decoder class that will be used to encode
the input sequence.
Note: Both video and text modality decoder are available.
"""

import torch
import warnings
import torch.nn as nn
from typing import List, Optional
from models.talesumm.positional_encoding import PositionalEncoding
from models.talesumm.custom_encoder import CustomTransformerEncoder, _get_activation
from models.talesumm.custom_transformer import TransformerEncoderLayer

class decoder(nn.Module):
    r"""
    Takes shot embedding and dialogue embedding from `(n-1)th` EPISODE and
    try to classify based on whether shots should be taken into RECAP or not.
    This is an all-purpose decoder which can be used for only `video` or `dialogue`
    modalities or `both`.
    ----------------------------------------------------------------------------
    `Note:` that Transformer's `encoder` and `decoder` differs only at the
    `cross-attention` technique.
    """

    def __init__(self,
                 vid_feat_dim: int,
                 dia_feat_dim: int,
                 d_model: int = 128,
                 num_heads: int = 8,
                 ffn_ratio: float = 4.0,
                 modality: str = 'both',
                 withGROUP: bool = True,
                 attention_type: str = 'full',
                 differential_attention: bool = False,
                 differential_attention_type: str = 'basic',
                 max_groups: int = 100,
                 max_pos_enc_len: int = 4000,
                 num_layers: int = 6,
                 drop_vid: float = 0.1,
                 drop_dia: float = 0.1,
                 drop_trm: float = 0.2,
                 drop_fc: float = 0.4,
                 activation_trm: str = "gelu",
                 activation_mlp: str = "gelu",
                 activation_clf: str = "relu",
                 hidden_sizes: List = [],
                 ) -> None:
        r"""
        ----------------------------------------------------------------------------
        Args:
            - vid_feat_dim: Dimension of video feature vector (e.g., 512).
            - dia_feat_dim: Dimension of dialogue feature vector (e.g., 512).
            - d_model: Dimension of the common feature space. `default=128`.
            - num_heads: number of heads in Transformer layers. `default=8`.
            - ffn_ratio: Ratio of `dim_feedforward` to `d_model` in
                `TransformerEncoderLayer`. `default=4.0`.
            - modality: Type of modality to be used. Available options are `vid` for video,
              `dia` for dialogue, or both. `default='both'`.
            - withGROUP: Whether to add GROUP (or <SEP> informally) token or not. `default=True`
            - attention_type: Type of attention to be used. Available options are
              `full` and `sparse`. `default='full'`.
            - differential_attention: When `True`, each encoder layer will have different
              `src_attention` mask. It shhould be used when `attention_type = sparse`.
              `default=False`.
            - differential_attention_type: Type of differential attention to be used.
              `basic` repeats the src_attention mask for alternate layer, while in `advanced`
              the attention mask for 3rd is combination of 1st and 2nd. Should be given when `differential_attention = True`. `default='basic'`.
            - max_groups: Maximum number of groups to be considered. `default=100`.
            - num_layers: number of Transformer decoder layers.
            - max_pos_enc_len: maximum length of positional encoding.
            - drop_vid: dropout ratio while projecting video features. `default=0.1`.
            - drop_dia: dropout ratio while projecting dialogue features. `default=0.1`.
            - drop_trm: dropout ratio for Transformer layers. `default=0.2`.
            - drop_fc: dropout ratio for MLP layers. `default=0.4`.
            - activation_trm: activation function for Transformer layers. `default=gelu`.
            - activation_mlp: activation function for MLP layers (during common space projection).
              `default=gelu`.
            - activation_clf: activation function for classification head. `default=relu`.
            - hidden_sizes: MLP hidden-layer sizes in form of a list. This maps the output vector
              in `n-dim` from `decoder` to `one` dimensional. `default=[]`
        """
        super(decoder, self).__init__()
        self.d_model = d_model
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_pos_enc_len)
        decoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=ffn_ratio*d_model,
                                                dropout=drop_trm,
                                                activation=_get_activation(activation_trm),
                                                batch_first=True)

        if attention_type == "sparse" and differential_attention and num_layers%3 != 0:
            warnings.warn(message=('`num_layers` should be multiple of 3 when `differential_attention` is True and '
                                   '`attention_type` is sparse. Setting `differential_attention` to False.'),
                          category=UserWarning)
            differential_attention = False

        self.transformer_decoder = CustomTransformerEncoder(decoder_layer,
                                                            num_layers=num_layers,
                                                            norm=nn.LayerNorm(d_model),
                                                            per_layer_src_mask=differential_attention)

        # Type embeddings for video, dialogue, [GROUP], and <PAD> tokens.
        # {'vid': 0, 'dia': 1, 'cls': 2, 'pad': 3}
        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=d_model, padding_idx=3)
        self.modality = modality
        self.withGROUP = withGROUP
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attention_type = attention_type
        self.differential_attention = differential_attention
        self.differential_attention_type = differential_attention_type
        # Group type embeddings
        # NOTE: If you want different initializations for each [GROUP] token
        # then you can use another `nn.Embedding` with `num_embeddings=max_groups`.
        # The fact is, `group_type` along with group embedding i.e., idx=2 of `self.emb` are sufficient.
        if self.attention_type == 'sparse':
            self.group_idx_emb = nn.Embedding(num_embeddings=max_groups,
                                              embedding_dim=d_model,
                                              padding_idx=0)

        # MLP - to project into d_model dim space.
        if vid_feat_dim != d_model and modality != 'dia':
            self.visMLP = nn.Sequential(nn.Linear(vid_feat_dim, d_model), _get_activation(activation_mlp), nn.Dropout(drop_vid))
        if dia_feat_dim != d_model and modality != 'vid':
            self.diaMLP = nn.Sequential(nn.Linear(dia_feat_dim, d_model), _get_activation(activation_mlp), nn.Dropout(drop_dia))

        # Calssification head MLP
        linear_layers = []
        old_size = d_model
        for size in hidden_sizes:
            linear_layers.extend([nn.Linear(old_size, size), _get_activation(activation_clf), nn.Dropout(drop_fc)])
            old_size = size
        linear_layers.append(nn.Linear(old_size, 1))
        self.mlp = nn.Sequential(*linear_layers)

    def forward(self,
                vid_feats: Optional[torch.Tensor],
                dia_feats: Optional[torch.Tensor],
                mask: torch.Tensor,
                time_idx: torch.Tensor,
                token_type_ids: torch.Tensor,
                group_ids: torch.Tensor,
                subseq_len: torch.Tensor,
                ) -> torch.Tensor:
        r"""
        Args:
            - vid_feats: of size `(b = batch of episodes, m = no. of shots, feature_size = e.g., 512)`
            - dia_feats: of size `(b, n, feature_size = e.g., 512)`. It should follow the
              temporal order of dialogue. Same goes for visual features.
            - mask: of size `(b, m+n+eps)`. `1` for relevant and `0` for irrelevant.
              Note: Here `eps` is a small number to account for the case for `GROUP` tokens.
            - time_idx: of size `(b, m+n+eps)`. Relative time index of each shot and dialogue for each episode.
            - token_type_ids: of size `(b, m+n+eps)`. `0` for video, `1` for dialogue, `2` for `[CLS]`.
            - group_ids: of size `(b, m+n+eps)`. a positive integer index for whole group/story-segment.
            - subseq_len: of size `(b, k, 2)`. `k` is the number of sub-sequences (or group). `2` is for
              length of `vid` and `dia` tokens for each sub-sequence.

        Returns:
            - out: of size `(b, m+n+eps)`

        Note that here mask tensor consists of `zeros` and `ones`. `Zeros` says which are irrelevant and `ones`
        for relevant.
        """
        # define device
        device = mask.device
        # batch length and token sequence length
        batch_len, seq_len = mask.shape

        # project to d_model dimension if required.
        if vid_feats is not None and vid_feats.shape[-1] != self.d_model:
            # instead can say self.modality == 'vid' or 'both'
            vid_feats = self.visMLP(vid_feats)
        if dia_feats is not None and dia_feats.shape[-1] != self.d_model:
            # instead can say self.modality == 'dia' or 'both'
            dia_feats = self.diaMLP(dia_feats)

        # construct batch feats by slicing and concatenating
        # valid video and dialogue features...
        batch_feats = torch.zeros(0, seq_len, self.d_model).to(device)
        for i in range(batch_len):
            v, d = 0, 0
            comb_feats = torch.zeros(1, 0, self.d_model).to(device)
            till_what = subseq_len[i].sum(dim=-1)>0
            for vid_bound, dia_bound in subseq_len[i][till_what]:
                if vid_bound > 0 and dia_bound > 0:
                    if self.modality == 'both':
                        comb_feats = torch.cat([comb_feats,
                                                vid_feats[i:i+1, v:v+vid_bound],
                                                dia_feats[i:i+1, d:d+dia_bound],
                                                torch.zeros(1, 1, self.d_model).to(device)], dim=1)
                    elif self.modality == 'vid':
                        comb_feats = torch.cat([comb_feats,
                                                vid_feats[i:i+1, v:v+vid_bound],
                                                torch.zeros(1, dia_bound, self.d_model).to(device),
                                                torch.zeros(1, 1, self.d_model).to(device)], dim=1)
                    elif self.modality == 'dia':
                        comb_feats = torch.cat([comb_feats,
                                                torch.zeros(1, vid_bound, self.d_model).to(device),
                                                dia_feats[i:i+1, d:d+dia_bound],
                                                torch.zeros(1, 1, self.d_model).to(device)], dim=1)
                else:
                    break
                v += vid_bound
                d += dia_bound
            # pad the rest with [PAD] token
            comb_feats = torch.cat([comb_feats,
                                    torch.zeros(1, seq_len - comb_feats.shape[1], self.d_model).to(device)], dim=1)
            batch_feats = torch.cat([batch_feats, comb_feats], dim=0)

        # add time embedding
        batch_feats = self.pos_decoder(batch_feats, idx_to_choose=time_idx)
        # Aannihilate stuffs added to [CLS] as it don't have any time info
        # in actual video if withGROUP is True. If False `mask` will take care of it.
        if self.withGROUP:
            batch_feats *= (token_type_ids != 2).unsqueeze(dim=2).to(torch.float32)
        # add token-type embedding
        batch_feats += self.emb(token_type_ids.to(torch.long))
        # add group-idx embedding
        if self.attention_type == 'sparse':
            batch_feats += self.group_idx_emb(group_ids.to(torch.long))
            # Construct sparse attention mask
            src_attn_mask = torch.zeros((0, seq_len, seq_len)).to(device)
            if self.differential_attention:
                global_batch_mask = torch.zeros((0, seq_len, seq_len)).to(device)
            # iterate over each episode
            for i in range(batch_len):
                # preparing src attention mask
                sample_mask = torch.zeros((seq_len, seq_len)).to(device)
                valid_grp_idx = 0
                num_grps = len(subseq_len[i])
                if not self.withGROUP:
                    not_GRP_mask = (token_type_ids[i] != 2).to(torch.float32)
                while (valid_grp_idx < num_grps) and (subseq_len[i][valid_grp_idx].sum(dim=-1)>0):
                    valid_grp_idx += 1
                    seg_mask = (group_ids[i] == valid_grp_idx).to(torch.float32)
                    if not self.withGROUP:
                        seg_mask *= not_GRP_mask
                    sample_mask = torch.logical_or(sample_mask, torch.outer(seg_mask, seg_mask))
                if self.withGROUP:
                    group_tok_mask = (token_type_ids[i] == 2).to(torch.float32)
                    group_tok_mask = torch.outer(group_tok_mask, group_tok_mask)
                    if self.differential_attention:
                        global_batch_mask = torch.cat([global_batch_mask, group_tok_mask.unsqueeze(0)], dim=0)
                    else:
                        sample_mask = torch.logical_or(sample_mask, group_tok_mask)
                src_attn_mask = torch.cat([src_attn_mask, sample_mask.unsqueeze(0)], dim=0)
            src_attn_mask = torch.repeat_interleave(src_attn_mask, self.num_heads, dim=0)

            # preparing layer wise attention mask
            if self.differential_attention:
                global_batch_mask = torch.repeat_interleave(global_batch_mask, self.num_heads, dim=0)
                tmp_mask = []
                if self.differential_attention_type == 'basic':
                    for _ in range(self.num_layers//3):
                        tmp_mask.extend([src_attn_mask, global_batch_mask, src_attn_mask])
    
                elif self.differential_attention_type == 'advanced':
                    for _ in range(self.num_layers//3):
                        tmp_mask.extend([src_attn_mask, global_batch_mask, torch.logical_or(src_attn_mask, global_batch_mask)])
                else:
                    raise NotImplementedError('differential_attention_type must be either basic or advanced')
                src_attn_mask = torch.stack(tmp_mask)
        else:
            src_attn_mask = torch.ones((seq_len, seq_len)).to(device)  # here 2D is OK
        # pass through transformer decoder
        out = self.transformer_decoder(batch_feats,
                                       mask=src_attn_mask.logical_not(),
                                       src_key_padding_mask=mask.logical_not())
        # pass through MLP
        logits = self.mlp(out)
        return logits.squeeze(dim=-1)



if __name__ == '__main__':
    # declare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test
    model = decoder(vid_feat_dim=1024,
                    dia_feat_dim=768,
                    modality='both',
                    attention_type='sparse',
                    withGROUP=True,
                    dropout=0.2,
                    differential_attention=False,
                    differential_attention_type='advanced',
                    num_layers=3)
    
    model.to(device)
    model.eval()
    # test data
    vid_feats = torch.randn(2, 15, 1024).to(device)
    dia_feats = torch.randn(2, 15, 768).to(device)
    mask = torch.ones(2, 33).to(device)
    time_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                              22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                              22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]).to(device)
    token_type_ids = torch.tensor(
        [[0]*5 + [1]*5 + [2] + [0]*5 + [1]*5 + [2] + [0]*5 + [1]*5 + [2],
         [0]*5 + [1]*5 + [2] + [0]*5 + [1]*5 + [2] + [0]*5 + [1]*5 + [2]]).to(device)
    group_ids = torch.tensor(
        [[1]*11 + [2]*11 + [3]*11, [1]*11 + [2]*11 + [3]*11]).to(device)
    subseq_len = torch.tensor([[[5, 5], [5, 5], [5, 5]], [[5, 5], [5, 5], [5, 5]]]).to(device)
    logits = model(vid_feats, dia_feats, mask, time_idx, token_type_ids, group_ids, subseq_len)
    print(logits.shape)
    # import ipdb; ipdb.set_trace()
