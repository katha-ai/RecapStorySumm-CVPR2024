#!/usr/bin/env python
"""
encoder.py: define Encoder class that will be used to encode
the input sequence.
Note: Both video and text modality encoder are available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from models.talesumm.positional_encoding import PositionalEncoding
from models.talesumm.custom_transformer import TransformerEncoderLayer
from models.talesumm.custom_encoder import _get_activation


class mlp_enc(nn.Module):
    def __init__(self,
                 input_size: int,
                 d_model: int,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 pool_type: str = "max",
                 pool_location: int = 0,
                 **kwargs
                 ) -> None:
        r"""
        Args:
            - input_size: feature size of input features extarcted from
              video frames.
            - d_model: size of output feature vector.
            - pool_type: Maxpooling or Average pooling, `max`, or `avg`.
              `default="max"`
            - pool_location: Either `0 = at first` or `-1 = last`. `default=0`
        """
        super(mlp_enc, self).__init__(**kwargs)
        self.pool_type = pool_type
        self.pool_location = pool_location
        self.mlp = nn.Sequential(nn.Linear(input_size, d_model),
                                 _get_activation(activation),
                                 nn.Dropout(dropout))

    def pool(self, X: torch.Tensor) -> torch.Tensor:
        r"""
        Apply pooling (max/avg) according to `pool_type`
        """
        if self.pool_type == "avg":
            # avoid zero division
            # tmp_mask[self.mask.sum(-2, True) == 0] += 1
            return X.sum(dim=-2)/(self.mask.sum(dim=-1).unsqueeze(dim=-1)+1e-6)
        elif self.pool_type == "max":
            return X.max(dim=-2)[0]
        else:
            raise ValueError("Enter correct pool type!")

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - X: of size `(b = batch of episodes, n = no. of shots,
                        m = no. of frames, feature_size = 1664)`
            - mask: Given for average type pooling (i.e., `avg`) of
                    size `(b, n, m)`, else `None`.
        """
        self.mask = mask
        if self.pool_location == 0:
            X = self.pool(X)
            X = self.mlp(X)
        elif self.pool_location == -1:
            X = self.mlp(X)
            X = self.pool(X)
        else:
            raise ValueError(
                f"Expected pool-location to be 0 or -1. Got {self.pool_location}.")
        return X

class simple_encoder(nn.Module):
    def __init__(self,
                 feature_size: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_model: int = 512,
                 ffn_ratio: float = 4,
                 drop_trm: float = 0.1,
                 drop_proj: float = 0.1,
                 activation_trm: str = "gelu",
                 activation_mlp: str = "gelu",
                 ) -> None:
        r"""
        Encode/Map shots (a matrix = collection of `n` frames) into a vector representation.
        Obtain the [CLS] embedding for each shot.
        -------------------------------------------------------------------------------------
        Args:
            - feature_size: Dimension of feature vetcor of one frame (e.g., 1664).
            - num_layers: number of Transformer encoder layers.
            - num_heads: number of attention heads.
            - d_model: The dimensions of token embedding. `default=512`
            - ffn_ratio: The ratio of feedforward dimension wrt `d_model`.
                `default=4`
            - drop_proj: Dropout probability for video features while projecting into `d_model`.
            - drop_trm: Dropout probability for Transformer encoder.
            - activation_trm: Activation function to be used in Transformer encoder.
            - activation_mlp: Activation function to be used in MLP (The projection operator).
        """
        super(simple_encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024)
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=ffn_ratio*d_model,
                                                dropout=drop_trm,
                                                activation=_get_activation(activation_trm),
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(d_model))
        # MLP - to project into d_model dim space.
        self.pre_linear = nn.Sequential(nn.Linear(feature_size, d_model),
                                        _get_activation(activation_mlp),
                                        nn.Dropout(drop_proj))

        self.d_model = d_model
        self.cls_emb = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=True)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self,
                X: torch.Tensor,
                mask: torch.Tensor,
                time_idx: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        r"""
        Args:
            - X: of size `(b = batch of episodes, m = no. of shots or dialog, 
              n=no. of frames or words, feature_size = e.g., 1664)`
            - src_mask (inline_given in code): For a given token which all tokens to attend.
              Hence for all tokens
              we will get stack of boolean vector = (a matrix of size = `(m, m)`).
            - mask: Which all tokens are relevant. Size = `(b, m, n)`, else `None`.
            - time_idx: Index of tokens in the shot. Size = `(b, m, n)`, else `None`.

        Returns:
            - out: of size `(b, d_model)`

        Note that here mask tensor consists of `zeros` and `ones`. `Zeros` says which are irrelevant and `ones`
        for relevant.
        """
        # reshape batch of episodes to (b*m, n, d_model)
        out_size = X.shape[0:-2] + torch.Size([self.d_model])
        src = X.reshape(-1, X.shape[-2], X.shape[-1])
        # similarly reshape mask to (b*m, n)
        mask = mask.reshape(-1, mask.shape[-1])
        if time_idx is not None:
            time_idx = time_idx.reshape(-1, time_idx.shape[-1])

        # pass through pre-linear layer
        src = self.pre_linear(src)

        # add positional encoding
        if time_idx is not None:
            src = self.pos_encoder(src, idx_to_choose=time_idx)
        else:
            src = self.pos_encoder(src)

        # add [CLS] token
        bs, seq_len, _ = src.shape
        cls_rep = self.cls_emb.repeat(bs, 1, 1)
        src = torch.cat((cls_rep, src), dim=-2)
        mask = torch.cat((torch.ones([bs, 1]).to(mask), mask), dim=1)

        # construct src_mask
        src_mask = torch.ones((seq_len+1, seq_len+1)).to(src)

        # pass through transformer encoder
        out = self.transformer_encoder(src,
                                       mask=src_mask.logical_not(),
                                       src_key_padding_mask=mask.logical_not())
        
        return out[:, 0].reshape(out_size)

class fusion_encoder(nn.Module):
    def __init__(self,
                 which_features: List[str],
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_model: int = 512,
                 ffn_ratio: float = 4,
                 drop_vid: float = 0.2,
                 drop_trm: float = 0.2,
                 activation_trm: str = "gelu",
                 activation_mlp: str = "gelu",
                 feat_fusion_style: str = "concat",
                 feat_dim_dict: Dict[str, int]={'imagenet': 1664, 'mvit': 768, 'clip': 512,
                'googlenet':1024, 'i3d_flow':1024, 'i3d_rgb':1024}) -> None:
        r"""
        Attention-based fusion of video features.
        -----------------------------------------------------------------------------------
        Args:
            - which_features: List of features to be used. Possible values are
              `["imagenet", "mvit", "clip"]`.
            - num_layers: number of Transformer encoder layers.
            - num_heads: number of attention heads.
            - d_model: The dimensions of token embedding. `default=512`
            - ffn_ratio: The ratio of feedforward dimension wrt `d_model`.
                `default=4`
            - drop_vid: Dropout probability for video features while projecting into `d_model`.
            - drop_trm: Dropout probability for Transformer encoder.
            - activation_trm: Activation function to be used in Transformer encoder.
            - activation_mlp: Activation function to be used in MLP (The projection operator).
            - feat_fusion_style: Style of fusion. Possible values are `concat`, `stack`.
                `default=concat`
            - feat_dim_dict: A dictionary of feature dimensions for each type of feature.
                For example, `{"imagenet": 1664, "mvit": 768, "clip": 512}`.
        """
        super(fusion_encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=1024)
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=ffn_ratio*d_model,
                                                dropout=drop_trm,
                                                activation=_get_activation(activation_trm),
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(d_model))

        self.feat_dims = feat_dim_dict
        # MLP - to project into d_model dim space.
        self.MLP = nn.ModuleDict()
        for feat in which_features:
            self.MLP[feat] = nn.Sequential(nn.Linear(self.feat_dims[feat], d_model),
                                           _get_activation(activation_mlp),
                                           nn.Dropout(drop_vid))

        # Attention layer
        if feat_fusion_style == "concat":
            self.attn = nn.Linear(len(which_features)*d_model, len(which_features))
        elif feat_fusion_style == "stack":
            self.attn = nn.Linear(d_model, 1)
        else:
            raise ValueError(f"Expected feat_fusion_style to be concat or stack. Got {feat_fusion_style}.")
        
        self.feat_fusion_style = feat_fusion_style
        self.which_features = which_features
        self.d_model = d_model

        # cls embedding
        self.cls_emb = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=True)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def forward(self,
                X: torch.Tensor,
                mask: torch.Tensor,
                time_idx: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        r"""
        Args:
            - X: of size `(b = batch of episodes, m = no. of shots or dialog, 
              n=no. of frames or words, d=feature_size = e.g., 1664)`
            - src_mask (inline_given in code): For a given token which all tokens to attend.
              Hence for all tokens
              we will get stack of boolean vector = (a matrix of size = `(m, m)`).
            - mask: Which all tokens are relevant. Size = `(b, m, n)`, else `None`.
            - time_idx: Index of tokens in the shot. Size = `(b, m, n)`, else `None`.

        Returns:
            - out: of size `(b, d_model)`

        Note that here mask tensor consists of `zeros` and `ones`. `Zeros` says which are irrelevant and `ones`
        for relevant.
        """
        # reshape batch of episodes to (b*m, n, d_model)
        out_size = X.shape[0:-2] + torch.Size([self.d_model])
        src = X.reshape(-1, X.shape[-2], X.shape[-1])
        # similarly reshape mask to (b*m, n)
        mask = mask.reshape(-1, mask.shape[-1])
        if time_idx is not None:
            time_idx = time_idx.reshape(-1, time_idx.shape[-1])
        # split src into different features and unpack them into dictionary.
        split_src = torch.split(src, [self.feat_dims[f] for f in self.which_features], dim=-1)
        split_src = {f: s for f, s in zip(self.which_features, split_src)}

        # project each feature into d_model dim space.
        for feat in self.which_features:
            split_src[feat] = self.MLP[feat](split_src[feat])

        # Concatenate -> linear -> tanh: (b * m, n, 3*d) -> (b * m, n, 3)
        # Stack -> linear -> tanh: (b * m, n, d, 3) -> (b * m, n, 1)
        if self.feat_fusion_style == "concat":
            fused_features = torch.tanh(self.attn(torch.cat([split_src[f] for f in self.which_features], dim=-1)))
        else:
            fused_features = torch.stack([split_src[f] for f in self.which_features], dim=-1).transpose(-2, -1)
            fused_features = torch.tanh(self.attn(fused_features).squeeze())

        # softmax -> weighted sum: (b * m, n, 3) -> (b * m, n, d_model)
        fused_features = F.softmax(fused_features, dim=-1)
        fused_features = torch.sum(torch.stack([fused_features[...,i:i+1] * split_src[f] \
                                    for i, f in enumerate(self.which_features)]), dim=0)

        # add positional encoding
        if time_idx is not None:
            fused_features = self.pos_encoder(fused_features, idx_to_choose=time_idx)
        else:
            fused_features = self.pos_encoder(fused_features)

        # add [CLS] token
        bs, seq_len, _ = fused_features.shape
        cls_rep = self.cls_emb.expand(bs, -1, -1)
        fused_features = torch.cat((cls_rep, fused_features), dim=1)
        mask = torch.cat((torch.ones([bs, 1]).to(mask), mask), dim=1)

        # construct src_mask
        src_mask = torch.ones((seq_len+1, seq_len+1)).to(fused_features)

        # pass through transformer encoder
        out = self.transformer_encoder(fused_features,
                                       mask=src_mask.logical_not(),
                                       src_key_padding_mask=mask.logical_not())
        
        return out[:, 0].reshape(out_size)


class vis_encoder_nc(nn.Module):
    def __init__(self,
                 which_features: List[str],
                 num_layers: int = 6,
                 num_heads: int = 8,
                 d_model: int = 512,
                 ffn_ratio: float = 4,
                 drop_vid: float = 0.2,
                 drop_trm: float = 0.2,
                 activation_trm: str = 'gelu',
                 activation_mlp: str = 'gelu',
                 feat_dim_dict: Dict[str, int]={'imagenet': 1664, 'mvit': 768, 'clip': 512,
                 'googlenet':1024, 'i3d_flow':1024, 'i3d_rgb':1024}) -> None:
        r"""
        Encode/Map shots (a matrix = collection of `n` frames) into a vector representation.
        Obtain the [CLS] embedding for each shot.
        This encoder instead of getting trained on `concatenated` vector it trains on each type
        of vector (namely, `MVIT`, `CLIP`, and `DenseNet-161` embeddings) separately.
        -----------------------------------------------------------------------------------
        Args:
            - which_features: List of features to be used. Possible values are
              `["imagenet", "mvit", "clip"]`.
            - num_layers: number of Transformer encoder layers.
            - num_heads: number of attention heads.
            - d_model: The dimensions of token embedding. `default=512`
            - ffn_ratio: The ratio of feedforward dimension wrt `d_model`.
                `default=4`
            - drop_vid: Dropout probability for video features while projecting into `d_model`.
            - drop_trm: Dropout probability for Transformer encoder.
            - activation_trm: Activation function to be used in Transformer encoder.
            - activation_mlp: Activation function to be used in MLP (The projection operator).
            - feat_dim_dict: A dictionary of feature dimensions for each type of feature.
                For example, `{"imagenet": 1664, "mvit": 768, "clip": 512}`.
        """
        super(vis_encoder_nc, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=2048)
        encoder_layer = TransformerEncoderLayer(d_model=d_model,
                                                nhead=num_heads,
                                                dim_feedforward=ffn_ratio*d_model,
                                                dropout=drop_trm,
                                                activation=_get_activation(activation_trm),
                                                batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(d_model))

        self.feat_type_map = dict(zip(which_features, range(1, len(which_features)+1)))

        # type embedding for each type of vector.
        # Note: don't slice embedding matrix in __init__ as it will be
        #       a problem when we will use it in forward pass.   
        self.emb = nn.Embedding(num_embeddings=len(self.feat_type_map)+1,
                                embedding_dim=d_model,
                                padding_idx=0)

        # MLP - to project into d_model dim space.
        self.MLP = nn.ModuleDict()
        for feat in which_features:
            self.MLP[feat] = nn.Sequential(nn.Linear(feat_dim_dict[feat], d_model),
                                           _get_activation(activation_mlp),
                                           nn.Dropout(drop_vid))
        self.which_features = which_features
        self.d_model = d_model

        # cls embedding
        self.cls_emb = nn.Parameter(torch.zeros(1,1,d_model), requires_grad=True)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    def reshaping_feat_mask_idx(self,
                                feat_dict: Dict[str, torch.Tensor],
                                apply_prefix: str = ''
                                ) -> Tuple[int, Dict[str, torch.Tensor]]:
    
        """
        Reshapes the feature tensor and mask tensor so as to pass it through encoder.
        """
        # Reshape all types of features to `(b, m, feat_dim)`
        # dense_feat, mvit_feat, clip_feat = vid_feat
        out_size_encoder = None
        for feat in self.which_features:
            shape_info = feat_dict[f'{apply_prefix}{feat}_enc'].shape
            if out_size_encoder is None:
                out_size_encoder = shape_info[0:-2] + torch.Size([self.d_model])

            feat_dict[f'{apply_prefix}{feat}_enc'] = \
                feat_dict[f'{apply_prefix}{feat}_enc'].reshape(-1, shape_info[-2], shape_info[-1])

        
            # reshape mask and idx
            feat_dict[f'{apply_prefix}{feat}_mask'] = feat_dict[f'{apply_prefix}{feat}_mask'].reshape(-1,
                                                feat_dict[f'{apply_prefix}{feat}_mask'].shape[-1])
            feat_dict[f'{apply_prefix}{feat}_idx'] = feat_dict[f'{apply_prefix}{feat}_idx'].reshape(-1,
                                                feat_dict[f'{apply_prefix}{feat}_idx'].shape[-1])
        return out_size_encoder, feat_dict

    def forward(self, vid_feat_dict: Dict[str, torch.Tensor], apply_prefix: str = '') -> torch.Tensor:
        r"""
        Args:
            - vid_feat_dict: A dictionary of video features. Each key is a type of feature
              whose description is explained as follows:  
                - imagenet_enc: of size `(b = batch of episodes, m = no. of shots, n= no. of frames,
                  feature_size = 1664)`
                - imagenet_mask: Which all dense tokens are relevant. Size = `(b, m, n)`.
                - imagenet_idx: Index of dense tokens in the shot. Size = `(b, m, n)`.
                - mvit_enc: of size `(b = batch of episodes, m1 = no. of shots, n1= no. of frames,
                  feature_size = 768)`
                - mvit_mask: Which all mvit tokens are relevant. Size = `(b, m1, n1)`.
                - mvit_idx: Index of mvit tokens in the shot. Size = `(b, m1, n1)`.
                - clip_enc: of size `(b = batch of episodes, m2 = no. of shots, n2= no. of frames,
                  feature_size = 512)`
                - clip_mask: Which all clip tokens are relevant. Size = `(b, m2, n2)`.
                - clip_idx: Index of clip tokens in the shot. Size = `(b, m2, n2)`.
            - Next feature is given inside code...
                - all_feats_attn_mask: For a given token which all tokens to attend. Hence for all tokens
                  we will get stack of boolean vector = (a matrix of size = `(n+n1+n2, n+n1+n2)`).
                    `n`, `n1`, `n2` are no. of frames for each type of feature.
            - apply_prefix: To change the name of keys in `vid_feat_dict` if needed (like during conditioning).

        Returns:
            - out: of size `(b, d_model)`

        Note that here mask tensor consists of `zeros` and `ones`. `Zeros` says which are irrelevant and `ones`
        for relevant.
        """
        dev = vid_feat_dict[apply_prefix+self.which_features[0]+"_enc"].device
        self.emb = self.emb.to(dev)

        # extract reshaped features, indices and masks
        out_size, vid_feat_dict = self.reshaping_feat_mask_idx(vid_feat_dict, apply_prefix=apply_prefix)

        # project clip/mvit/dense features into d_model dim space.
        for feat in self.which_features:
            vid_feat_dict[apply_prefix+feat+"_enc"] = self.MLP[feat](vid_feat_dict[apply_prefix+feat+"_enc"])

        # add type embedding
        for feat in self.which_features:
            feat_shape = vid_feat_dict[apply_prefix+feat+"_enc"].shape
            idx = torch.LongTensor([self.feat_type_map[feat]]).to(dev)
            vid_feat_dict[apply_prefix+feat+"_enc"] = vid_feat_dict[apply_prefix+feat+"_enc"] +\
                self.emb(idx).repeat(feat_shape[0], feat_shape[1], 1)

        # add positional embedding.
        for feat in self.which_features:
            vid_feat_dict[apply_prefix+feat+"_enc"] = self.pos_encoder(vid_feat_dict[apply_prefix+feat+"_enc"],
                                                                       idx_to_choose=vid_feat_dict[apply_prefix+feat+"_idx"])

        # add [CLS] embedding
        first_model_name = self.which_features[0]
        bs = vid_feat_dict[apply_prefix+first_model_name+"_mask"].shape[0]
        cls_rep = self.cls_emb.expand(bs, -1, -1)
        vid_feat_dict[apply_prefix+first_model_name+"_enc"] = torch.cat((cls_rep,
                                                                        vid_feat_dict[apply_prefix+first_model_name+"_enc"]),
                                                                        dim=1)
        vid_feat_dict[apply_prefix+first_model_name+"_mask"] = torch.cat((torch.ones([bs, 1]).to(vid_feat_dict[apply_prefix+first_model_name+"_mask"]),
                                                                        vid_feat_dict[apply_prefix+first_model_name+"_mask"]), dim=1)

        # combine all features
        comb_feat = torch.cat([vid_feat_dict[apply_prefix+feat+"_enc"]
                              for feat in self.which_features], dim=1)
        comb_mask = torch.cat([vid_feat_dict[apply_prefix+feat+"_mask"]
                              for feat in self.which_features], dim=1)

        # construct self-attention mask
        all_feats_attn_mask = torch.ones((comb_feat.shape[1], comb_feat.shape[1])).to(comb_feat)

        out = self.transformer_encoder(comb_feat,
                                       mask=all_feats_attn_mask.logical_not(),
                                       src_key_padding_mask=comb_mask.logical_not())
        return out[:, 0].reshape(out_size)
