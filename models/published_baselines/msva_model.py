from models.talesumm.encoder import mlp_enc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSVA_LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(MSVA_LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MSVA_SelfAttention(nn.Module):
    def __init__(self, apperture, input_size, output_size, dropout=0.5):
        super(MSVA_SelfAttention, self).__init__()
        self.apperture = apperture
        self.inp_dim = input_size
        self.out_dim = output_size
        self.K = nn.Linear(in_features=self.inp_dim, out_features=self.out_dim, bias=False)
        self.Q = nn.Linear(in_features=self.inp_dim, out_features=self.out_dim, bias=False)
        self.V = nn.Linear(in_features=self.inp_dim, out_features=self.out_dim, bias=False)
        self.output_linear = nn.Linear(in_features=self.out_dim, out_features=self.inp_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(-2,-1))
        if self.apperture > 0:
            onesmask = torch.ones(x.shape[0], x.shape[1], x.shape[1])
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(att_weights_)
        y = torch.matmul(V.transpose(-2,-1), weights).transpose(-2,-1)
        y = self.output_linear(y)
        return y, att_weights_


class MSVA_base(nn.Module):
    def __init__(self, apperture, inp_dim, hidden_dim, out_dim, dropout=0.5):
        super(MSVA_base, self).__init__()
        self.apperture = apperture
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.att1_3 = MSVA_SelfAttention(apperture=self.apperture, input_size=self.inp_dim, output_size=self.inp_dim, dropout=dropout)
        self.ka1_3 = nn.Linear(in_features=self.inp_dim , out_features=self.hidden_dim)
        self.kb = nn.Linear(in_features=self.ka1_3.out_features, out_features=self.hidden_dim)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.hidden_dim)
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=self.out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_y_1_3 = MSVA_LayerNorm(self.inp_dim)
        self.layer_norm_kc = MSVA_LayerNorm(self.kc.out_features)

    def forward(self, x_list):
        # Intermediate fusion
        y_out_ls = []
        for i in range(len(x_list)):
            y, _ = self.att1_3(x_list[i])
            y = y + x_list[i]
            y = self.dropout(y)
            y = self.layer_norm_y_1_3(y)
            y_out_ls.append(y)
        y_out = torch.stack(y_out_ls)
        y = torch.sum(y_out, dim=0)
        y = self.ka1_3(y)
        y = self.kb(y)
        y = self.kc(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.layer_norm_kc(y)
        y = self.kd(y)
        y = y.view(y.shape[0], -1)
        return y


class MSVA_adapted(nn.Module):
    def __init__(self,
                 d_model: int = 128,
                 drop_proj: float = 0.1,
                 drop_trm: float = 0.2,
                 first_feat_dim: int = 1664,
                 second_feat_dim: int = 768,
                 third_feat_dim: int = 512,
                 encoder_type: str = 'max',
                 activation_mlp: str = 'gelu',
                 msva_out_dim: int = 1,
                 apperture: int = 250,
                ) -> None:
        super(MSVA_adapted, self).__init__()
        self.apperture = apperture
        self.msva_hidden_dim = d_model
        self.d_model = d_model
        self.first_feat_dim = first_feat_dim
        self.second_feat_dim = second_feat_dim
        self.third_feat_dim = third_feat_dim
        self.msva_out_dim = msva_out_dim
        self.encoder_type = encoder_type
        if self.encoder_type in ['max', 'avg']:
            self.image_shot_enc = mlp_enc(input_size=first_feat_dim, d_model=self.d_model,
                                            dropout=drop_proj, activation=activation_mlp,
                                            pool_type=self.encoder_type, pool_location=-1)
            self.motion_shot_enc = mlp_enc(input_size=second_feat_dim, d_model=self.d_model,
                                            dropout=drop_proj, activation=activation_mlp,
                                            pool_type=self.encoder_type, pool_location=-1)
            self.clip_shot_enc = mlp_enc(input_size=third_feat_dim, d_model=self.d_model,
                                            dropout=drop_proj, activation=activation_mlp,
                                            pool_type=self.encoder_type, pool_location=-1)
        else:
            raise ValueError('encoder_type must be one of [trm, max, avg]. Got {}'.format(self.encoder_type))
        self.msva_base = MSVA_base(apperture=apperture, inp_dim=self.d_model, dropout=drop_trm,
                                   hidden_dim=self.msva_hidden_dim, out_dim=self.msva_out_dim)

    def forward(self, 
                X: torch.Tensor,
                mask: Optional[torch.Tensor]
               ) -> torch.Tensor:
        imagenet_feats = X[:,:,:,:self.first_feat_dim]
        mvit_feats = X[:,:,:,self.first_feat_dim:self.first_feat_dim+self.second_feat_dim]
        clip_feats = X[:,:,:,-self.third_feat_dim:]
        stream1_inp = self.image_shot_enc(X=imagenet_feats, mask=mask)
        stream2_inp = self.motion_shot_enc(X=mvit_feats, mask=mask)
        stream3_inp = self.clip_shot_enc(X=clip_feats, mask=mask)
        msva_preds = self.msva_base([stream1_inp, stream2_inp, stream3_inp])
        return msva_preds


if __name__ == "__main__":
    mod = MSVA_adapted().eval()
    inp = torch.rand(5, 60, 100, 1664+768+512)
    mask = torch.zeros(5, 60, 100)
    out = mod(inp, mask)
