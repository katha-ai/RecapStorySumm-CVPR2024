import math
import torch
import typing


class PositionalEncoding(torch.nn.Module):
    """
    Injects absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the
    embeddings so that the two can be summed.
    """

    def __init__(self, d_model: int, max_len: int = 1000)->None:
        """
        Args:
            - d_model: Embedding dimension
            - max_len: Maximum sequence length
              `Note`: This should maximum go upto 5000. If you want to go beyond that,
              you need to change the positional encoding formula - change 10,000 in `-math.log()`
              to a number such that its `half` is your desired sequence length.
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # self.d_model = d_model

    def forward(self, x: torch.Tensor, idx_to_choose: typing.Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape = [Batch_size, Seq_len, Embedding_dim]
            idx_to_choose: which all positional encoding should be chosen.
        """
        if idx_to_choose is None:
            x = x + self.pe[:, 0:x.size(1), :]
        elif idx_to_choose.ndim == 1:
            idx_to_choose = idx_to_choose.to(torch.long)
            x = x + self.pe[:, idx_to_choose, :]
        else:
            idx_to_choose = idx_to_choose.to(torch.long)
            x = x + self.pe[:, idx_to_choose, :].squeeze(0)
        return x
