import math
from typing import Optional, List
import torch
from torch import nn
from labml import tracker

# got from https://arxiv.org/abs/1706.03762
#     and https://nlp.seas.harvard.edu/2018/04/03/attention.html
#     and https://nn.labml.ai/?_gl=1*1l3fv1n*_ga*NDc0MzgxMjA0LjE2OTE1MDI4ODU.*_ga_PDCL9PHMHT*MTY5MTUwMjg4NC4xLjEuMTY5MTUwMjkxMi4wLjAuMA

class PrepareForMultiheadAttention(nn.Module):
    """
    Module to prepare linear layer to 3 splited linear layers for QKV  
    """
    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()

        # Init layer to transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)

        # Number of heads in attention
        self.heads = heads

        # Number of dimensions in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor): 
        # input has shape [seq_len, batch_size, d_model] or [batch_size, d_model]
        # We wan tto split last dim into heads 
        head_shape = x.shape[:-1]

        x = self.linear(x)

        # reshape to heads
        x = x.view(*head_shape, self.heads, self.d_k)

        return x
    
class MultiHeadAttention(nn.Module):
    """
    Main Multi-head attention module

    softmax(QK^t/sqrt(d_k)) * V

    It finds key that find query and gives value of it

    Uses dot-product before V to indicate how q and k are similar

    Also it normalizes vy sqty(d_k) to avoid large dot-products, when softmax gives small gradients in big d_k
    """
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()

        assert d_model % heads == 0, "heads must be divisor of d_model"
        # get dimension of number of features in each head
        self.d_k = d_model // heads

        self.heads = heads

        # init Q K and V
        self.query = PrepareForMultiheadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiheadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiheadAttention(d_model, heads, self.d_k, bias=bias)

        # Init softmax that applies for 1-st dim(along key dimension)
        self.softmax = nn.Softmax(dim=1)

        # output layer
        self.output = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout_prob)

        # scaling before softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # store attns for logging 
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        Calculate scores between query ans key

        This method can be overridden for other implementations of attention
        """
        # for each combination of i,j,b,h elements of the tensors are multiplied 
        # and summed over dimension d --> output dim = i,j,b,h
        return torch.einsum('ibhd, jbhd->ijbh', query, key)
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        Prepare mask to QK^T

        mask has shape [seq_len_q, seq_len_k, batch_size]
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # same mask applied to all heads
        mask = mask.unsqueeze(-1)

        # [seq_len_q, seq_len_k, batch_size, heads]
        return mask
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape

        if mask is not None:mask = self.prepare_mask(mask, list(query.shape), list(key.shape))

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        scores = self.get_scores(query, key)

        scores *= self.scale

        # apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = self.softmax(scores)

        # debug if needed
        # tracker.debug('attn', attn)

        attn = self.dropout(attn)

        # multiplying softmaxed QK^T/sqrt(d_k) by V
        x = torch.einsum('ijbh, jbhd->ibhd', attn, value)

        self.attn = attn.detach()

        # concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        x = self.output(x)

        return x

        