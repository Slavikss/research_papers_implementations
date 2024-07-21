from numpy import isin
import torch
import torch.nn as nn
from typing import Union, List
from labml.logger import inspect

class LayerNorm(nn.Module):
    """
    Layer normalization transforms the inputs to have zero mean and unit 
    variance across the features. Note that batch normalization fixes 
    the zero mean and unit variance for each element. Layer normalization 
    does it for each batch across all elements.
    

    Limitations of Batch Normalization
    You need to maintain running means.
    Tricky for RNNs. Do you need different normalizations for each step?
    Doesn't work with small batch sizes; large NLP models are usually trained with small batch sizes.
    Need to compute means and variances across devices in distributed training.

    """
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], *, eps: float=1e-5, elementwise_affine: bool=True):
        super().__init__()


        # convert normalized_shape to torch.Size
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)


        self.normalized_shape = normalized_shape
        self.eps = eps
        # scale and shift analogue for layernorm
        self.elementwise_affine = elementwise_affine

        # create parameters for lambda and beta for gain and bias
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

        
    def forward(self, x: torch.Tensor):
        # x = [*, S[0], ..., S[n]]
        # * could by any number of dimensions

        # sanity check for shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # calculate dims where layernorm apply to
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # calculate mean of all elements
        mean = x.mean(dim=dims, keepdim=True)

        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)

        var = mean_x2 - mean ** 2

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # scale and shift
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm
    
if __name__ == "__main__":
    x = torch.zeros([2, 3, 2, 4])
    inspect(x.shape)
    ln = LayerNorm(x.shape[2:])

    x = ln(x)
    inspect(x.shape)
