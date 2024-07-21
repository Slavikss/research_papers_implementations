import torch
import torch.nn as nn
from labml.logger import inspect

class BatchNorm(nn.Module):
    """
    Implementation of Batch normalization

    Normalization
    It is known that whitening improves training speed and convergence. 
    Whitening is linearly transforming inputs to have zero mean, unit variance, and be uncorrelated.

    BatchNorm is very useful for deep NNs, where convariance shift maximizes without batchnorm and destructs training
    """
    def __init__(self, channels: int, eps: float=1e-5, momentum: float=0.1, affine: bool=True, track_running_stats=True):
        super().__init__()

        # number of features in input
        self.channels = channels

        # epsilon for numerical stability
        self.eps = eps

        # momentum in tanking exponential moving average
        self.momentum = momentum

        # whether to scale and shift the normalized value
        self.affine = affine

        # whether to calculate moving average or mean and variance
        self.track_running_stats = track_running_stats

        # create parameters for lambda and beta 
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.ones(channels))
        
        # register parameters for moving average mean and variance
        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        # x = [batch_size, n_channels, *]
        # where * denotes any other possibly dimensions
        # images: * = [H, W] - height and weight
        # embeddings: * = []
        # sequence: * = [L] - length of sequence
        x_shape = x.shape

        batch_size = x_shape[0]

        assert self.channels == x_shape[1]

        # reshape into [batch_sizem, n_channels, n]
        x = x.view(batch_size, self.channels, -1)

        # if we are on training mode or havent tracked running stats
        if self.training or not self.track_running_stats:

            # calculate mean across first and last dimension i.e. the means of each feature
            mean = x.mean(dim=[0,2])

            # calculate mean of squared features
            mean_x2 = (x ** 2).mean(dim=[0,2])

            # calculating variance
            var = mean_x2 - mean ** 2

            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        else:
            # use mena and var as estimates
            mean = self.exp_mean
            var = self.exp_var
            
        # normalize 
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)

        inspect(x_norm)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm * self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)

if __name__=='__main__':
    x = torch.randn([2, 3, 2, 4])
    inspect(x)

    bn = BatchNorm(3)
    x = bn(x)

    inspect(x)