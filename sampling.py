from dbus import IntrospectionParserException
import torch
import math
from torch.distributions import Categorical
import torch.nn.functional as F
from labml.logger import inspect
import torch.nn as nn


class GreedySampler:
    """
    Most basic sampler - returns most probability-high token
    """
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor):
        return x.argmax(dim=-1, keepdim=True)
    

class TemperatureSampler:
    """
    Temperature-bsaed sampler. Returns sampled token from 
    probability distribution, controlled by temperature
    """
    def __init__(self, temperature: float = 1.00):
        self.temperature = temperature

    def __call__(self, x: torch.Tensor):
        
        # Making distribution, divised by temperature parameter
        dist = Categorical(logits=x / self.temperature )

        return dist.sample().unsqueeze(0)
    

class topKSampler:
    """
    Sampler based on picking first k tokens for distribution and sampling from them
    """
    def __init__(self, k: int):
        self.k = k

    def __call__(self, x: torch.Tensor):

        # new logits filled with 0 probability
        zeros = x.new_ones(x.shape) * float('-inf')

        # getting first top k tokens and indexes
        values, indices = torch.topk(x, self.k, dim=-1)

        # filling zeros with actual logits
        zeros.scatter_(-1, indices, values)

        # getting probabilities of tokens
        probabilities = F.softmax(zeros, dim=-1)

        # Sample from the distribution defined by these probabilities
        return torch.multinomial(probabilities, num_samples=1)
    
class NucleusSampler:
    """
    Implementation of nucleus sampling technique from https://arxiv.org/abs/1904.09751
    
    Nucleus sampling suggests picking a subset of the vocabulary, where the subset is the smallest 
    set of tokens such that the sum of P(x_i|x_i:i-1) >= p.

    We pick the highest probable tokens until the sum of their probabilities is less than p,
    then sample from it.
    """
    def __init__(self, p: float):
        # p is the sum of probabilities of tokens to pick up p
        self.p = p
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, x: torch.Tensor):
        probs = self.softmax(x)

        # sort indices by probs in descending order
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)

        # getting cumulative sum of probabilities of our tokens
        cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)

        # find cumulative sums less than or equal to p
        nucleus = cum_sum_probs <= self.p

        # Ensure at least one token is included
        nucleus[..., 1:] = nucleus[..., 1:] & ~nucleus[..., :-1].cumprod(dim=-1).bool()
        nucleus[..., 0] = True

        # masking non-nucleus tokens
        sorted_probs[~nucleus] = 0

        # sampling 
        sampled_sorted_indexes = torch.multinomial(sorted_probs, num_samples=1)

        # getting actual indices
        res = indices.gather(-1, sampled_sorted_indexes)

        return res




if __name__ =='__main__':
    logits = torch.tensor([1.0, 2.0, 1.5, 0.9, 2.1, 1.6, 0.8, 0.7, 0.99])
    inspect(logits)

    greedysampler = GreedySampler()
    print('greedy samples') 
    inspect(greedysampler(logits))
    inspect(greedysampler(logits))

    tempsampler = TemperatureSampler()
    print('temperature sampler samples') 
    inspect(tempsampler(logits))
    inspect(tempsampler(logits))

    topksampler = topKSampler(k=8)
    print('topk samples')
    inspect(topksampler(logits))
    inspect(topksampler(logits))

    nucleussampler = NucleusSampler(2.0)
    print('nucleus samples')   
    inspect(nucleussampler(logits))
    inspect(nucleussampler(logits))


    

