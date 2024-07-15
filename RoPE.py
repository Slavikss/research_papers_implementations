import torch
from torch import nn
from labml.logger import inspect

class RotaryPositionEmbeddings(nn.Module):
    """
    Rotary encoding transforms pair of features by rotating in 2D plane. 
    We can have d features, each of d/2 features organized separately. 
    Each pair can be considered a coodinate in 2D plane and rotating it by an angle
    depending on token's position

    We use it for Q and K 

    Important: for dot-product attention, RoPE is relative attention i.e:
    <RoPE(x1,x2,m), RoPE (x1,x2,n)> = <RoPE(x1,x2,m-n),RoPE(x1,x2,0)>
    """
    def __init__(self, d: int, base: int = 10000):
        """
        d is the number of features d
        base is constant used for calculating theta
        """ 
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache cos and sin values
        """
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return 
        
        seq_len = x.shape[0]

        # apply theta for every token in pair
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # create position indexes [0, .., seq_len -1]
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # dot product between poisition index and theta_i
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # concatenate so that for row m we have [mtheta0, ..., mtheta_d/2, mtheta0, ..., mtheta_d/2]
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        """
        calculate [-x^(d/2 + 1),-x^(d/2), ..., -x^(d), -x^(d/2)]
        """
        d_2 = self.d // 2

        return torch.cat([-x[:,:,:,d_2:], x[:,:,:, :d_2]], dim=-1)
    
    def forward(self, x: torch.Tensor):
        """
        x.shape = [seq_len, batch_size, n_heads, d]
        """
        self._build_cache(x)

        # split the features, we can choose to apply rotary embeddings only to a partial set of features
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # calculating neg_half
        neg_half_x = self._neg_half(x_rope)

        # calculating matmul between rotation matrix and our features
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        return torch.cat([x_rope, x_pass], dim=-1)
    

def _test_rotary():

   x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
   x = x[:, None, None, :]
   inspect(x)
   rotary_pe = RotaryPositionEmbeddings(4)
   inspect(rotary_pe(x))


if __name__ == '__main__':
    _test_rotary()