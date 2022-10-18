# Modified from S4: https://github.com/HazyResearch/state-spaces/blob/main/src/models/sequence/ss/s4.py
# We will release the whole codebase upon acceptance.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from einops import rearrange, repeat
from omegaconf import DictConfig
import opt_einsum as oe
import numpy as np
from IPython import embed

optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

from src.models.nn import LinearActivation, Activation, Normalization

class GConv(nn.Module):
    requires_length = True

    def __init__(
            self,
            d_model,
            d_state=64,
            l_max=1, # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            ln=False, # Extra normalization
            postact=None, # activation after FF
            initializer=None, # initializer on FF
            weight_norm=False, # weight normalization on FF
            hyper_act=None, # Use a "hypernetwork" multiplication
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            verbose=False,
            shift=False,
            linear=False,
            mode="cat_randn",
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()
        if verbose:
            import src.utils.train
            log = src.utils.train.get_logger(__name__)
            log.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.mode = mode
        self.l_max = l_max

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # Pointwise
        if not self.linear:
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
            if self.ln:
                self.norm = Normalization(self.h*self.channels, transposed=transposed)
            else:
                self.norm = nn.Identity()

        # position-wise output transform to mix features
        if not self.linear:
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

        self.init_scale = kernel_args.get('init_scale', 0)
        self.kernel_dim = kernel_args.get('kernel_dim', 64)
        self.num_scales = kernel_args.get('n_scales', 1+math.ceil(math.log2(l_max/self.kernel_dim))-self.init_scale)
        if self.num_scales is None:
            self.num_scales = 1 + math.ceil(math.log2(l_max/self.kernel_dim)) - self.init_scale
        self.kernel_list = nn.ParameterList()

        decay_min = kernel_args.get('decay_min', 1)
        decay_max = kernel_args.get('decay_max', 4)
        
        for _ in range(self.num_scales):
            if 'randn' in mode:
                kernel = nn.Parameter(torch.randn(channels, self.h, self.kernel_dim))
            elif 'cos' in mode:
                kernel = nn.Parameter(torch.cat([torch.cos(torch.linspace(0, 2*i*math.pi, self.kernel_dim)).expand(channels, 1, self.kernel_dim) for i in range(self.h)], dim=1)[:, torch.randperm(self.h), :])
            else:
                raise ValueError(f"Unknown mode {mode}")
            kernel._optim = {
                'lr': kernel_args.get('lr', 0.001),
            }
            self.kernel_list.append(kernel)
            
        if 'learnable' in mode:
            self.decay = nn.Parameter(torch.rand(self.h) * (decay_max - decay_min) + decay_min)
            if 'fixed' in mode:
                self.decay.requires_grad = False
            else:   
                self.decay._optim = {
                    'lr': kernel_args.get('lr', 0.001),
                }
            self.register_buffer('multiplier', torch.tensor(1.0))
        else:
            self.register_buffer('multiplier', torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1))

        self.register_buffer('kernel_norm', torch.ones(self.h, 1))
        self.register_buffer('kernel_norm_initialized', torch.tensor(0, dtype=torch.bool))


    def forward(self, u, state=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        kernel_list = []
        interpolate_mode = 'nearest' if 'nearest' in self.mode else 'linear'
        multiplier = self.multiplier
        if 'sum' in self.mode:
            for i in range(self.num_scales):
                kernel = F.pad(
                    F.interpolate(
                        self.kernel_list[i],
                        scale_factor = 2**(i+self.init_scale),
                        mode = interpolate_mode,
                    ),
                    (0, self.kernel_dim*2**(self.num_scales-1+self.init_scale) - self.kernel_dim*2**(i+self.init_scale)),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor = 2**(max(0, i-1)+self.init_scale),
                    mode = interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if 'learnable' in self.mode:
            k = k * torch.exp(-self.decay.view(1, -1, 1)*torch.log(torch.arange(k.size(-1), device=k.device)+1).view(1, 1, -1))

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(1, dtype=torch.bool, device=k.device)
            print(f"Kernel norm: {self.kernel_norm.mean()}")
            print(f"Kernel size: {k.size()}")  

        if k.size(-1) > L:
            k = k[..., :L]
        elif k.size(-1) < L:
            k = F.pad(k, (0, L - k.size(-1)))

        k = k / self.kernel_norm #* (L / self.l_max) ** 0.5
            
        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)
       
        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        return y, None

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)