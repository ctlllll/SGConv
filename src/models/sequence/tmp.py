if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

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

from src.models.sequence.ss.kernel import HippoSSKernel
from src.models.nn import LinearActivation, Activation, Normalization

class Learnable(nn.Module):
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
            mode="sum_randn",
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

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        self.positional_encoding_len = d_state

        # SSM Kernel
        # self.kernel =
        # 
         
        # self.positional_encoding = nn.Embedding(self.positional_encoding_len, self.channels*self.h)
        # dist = torch.arange(self.positional_encoding_len, dtype=torch.float, device=self.positional_encoding.weight.data.device)
        # coeff = torch.rand(self.channels*self.h, device=self.positional_encoding.weight.data.device)
        # self.positional_encoding.weight.data = torch.exp(-dist.pow(2).view(-1, 1) / 2) * coeff.view(1, -1)
        self.positional_encoding_freq = nn.Parameter(torch.randn(self.channels*self.h, self.positional_encoding_len))

        # self.positional_encoding.weight.data.zero_()
        # self.positional_encoding.weight.data = torch.sort(self.positional_encoding.weight.data.abs(), descending=True)[0]
        self.cached_relative_positions = {}
        print("===========================")
        print("Use learnable kernel!")
        print("===========================")

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

        self.num_scales = kernel_args.get('n_ssm', 4)
        self.kernel_dim = kernel_args.get('d_state', 64)
        self.kernel_list = nn.ParameterList()

        # self.kernel_norm = nn.LayerNorm(l_max)

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

        self.kernel_norm = None

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
        if 'decay' in self.mode:
            multiplier = torch.linspace(1, 4, self.h, device=u.device).view(1, -1, 1)
        else:
            multiplier = 2
        if 'sum' in self.mode:
            for i in range(self.num_scales):
                kernel = F.pad(
                    F.interpolate(
                        self.kernel_list[i],
                        scale_factor = 2**i,
                        mode = interpolate_mode,
                    ),
                    (0, self.kernel_dim*2**(self.num_scales-1) - self.kernel_dim*2**i),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i],
                    scale_factor = 2**max(1, i),
                    mode = interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
            
        if 'expand' in self.mode:
            k = F.interpolate(k, size=L, mode=interpolate_mode)
        else:
            if k.size(-1) > L:
                k = k[..., :L]
            elif k.size(-1) < L:
                k = F.pad(k, (0, L - k.size(-1)))

        if self.kernel_norm is None:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            # embed()
            print(f"Kernel norm: {self.kernel_norm.mean()}")
        k = k / self.kernel_norm
            
        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)

        # y = self.kernel_norm(y)
       
        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D)# u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        return y, None

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)


def test_state(random_init=False, **kwargs):
    # B = 1
    # H = 64
    # N = 64
    # L = 1024
    B = 2
    H = 3
    N = 4
    L = 8
    s4 = S4(H, d_state=N, l_max=L, **kwargs)
    s4.to(device)
    s4.eval()
    for module in s4.modules():
        if hasattr(module, 'setup_step'): module.setup_step()

    u = torch.ones(B, H, L).to(device)
    initial_state = s4.default_state(B)
    if random_init:
        if initial_state.size(-1) == N:
            initial_state = initial_state[..., :N//2]
        initial_state = torch.randn_like(initial_state)
        initial_state = torch.cat([initial_state, initial_state.conj()], dim=-1)

    state = initial_state.clone()
    y, final_state = s4(u, state=state)
    print("output:\n", y, y.shape)
    print("final state:\n", final_state, final_state.shape)

    # Use Stepping
    state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    ys = torch.stack(ys, dim=-1)
    print("step outputs:\n", ys)
    print("step final state:\n", state)

    # Use Chunking

    chunks = 4
    state = initial_state.clone()
    ys = []
    for u_ in u.chunk(chunks, dim=-1):
        y_, state = s4(u_, state=state)
        ys.append(y_)
    ys = torch.cat(ys, dim=-1)
    print("chunk outputs:\n", ys)
    print("chunk final state:\n", state)
    print("chunk output error:")
    utils.compare_outputs(y, ys)
    print("chunk final state error:")
    utils.compare_outputs(final_state, state)


if __name__ == '__main__':
    from benchmark import utils
    torch.manual_seed(42)

    device = 'cuda' # 'cpu'
    device = torch.device(device)

    test_state(random_init=True, mode='nplr', measure='legt', rank=2)
