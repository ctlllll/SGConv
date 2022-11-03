# [What Makes Convolutional Models Great on Long Sequence Modeling?](https://arxiv.org/abs/2210.09298)

*Yuhong Li\*, Tianle Cai\*, Yi Zhang, Deming Chen, Debadeepta Dey*

## Update log

- 10/17/2022: Core code released.
- 11/3/2022: Standalone code released. Easier to use if you want to try SGConv on your own model!
- Upcoming: Full code release.

## Overview

Convolutional models have been widely used in multiple domains. However, most existing models only use local convolution, making the model unable to handle long-range dependency efficiently. Attention overcomes this problem by aggregating global information but also makes the computational complexity quadratic to the sequence length. 

Recently, Gu et al. [2021] proposed a model called S4 inspired by the state space model. S4 can be efficiently implemented as a global convolutional model whose kernel size equals the input sequence length. S4 can model much longer sequences than Transformers and achieve significant gains over SoTA on several long-range tasks. Despite its empirical success, S4 is involved. It requires sophisticated parameterization and initialization schemes. As a result, S4 is less intuitive and hard to use. 

Here we aim to demystify S4 and extract basic principles that contribute to the success of S4 as a global convolutional model. We focus on the structure of the convolution kernel and identify two critical but intuitive principles enjoyed by S4 that are sufficient to make up an effective global convolutional model: 

1. The parameterization of the convolutional kernel needs to be efficient in the sense that the number of parameters should scale sub-linearly with sequence length. ![image](imgs/decay.png)
2. The kernel needs to satisfy a decaying structure that the weights for convolving with closer neighbors are larger than the more distant ones. 

Based on the two principles, we propose a simple yet effective convolutional model called Structured Global Convolution (SGConv). 

![image](imgs/sgconv.png)

SGConv exhibits strong empirical performance over several tasks: 
1. With faster speed, SGConv surpasses S4 on Long Range Arena and Speech Command datasets. ![image](imgs/lra.png)
2. When plugging SGConv into standard language and vision models, it shows the potential to improve both efficiency and performance.



## Code

Based on the amazing [codebase](https://github.com/HazyResearch/state-spaces) by HazyResearch. Please refer to the repo to install the dependencies.

### Standalone
In *gconv_standalone.py*, we provide a standalone implementation of SGConv. You can use it as a drop-in replacement for your existing models. The example of how to use it is shown in *test.ipynb*, where we tried on sequence with *1M* tokens and it cost ~20GB GPU memory per layer.

```python
import torch
from gconv_standalone import GConv

layer = GConv(
    d_model=256,
    d_state=64,
    l_max=1_000_000,
    bidirectional=True,
    kernel_dim=32,
    n_scales=None,
    decay_min=2,
    decay_max=2,
)

x = torch.randn(1, 256, 1_000_000)
y, k = layer(x, return_kernel=True)
```


## Citation

```
@misc{li2022makes,
      title={What Makes Convolutional Models Great on Long Sequence Modeling?}, 
      author={Yuhong Li and Tianle Cai and Yi Zhang and Deming Chen and Debadeepta Dey},
      year={2022},
      eprint={2210.09298},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```