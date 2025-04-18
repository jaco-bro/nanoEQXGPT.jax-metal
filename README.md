# nanoEQXGPT.jax-metal

This is a fork of [TugdualKerjan's nanoEQXGPT](https://github.com/TugdualKerjan/nanoEQXGPT), which is an implementation of Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) in Equinox/JAX. 

## Changes in This Fork

I've created this fork to make the project run on my M1 macbook. The modifications I made to the original repo were minimal and include:

- Replacing the `pyproject.toml` and `uv.lock` files with a `requirements.txt` file that includes [JAX-metal v0.1.0, jaxlib >=v0.4.26](https://developer.apple.com/metal/jax/), and [jax v0.5.0](https://github.com/jax-ml/jax/issues/27062#issuecomment-2726913181) because the original uv.lock contained dependencies not compatible with jax-metal.
- Replacing 'cuda' with 'mps' where applicable (`grep cuda *`)
- Replacing `jax.lax.top_k` (fails on jax-metal) with an ad-hoc workaround

## Getting Started

```bash
git clone https://github.com/jaco-bro/nanoEQXGPT.jax-metal
cd nanoEQXGPT.jax-metal
# conda create -n jax python=3.12.8 -y
# pip install -U pip
# pip install numpy wheel
# pip install jax-metal==0.1.0
pip install -r requirements.txt
python data/shakespeare_char/prepare.py
python train.py
python sample.py
```
## Original Description

An implementation of Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). The goal here is to reproduce the same GPT2 model in Equinox, a neural network library written on top of JAX. JAX allows us to use OpenXLA more effectively compared to Torch, so we should be more efficient hardware wise. We now want to make efficiency comparisons.
