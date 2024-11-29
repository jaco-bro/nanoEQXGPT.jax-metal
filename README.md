# nanoEQXGPT

An implementation of Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). The goal here is to reproduce the same GPT2 model in Equinox, a neural network library written on top of JAX. JAX allows us to use OpenXLA more effectively compared to Torch, so we should be more efficient hardware wise. We now want to make efficiency comparisons.

# install

    pip install wandb jax numpy equinox optax tqdm tiktoken

# roadmap

- [ ] Compare speed to nanoGPT in torch
- [ ] provide checkpoints for people to test.
- [ ] fix download datasets issuse
- [ ] fix scaling in the train
- [ ] implement multidevice train
- [ ] mixed precision
- [ ] model surgery if it's greater than block_size
- [ ] profile code to avoid wasted time (mfu goes brr)
- [ ] microbatching in JAX -> does it even make sense 
- [ ] loading the optax state from the correct position# nanoEQXGPT
- [ ] convert to bfloat32 possible
- [ ] Check if this is useful: os.environ["XLA_FLAGS"] = "--xla_gpu_enable_tf32=true" 