# nanoEQXGPT

An implementation of Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). The goal here is to reproduce the same GPT2 model in Equinox, a neural network library written on top of JAX. JAX allows us to use OpenXLA more effectively compared to Torch, so we should be more efficient hardware wise. We now want to make efficiency comparisons.

# install

    `conda env create -f environment.yml`
    `conda activate gpt`
    `pip install jax` or `pip install `jax["cuda"]` depending on if you are running CPU or GPU under the hood.

# notable differences with the nanoGPT version

### datasets

[Tinystories](https://arxiv.org/abs/2305.07759) is added 

### config

`out_dir` is replaced with `out_path` which allows avoids hardcoding the model name saved and loaded.
`tensorboard_log` is available and `wandb_project` and `wandb_run_name` are changed to `log_project` and `log_run_name` respectively.

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
