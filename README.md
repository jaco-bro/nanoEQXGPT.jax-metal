# nanoEQXGPT

An implementation of Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master). The goal here is to reproduce the same GPT2 model in Equinox, a neural network library written on top of JAX. JAX allows us to use OpenXLA more effectively compared to Torch, so we should be more efficient hardware wise. We now want to make efficiency comparisons.

# notable differences with the nanoGPT version

### datasets

[Tinystories](https://arxiv.org/abs/2305.07759) is added 

### config

`out_dir` is replaced with `out_path` which allows avoids hardcoding the model name saved and loaded.
`tensorboard_log` is available and `wandb_project` and `wandb_run_name` are changed to `log_project` and `log_run_name` respectively.

## Roadmap 🚎

- [x] Compare speed to nanoGPT in torch
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

## Getting started

```bash
git clone git@github.com:TugdualKerjan/nanoEQXGPT.git
uv sync
uv run data/shakespear_char/prepare.py
uv run train.py
```

## Speed 

It seems like kaparthy has spent more time than me on optimization because the model here is about x10 slower that the PyTorch version lol (Around 300ms vs 30ms) for the shakespear_char dataset.