import json
import os
import pickle
import subprocess
import equinox as eqx
import jax.numpy as jnp
import jax
import tiktoken
from tqdm import tqdm
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = (
    "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
)
out_dir = "out"  # ignored if init_from is not 'resume'
start = "Once upon "  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1  # number of samples to draw
max_new_tokens = 100  # number of tokens generated in each sample
temperature = (
    1  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

key = jax.random.key(1)


# model
# if init_from == "resume": #TODO add GPT2 load
# init from a model saved in a specific directory
def load(filename):
    with open(filename, "rb") as f:
        checkpoint_params = json.loads(f.readline().decode())
        gptconf = GPTConfig(**checkpoint_params["model_args"])
        return (
            eqx.tree_deserialise_leaves(
                f, GPT.create_instance(gptconf, key=jax.random.key(1))
            ),
            checkpoint_params,
        )


model, checkpoint = load(os.path.join(out_dir, "model.eqx"))
model = eqx.nn.inference_mode(model)


# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if (
    init_from == "resume"
    and "config" in checkpoint
    and "dataset" in checkpoint["config"]
):  # older checkpoints might not have these...
    meta_path = os.path.join("data", checkpoint["config"]["dataset"], "meta.pkl")
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = jnp.array([start_ids])

for k in tqdm(range(num_samples), desc="samples"):
    k, key = jax.random.split(key)

    generated = model.generate(x, max_new_tokens, temperature=temperature, key=k)
    print(decode(generated[0]))
    print("---------------")
