# nanogpt-fp8

An educational project exploring FP8 training with NVIDIA Transformer Engine. This is a modified version of nanoGPT that uses Transformer Engine layers for FP8/NVFP4 training, with architecture based on nanochat (561M parameters).

## What This Is

This project started as Karpathy's nanoGPT and evolved through experimentation with modern training techniques. The goal is to understand and test FP8 training, distributed training strategies, and various optimizations on newer NVIDIA Blackwell GPUs (RTX 5090, RTX Pro 6000, B200).

The codebase uses the FineWeb10B dataset following the modded-nanogpt approach by kellerjordan.

## Features

The model uses NVIDIA Transformer Engine with FP8 or NVFP4 quantization. Training is distributed via FSDP2 (or DDP as an option but FSDP2 is faster and uses less vram) with the Muon optimizer from Microsoft's Dion library for hidden layer weights and AdamW for embeddings.

Architecture includes RoPE positional embeddings, RMSNorm, QK normalization, logit softcapping (like Gemma2), and squared ReLU activation. Weight initialization follows the approach from nanochat

Other features include optional activation checkpointing for the LM head to save vram, cosine LR schedule with warmup, Muon momentum scheduling, MFU calculation (hardcoded for B200), and wandb logging.

## Development History

Started from a basic transformer training on tinyshakespeare. Added DDP for multi-GPU support, then gradient accumulation and grad norm logging. Experimented with CUDA graphs and compiled optimizer. Added AMP and tested torch native Muon.

Added FP8 support with compute capability checks. Got NVFP4 working. Switched to nanochat architecture (~500M params). Added fineweb dataset and distributed dataloader. Implemented FSDP, then upgraded to FSDP2. Integrated Muon from Dion with FSDP2 compatibility.

Added wandb logging and MFU tracking. Separated optimizers for different parameter groups. Implemented activation checkpointing for LM head to save VRAM. Various fixes for QK normalization (settled on RMSNorm - L2 hurts convergence), dataloader, and timing calculations.

## Requirements

- PyTorch with CUDA
- NVIDIA Transformer Engine (2.9+)
- wandb
- Dion (Microsoft's optimized Muon implementation with FSDP support)

## Usage

Set up the environment and deps:

for cuda 12:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

for cuda 13:
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install transformer-engine-torch 
uv pip install --reinstall transformer_engine[core_cu13]
uv pip install -r requirements_no_torch_te.txt
```

Download the dataset (number is how many 100M token chunks to download, max 103) this is what modded-nanogpt uses:
```bash
python data/cachedfineweb10b.py 9
```

Edit [run.sh](run.sh) to set the number of GPUs (`--nproc-per-node`), then:
```bash
bash run.sh
```

## Configuration

Main settings in [train.py](train.py):

- `USE_FP8` / `USE_NVFP4` - quantization mode
- `USE_FSDP2` / `USE_DDP` - distributed strategy
- `USE_AC_LM_HEAD` - activation checkpointing for LM head
- `total_batch_size`, `batch_size`, `block_size` - batch configuration
- `n_layer`, `n_embd`, `n_head` - model size (defaults: 20 layers, 1280 hidden, 10 heads)

## Results

Training on 8x RTX Pro 6000 with batch_size=16 reached val loss <3.28 in 36.5 minutes.

## Known Issues and TODOs:

**PyTorch dispatches sm89 kernels on sm120 GPUs.** Found this with the profiler. To use sm120 fully, you need PyTorch compiled with sm120 support which takes hours to build from source, I haven't tried it yet but I've seen people claim up to 40% speedups...

**No model checkpointing or saving.** Training state and model weights are not saved. this repo is just a benchmark at this stage. 

**No inference code.** The generate method exists but I haven't tested it, also there's no standalone inference script or loading from checkpoints.

**Activation Dtypes.** comparison with nanochat in terms of vram usage shows activation dtypes may be different between this repo and nanochat (bf16), even though amp with bf16 is used. here's some test results:

8xRTX 5090 32GB - total_BS = 524288 - FSDP2 (reshard_after_forward=False):

BS=1 nanochat: 14.27GB 273K tps - this: 9.770GB 144K tps
BS=4 nanochat: 20.34GB 333K tps - this: 21.27GB 360K tps
BS=6 nanochat: 24.47GB 355K tps - this: 30.39GB 431K tps (total_BS = 589824 for both)
BS=8 nanochat: 28.48GB 348K tps - this: OOM

as you can see the mem usage grows more rapidly with BS in this repo.

**Logit Softcapping is too expensive.** causes ~15% tps slowdown and uses precious vram, maybe there's a better alternative?

**Param Count is slightly higher than nanochat.** I haven't been able to figure out why yet, I have tried my best to match the arch but there's still param count mismatch: 

561,045,760 (this) vs 560,988,160 (nanochat)

## Files

- [train.py](train.py) - main training script
- [train_orig.py](train_orig.py) - original nanoGPT for reference
- [run.sh](run.sh) - launch script for torchrun
- [data/cachedfineweb10b.py](data/cachedfineweb10b.py) - downloads FineWeb10B dataset
- [data/openwebtext-1M/](data/openwebtext-1M/) and [data/openwebtext-100k/](data/openwebtext-100k/) - alternative smaller datasets
