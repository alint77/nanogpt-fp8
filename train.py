import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import math
import torch
import torch.nn as nn
import time
import wandb
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling, Float8BlockScaling,Float8CurrentScaling,NVFP4BlockScaling,MXFP8BlockScaling
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# FSDP2 imports
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

USE_FP8 = True
USE_NVFP4 = False
USE_COMPILE_MODEL = False
USE_AMP = True
USE_FP8_WEIGHTS = False # TODO: currently not supported
USE_FSDP2 = True  
USE_DDP = False  # For clarity - only one of FSDP2, DDP should be True
USE_AC_LM_HEAD = False  # Enable activation checkpointing for lm_head
USE_AC_BLOCKS = False # Enable activation checkpointing for transformer blocks
USE_AC_ATTENTION = False  # Enable activation checkpointing for attention core (inside transformer blocks)
USE_FAST_FSDP = True  # If True, use reshard_after_forward=False to keep params gathered after forward
USE_DOC_MASK = True  # Enable document-separated attention (prevents cross-document attention)

# Wandb logging
USE_WANDB = True
WANDB_PROJECT = "nanogpt-fp8"
WANDB_RUN_NAME = None  # Set to None for auto-generated name


# hyperparameters
total_batch_size = 524288 # total tokens per batch
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 1024 # what is the maximum context length for predictions?
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50
n_layer = 12
n_embd = n_layer*64
n_head = max(1, (n_embd + 127) // 128)
dropout = 0.0
vocab_size = 50304
dataset = 'openwebtext-1M'  # name of the dataset subdirectory in ./data/
input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on

warmup_iters = 400  # Number of warmup steps
lr_decay_iters = max_iters  # Should be ~= max_iters
min_lr = 1e-8  # Minimum learning rate (10% of max LR is typical)

# ------------

check_compute_capability = te.get_device_compute_capability()

if USE_NVFP4:
    # RTX 5000 series and RTX Pro 6000 do not support rht and sr as of TE==2.9.0
    is_rtx = check_compute_capability == (12,0) 
    recipe = NVFP4BlockScaling(disable_rht=is_rtx,disable_stochastic_rounding=is_rtx)
else:
    recipe = DelayedScaling()

if USE_DDP or USE_FSDP2:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    data_parallel_group = dist.new_group(backend='nccl')
    # Initialize device mesh for FSDP2
    if USE_FSDP2:
        device_mesh = init_device_mesh("cuda", (ddp_world_size,))
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    data_parallel_group = None
    ddp_rank = 0
    ddp_local_rank = 0
    device_mesh = None

# GPU configuration for MFU calculation
GPU_PEAK_TFLOPS = 2250  # B200 BF16 peak TFLOPS per GPU
NUM_GPUS = ddp_world_size  # Number of GPUs being used

grad_accum_steps = max(1, math.ceil(total_batch_size / (batch_size * ddp_world_size*block_size)))
total_batch_size = batch_size * grad_accum_steps * ddp_world_size * block_size
print_interval = grad_accum_steps
eval_interval = grad_accum_steps * 40
max_iters = max_iters * grad_accum_steps
warmup_iters = warmup_iters // grad_accum_steps

def print0(*args, **kwargs):
    if master_process:
        print(*args, **kwargs)
print0("Transformer Engine FP8 support status:")
print0("\nFP8 Block Scaling Support:", te.is_fp8_block_scaling_available())
print0("\nMXFP8 Support:", te.is_mxfp8_available())
print0("\nNVFP4 Support:", te.is_nvfp4_available())
print0("\nDevice Compute Capability:", te.get_device_compute_capability())
print0("__________________________________")
print0("n_layer:", n_layer, "n_embd:", n_embd, "n_head:", n_head, "batch_size:", batch_size, "block_size:", block_size)
print0(f"Total_batch_size: {total_batch_size}")
print0(f"Gradient Accumulation Steps: {grad_accum_steps}")

# Initialize wandb on master process
if USE_WANDB and master_process:
    wandb_config = {
        "total_batch_size": total_batch_size,
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "n_head": n_head,
        "dropout": dropout,
        "vocab_size": vocab_size,
        "dataset": dataset,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "grad_accum_steps": grad_accum_steps,
        "use_fp8": USE_FP8,
        "use_nvfp4": USE_NVFP4,
        "use_amp": USE_AMP,
        "use_fsdp2": USE_FSDP2,
        "use_ddp": USE_DDP,
        "use_ac_lm_head": USE_AC_LM_HEAD,
        "use_ac_blocks": USE_AC_BLOCKS,
        "use_doc_mask": USE_DOC_MASK,
        "world_size": ddp_world_size,
    }
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=wandb_config,
    )
    
torch.manual_seed(1337 + seed_offset)

data_dir = os.path.join('data', dataset)

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        
        # Compute document boundaries for document-separated attention
        # EOT token (50256 for GPT-2) marks document boundaries
        cu_seqlens = None
        if USE_DOC_MASK:
            cu_seqlens = self._compute_cu_seqlens(x)
        
        return x.cuda(), y.cuda(), cu_seqlens
    
    def _compute_cu_seqlens(self, x):
        """
        Compute cumulative sequence lengths for document-separated attention.
        
        For each sequence in the batch, find EOT tokens (50256) which mark
        document boundaries. Returns cu_seqlens tensor for use with TE's
        padding_causal mask.
        
        Args:
            x: Input tensor of shape (B, T)
        
        Returns:
            cu_seqlens: Tensor of shape (B, max_seqs+1) with cumulative lengths,
                       padded with -1 for sequences with fewer documents.
                       Returns None if no EOT tokens found.
        """
        B, T = x.shape
        EOT_TOKEN = 50256  # GPT-2 EOT token
        
        # Find all EOT positions in each sequence
        # cu_seqlens format: [0, end1, end2, ..., T] for each sequence
        cu_seqlens_list = []
        max_seqs = 1  # Track max number of sequences for padding
        
        for b in range(B):
            # Find EOT positions in this sequence
            eot_positions = (x[b] == EOT_TOKEN).nonzero(as_tuple=True)[0]
            
            # Build cumulative sequence lengths
            # Start with 0, then position after each EOT, end with T
            cu_seq = [0]
            for pos in eot_positions:
                # The document ends at pos (inclusive), next starts at pos+1
                cu_seq.append(pos.item() + 1)
            # Always end with T (handles case where last doc doesn't end with EOT)
            if cu_seq[-1] != T:
                cu_seq.append(T)
            
            cu_seqlens_list.append(cu_seq)
            max_seqs = max(max_seqs, len(cu_seq))
        
        # Pad all cu_seqlens to same length and convert to tensor
        # Use T as padding value (safe since it's a valid end position)
        cu_seqlens_padded = []
        for cu_seq in cu_seqlens_list:
            padded = cu_seq + [cu_seq[-1]] * (max_seqs - len(cu_seq))
            cu_seqlens_padded.append(padded)
        
        cu_seqlens = torch.tensor(cu_seqlens_padded, dtype=torch.int32)
        return cu_seqlens.cuda()

train_loader = DistributedDataLoader(input_bin, batch_size, block_size, ddp_rank if USE_DDP or USE_FSDP2 else 0, ddp_world_size if USE_DDP or USE_FSDP2 else 1)
val_loader = DistributedDataLoader(input_val_bin, batch_size, block_size, ddp_rank if USE_DDP or USE_FSDP2 else 0, ddp_world_size if USE_DDP or USE_FSDP2 else 1)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")

@torch.no_grad()
def get_batch(split):
    """Get a batch from the appropriate dataloader."""
    if split == 'train':
        return train_loader.next_batch()
    else:
        return val_loader.next_batch()

def cu_seqlens_to_attention_mask(cu_seqlens, T, device):
    """
    Convert cumulative sequence lengths to an attention mask for document separation.
    
    This creates a block-diagonal causal mask where tokens can only attend to
    tokens within the same document (defined by cu_seqlens boundaries).
    
    Args:
        cu_seqlens: Tensor of shape (B, num_docs+1) with cumulative positions
        T: Sequence length
        device: Device to create mask on
    
    Returns:
        attention_mask: Tensor of shape (B, 1, 1, T) for use with TE attention
    """
    B = cu_seqlens.shape[0]
    # Create position indices
    positions = torch.arange(T, device=device)
    
    # For each position, find which document it belongs to
    # A position p belongs to document d if cu_seqlens[d] <= p < cu_seqlens[d+1]
    # We create a mask where True means the position should be masked (not attended to)
    
    # Simple approach: create per-sample masks
    masks = []
    for b in range(B):
        cu_seq = cu_seqlens[b]
        # Find valid boundaries (non-padded)
        valid_mask = torch.zeros(T, device=device, dtype=torch.bool)
        # Mark positions that are within document boundaries
        for i in range(len(cu_seq) - 1):
            start, end = cu_seq[i].item(), cu_seq[i + 1].item()
            if start >= end or start >= T:
                break
            valid_mask[start:min(end, T)] = True
        masks.append(~valid_mask)  # True = masked out
    
    attention_mask = torch.stack(masks).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
    return attention_mask

@torch.no_grad()
def estimate_loss():
    """
    Evaluate on validation data only (don't consume training data).
    """
    model.eval()
    val_loader.reset()
    
    loss_sum = 0.0  
    for _ in range(eval_iters):
        x, y, cu_seqlens = val_loader.next_batch()
        
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16), te.autocast(enabled=(USE_FP8 or USE_NVFP4), recipe=recipe, amax_reduction_group=data_parallel_group):
            _, loss = raw_model(x, y, cu_seqlens=cu_seqlens)
        
        loss_sum += loss.item()
    
    val_loss = loss_sum / eval_iters
    
    model.train()
    return val_loss


# Custom init functions for TE TransformerLayer (following https://arxiv.org/pdf/2310.17813)
def te_init_method(weight):
    """Init for QKV and FC1 weights in TE TransformerLayer."""
    fan_out = weight.size(0)
    fan_in = weight.size(1)
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    torch.nn.init.normal_(weight, mean=0.0, std=std)

def te_output_layer_init_method(weight):
    """Init for PROJ and FC2 weights in TE TransformerLayer - zeroed out."""
    torch.nn.init.zeros_(weight)


class LLM(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Initialize RoPE - dimension should be head_dim (hidden_size / num_heads)
        self.rope = te.RotaryPositionEmbedding(
            dim=n_embd // n_head,  # Apply RoPE to each attention head
            pretrained_max_position_embeddings=block_size
        )
        self.blocks = nn.ModuleDict({f"block_{i}": te.TransformerLayer(
            activation='srelu', 
            attention_dropout=dropout, 
            hidden_dropout=dropout, 
            hidden_size=n_embd, 
            ffn_hidden_size=4*n_embd, 
            num_attention_heads=n_head,
            attn_input_format='bshd',  # Critical: our input is (batch, seq, hidden)
            bias=False,
            qk_norm_type="RMSNorm", # L2 normalization degrades stability
            normalization="RMSNorm",
            fuse_qkv_params=True,
            seq_length=block_size,
            micro_batch_size=batch_size,
            init_method=te_init_method,
            output_layer_init_method=te_output_layer_init_method,
            # set_parallel_mode=USE_FSDP,
            # num_gqa_groups=n_head//2,
            # fuse_wgrad_accumulation=True,
            
        ) for i in range(n_layer)}) 
        self.ln_f = te.RMSNorm(n_embd) 
        self.lm_head = te.Linear(n_embd, vocab_size, bias=False)   
        # self.token_embedding_table.weight = self.lm_head.weight # tie weights
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights following Karpathy's nanochat."""
        self.apply(self._init_weights)
        # Zero out lm_head (classifier) weights
        if hasattr(self.lm_head, 'weight'):
            torch.nn.init.zeros_(self.lm_head.weight)
        # TE TransformerLayer handles its own init via init_method and output_layer_init_method
        # Cast embeddings to bf16 to save memory (optimizer can tolerate it)
        if self.token_embedding_table.weight.device.type == "cuda":
            self.token_embedding_table.to(dtype=torch.bfloat16)
    
    def _init_weights(self, module):
        """Weight init following https://arxiv.org/pdf/2310.17813"""
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
    
    def _lm_head_with_loss(self, x, targets):
        """Checkpointed lm_head + loss computation to avoid storing large logits tensor."""
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # Logit softcapping (like Gemma2) - prevents logits from growing too large
        logits = 30.0 * torch.tanh(logits / 30.0)
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)
        # Only return loss - returning logits would defeat the purpose of checkpointing
        # since they'd need to be saved for backward anyway
        return loss

    def _run_block(self, block, x, rotary_pos_emb, is_first_microbatch, attention_mask=None):
        # Use padding_causal when we have document boundaries, otherwise standard causal
        mask_type = "padding_causal" if attention_mask is not None else "causal"
        return block(x, rotary_pos_emb=rotary_pos_emb,
                        self_attn_mask_type=mask_type,
                        attention_mask=attention_mask,
                        is_first_microbatch = is_first_microbatch,
                        checkpoint_core_attention=USE_AC_ATTENTION)

    def forward(self, idx, targets=None, is_first_microbatch=False, cu_seqlens=None):
        B, T = idx.shape
        rotary_pos_emb = self.rope(T)  # Shape: (T, 1, 1, head_dim)
        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        
        # Create attention mask for document-separated attention if cu_seqlens provided
        attention_mask = None
        if cu_seqlens is not None:
            attention_mask = cu_seqlens_to_attention_mask(cu_seqlens, T, x.device)
        
        for i in range(n_layer):
            block = self.blocks[f"block_{i}"]
            if USE_AC_BLOCKS:
                x = te.checkpoint(self._run_block, block, x, rotary_pos_emb, is_first_microbatch, attention_mask, use_reentrant=True)
            else:
                x = self._run_block(block, x, rotary_pos_emb, is_first_microbatch, attention_mask)
                
        x = self.ln_f(x) # (B,T,C)

        # Apply activation checkpointing to lm_head + loss if enabled
        # This saves memory by NOT storing the large (B*T, vocab_size) logits tensor
        # Instead, we recompute lm_head during backward pass
        if USE_AC_LM_HEAD and self.training and targets is not None:
            loss = te.checkpoint(self._lm_head_with_loss, x, targets, use_reentrant=True)
            logits = None  # Don't compute logits separately during training with AC
        else:
            logits = self.lm_head(x)  # (B,T,vocab_size)
            # Logit softcapping (like Gemma2) - prevents logits from growing too large
            logits = 30.0 * torch.tanh(logits / 30.0).to(torch.bfloat16)
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

        return logits, loss
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
with te.quantized_model_init(enabled=USE_FP8_WEIGHTS, recipe=recipe):
    model = LLM().to(device)

# Calculate model parameters and FLOPs for MFU
num_params = sum(p.numel() for p in model.parameters())
print0(num_params/1e6, 'M parameters')

# FLOPs calculation for transformer:
# Forward pass: ~2 * N * T FLOPs per token (matmuls)
# Backward pass: ~4 * N * T FLOPs per token (2x forward for gradients)
# Total: ~6 * N FLOPs per token (simplified approximation)
flops_per_token = 6 * num_params
flops_per_iter = flops_per_token * (batch_size * block_size * ddp_world_size)
# Total peak FLOPs across all GPUs (in FLOPs, not TFLOPs)
total_peak_flops = GPU_PEAK_TFLOPS * 1e12 * NUM_GPUS

# Wrap with FSDP/FSDP2 BEFORE compiling
if USE_FSDP2:
    # FSDP2 uses fully_shard() - a composable API
    # Define mixed precision policy for FSDP2
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    
    # Apply fully_shard to each TransformerLayer block first (inner sharding)
    for name, block in model.blocks.items():
        fully_shard(
            block,
            mesh=device_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=(not USE_FAST_FSDP),  # Keeps params gathered after forward
        )
    fully_shard(model.lm_head,
        mesh=device_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=(not USE_FAST_FSDP),  # Keeps params gathered after forward
    )
    
    # Then apply fully_shard to the entire model (outer sharding)
    fully_shard(
        model,
        mesh=device_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=(not USE_FAST_FSDP),  # Keeps params gathered after forward
    )
    raw_model = model
    print0(f"FSDP2 enabled with {ddp_world_size} GPUs")

elif USE_DDP:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp_local_rank],
        process_group=data_parallel_group,
    )
    raw_model = model.module
else:
    raw_model = model

# Create param_groups AFTER FSDP2/DDP wrapping to get correct parameter references
hidden_weights = [p for p in raw_model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in raw_model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*raw_model.token_embedding_table.parameters(), *raw_model.ln_f.parameters()]
unembedding_params = [*raw_model.lm_head.parameters()]

param_groups = [
    dict(params=hidden_weights,lr=0.02), 
    dict(params=hidden_gains_biases+nonhidden_params, algorithm="adamw",lr=0.2, betas=(0.9, 0.95), weight_decay=0.01),
    dict(params=unembedding_params, algorithm="adamw",lr=0.004, betas=(0.9, 0.95), weight_decay=0.01)
]

from dion import Muon,NorMuon,Dion2,Dion

optimizer_muon = NorMuon(param_groups,
                        distributed_mesh=device_mesh if (USE_FSDP2) else data_parallel_group if (USE_DDP) else None,
                        use_triton=True, 
                        weight_decay=0.01,
                        cautious_wd=True,
                        )

    
T_max = (max_iters - warmup_iters)//2
# After creating optimizers:
warmup_scheduler_muon = LinearLR(optimizer_muon, start_factor=0.1, total_iters=warmup_iters)
main_scheduler_muon = CosineAnnealingLR(optimizer_muon, T_max=T_max, eta_min=min_lr)
scheduler_muon = SequentialLR(optimizer_muon, [warmup_scheduler_muon, main_scheduler_muon], milestones=[warmup_iters])

# Compile after wrapping with FSDP
if USE_COMPILE_MODEL:
    model = torch.compile(model)

# Momentum scheduler for Muon optimizer (copied from Karpathy's nanochat)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

def optimizer_step(step=None):
    
    for group in optimizer_muon.param_groups:
        group["mu"] = get_muon_momentum(step)  # NorMuon uses 'mu' not 'momentum'
    
    optimizer_muon.step()
    scheduler_muon.step()
    
def optimizer_zero_grad():
    optimizer_muon.zero_grad(set_to_none=True)

t0 = time.time()  
total_training_time = 0  # Track total time (excluding first few iterations)
grad_norm = 0.0  # Track gradient norm for logging
last_print_iter = 0  # Track which iteration we last printed at
torch.cuda.synchronize()
tlast = time.time()  # Initialize after first sync for accurate first interval

# Reset train loader before training
train_loader.reset()

for iter in range(max_iters):
    # Skip printing at iter=0 to avoid incorrect token counting (only 1 iter done, but would count as print_interval)
    should_print = (iter > 0 and iter % print_interval == 0) or iter == max_iters - 1
    
    step = iter // grad_accum_steps
       
    xb, yb, cu_seqlens = get_batch('train')
    if USE_DDP:
        model.require_backward_grad_sync = ((iter + 1) % grad_accum_steps == 0)
    
    # FSDP2 handles gradient sync automatically but we can use set_requires_gradient_sync for efficiency
    if USE_FSDP2:
        model.set_requires_gradient_sync((iter + 1) % grad_accum_steps == 0)  
        
    # FSDP handles gradient synchronization automatically
    with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
        
        with te.autocast(enabled=(USE_FP8 or USE_NVFP4), recipe=recipe, amax_reduction_group=data_parallel_group):
            _, loss = model(xb, yb, is_first_microbatch=(iter % grad_accum_steps == 0), cu_seqlens=cu_seqlens)
    
    # Scale loss for gradient accumulation
    scaled_loss = loss / grad_accum_steps
    scaled_loss.backward()
    
    if (iter + 1) % grad_accum_steps == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_step(step)
        optimizer_zero_grad()
    
    if should_print and master_process:
        # Sync GPU before timing to get accurate measurement
        torch.cuda.synchronize()
        t2 = time.time()
        dt = t2 - tlast 
        
        # Calculate actual iterations since last print (handles max_iters-1 case correctly)
        iters_since_last_print = iter - last_print_iter
        tokens_per_iteration = batch_size * block_size * iters_since_last_print * ddp_world_size
        tok_per_sec = int(tokens_per_iteration / dt)
        avg_iter_time = dt / iters_since_last_print  # Average time per iteration
        
        # Track total training time (skip first interval for warmup)
        if iter > print_interval:
            total_training_time += dt
        
        # Cache values that require GPU sync (avoid multiple .item() calls)
        loss_value = loss.detach().item()
        grad_norm_value = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
        free, total_mem = torch.cuda.mem_get_info(device)
        mem_used_GB = (total_mem - free) / 1024 ** 3
        
        # Calculate MFU
        flops_achieved = flops_per_iter * iters_since_last_print / dt
        mfu = flops_achieved / total_peak_flops * 100  # as percentage
        
        print(f"step {step}: train loss {loss_value:.4f}")
        print(f"avg iter time: {avg_iter_time*1000:.2f}ms | tok/sec: {tok_per_sec:,}")
        # print(f"lr: Muon:{scheduler_muon.get_last_lr()[0]:.6f}, AdamW Embd: {scheduler_adamw_embedding.get_last_lr()[0]:.6f}, AdamW LM_head: {scheduler_adamw_unembedding.get_last_lr()[0]:.6f}")
        print(f"total time: {total_training_time/60:.2f} min")
        print(f"MFU: {mfu:.2f}%")
        print(f"grad norm: {grad_norm_value:.4f}")
        print(f"{mem_used_GB:.2f} GB used")
        
        # Measure logging overhead
        t_log_start = time.time()
        
        # Log to wandb
        if USE_WANDB:
            wandb.log({
                "train/loss": loss_value,
                "train/grad_norm": grad_norm_value,
                "train/tokens_per_sec": tok_per_sec,
                "train/mfu": mfu,
                "train/iter_time": avg_iter_time,
                "train/total_time_min": total_training_time / 60,
                "lr/muon": scheduler_muon.get_last_lr()[0],
                # "lr/adamw_embedding": scheduler_adamw_embedding.get_last_lr()[0],
                # "lr/adamw_unembedding": scheduler_adamw_unembedding.get_last_lr()[0],
                "system/gpu_memory_gb": mem_used_GB,
            }, step=step)
        
        log_overhead = (time.time() - t_log_start) * 1000
        print_overhead = (time.time() - t2) * 1000
        print(f"[DEBUG] print+log overhead: {print_overhead:.1f}ms (wandb: {log_overhead:.1f}ms)")
        print('------------')
        
        tlast = time.time()
        last_print_iter = iter  # Update for next tokens/sec calculation
    if iter % eval_interval == 0 and iter > 0:
        # Evaluate on validation data only
        val_loss = estimate_loss()
        
        # Compute logit statistics during eval (no perf impact since we're already syncing)
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16), te.autocast(enabled=(USE_FP8 or USE_NVFP4), recipe=recipe, amax_reduction_group=data_parallel_group):
            # Pass cu_seqlens for document-separated attention if enabled
            cu_seqlens_single = cu_seqlens[:1] if cu_seqlens is not None else None
            logits, _ = raw_model(xb[:1], yb[:1], cu_seqlens=cu_seqlens_single)  # Use batch size 1 for stats
            if logits is not None:
                logit_stats = {
                    'max': logits.max().item(),
                    'min': logits.min().item(),
                    'mean': logits.mean().item(),
                    'std': logits.std().item(),
                }
            else:
                logit_stats = None
        model.train()
        
        print0('---- eval ----')
        print0(f"step {step}: val loss {val_loss:.4f}")
        if logit_stats is not None:
            print0(f"logits: max={logit_stats['max']:.2f}, min={logit_stats['min']:.2f}, mean={logit_stats['mean']:.4f}, std={logit_stats['std']:.4f}")
        print0('------------')
        
        # Log eval metrics to wandb
        if USE_WANDB and master_process:
            log_dict = {
                "eval/val_loss": val_loss,
            }
            if logit_stats is not None:
                log_dict.update({
                    "logits/max": logit_stats['max'],
                    "logits/min": logit_stats['min'],
                    "logits/mean": logit_stats['mean'],
                    "logits/std": logit_stats['std'],
                })
            wandb.log(log_dict, step=step)

# Finish wandb run
if USE_WANDB and master_process:
    wandb.finish()

# Clean up distributed
if USE_DDP or USE_FSDP2:
    destroy_process_group()
