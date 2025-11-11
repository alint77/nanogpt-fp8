import glob
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support,check_mxfp8_support,check_nvfp4_support,get_device_compute_capability
from transformer_engine.common.recipe import Format, DelayedScaling, Float8BlockScaling,Float8CurrentScaling,NVFP4BlockScaling,MXFP8BlockScaling
from normuon import NorMuon, SingleDeviceNorMuon, SingleDeviceNorMuonWithAuxAdam
import bitsandbytes as bnb
import numpy as np
import os

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from dataloader import VectorizedFastDataLoader, create_dataloader



print ("Transformer Engine FP8 support status:")
print("\nFP8 Block Scaling Support:", check_fp8_block_scaling_support())
print("\nMXFP8 Support:", check_mxfp8_support())
print("\nNVFP4 Support:", check_nvfp4_support())
print("\nDevice Compute Capability:", get_device_compute_capability())

USE_FP8 = True
USE_AMP = True
USE_FP8_WEIGHTS = False
USE_CUDA_GRAPH = False # does not work currently
USE_DDP = True


# hyperparameters
# total_batch_size = 524288 # total tokens per batch
batch_size = 24 # how many independent sequences will we process in parallel?
block_size = 1024 # what is the maximum context length for predictions?
max_iters = 5000
print_interval = 20
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_layer = 12
n_embd = n_layer*64
n_head = max(1, (n_embd + 127) // 128)
dropout = 0.0
vocab_size = 50304
grad_accum_steps = 64
dataset = 'openwebtext-1M'  # name of the dataset subdirectory in ./data/
input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on

warmup_iters = 100  # Number of warmup steps
lr_decay_iters = max_iters  # Should be ~= max_iters
min_lr = 1e-4  # Minimum learning rate (10% of max LR is typical)

# ------------
print("n_layer:", n_layer, "n_embd:", n_embd, "n_head:", n_head, "batch_size:", batch_size, "block_size:", block_size)
fp8_recipe = DelayedScaling()

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


if USE_DDP:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert grad_accum_steps % ddp_world_size == 0
    grad_accum_steps //= ddp_world_size
    
    # Create process groups for FP8 amax reduction (recommended by TE docs)
    # For data parallelism only, we use the default world group
    data_parallel_group = torch.distributed.group.WORLD
    amax_reduction_group = data_parallel_group  # Synchronize FP8 scales across all GPUs
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    data_parallel_group = None
    amax_reduction_group = None

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
        return x.cuda(), y.cuda()

train_loader = DistributedDataLoader(input_bin, batch_size, block_size, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(input_val_bin, batch_size, block_size, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

@torch.no_grad()
def get_batch(split):
    """Get a batch from the appropriate dataloader."""
    if split == 'train':
        return train_loader.next_batch()
    else:
        return val_loader.next_batch()

@torch.no_grad()
def estimate_loss(reuse_input=None, reuse_target=None):
    """
    Memory-efficient evaluation that runs at batch size 1.
    Uses a slice of the provided tensors or allocates a batch-size-1 tensor.
    """
    out = {}
    model.eval()
    
    # Always use batch size 1 for evaluation to minimize VRAM
    eval_batch_size = 1
    
    # Determine if we can reuse a slice of training tensors or need to allocate
    if reuse_input is not None and reuse_target is not None:
        # Use only the first element (batch size 1) of the training tensors
        eval_input = reuse_input[:eval_batch_size]  # Shape: (1, block_size)
        eval_target = reuse_target[:eval_batch_size]  # Shape: (1, block_size)
    else:
        # Fallback: allocate batch-size-1 tensors
        eval_input = torch.empty((eval_batch_size, block_size), dtype=torch.long, device=device)
        eval_target = torch.empty((eval_batch_size, block_size), dtype=torch.long, device=device)
    
    for split in ['train', 'val']:
        loss_sum = 0.0  
        for _ in range(eval_iters):
            # Get batch data for batch size 1
            # data = train_data if split == 'train' else val_data
            # ix = torch.randint(len(data) - block_size, (eval_batch_size,))
            # idx_x = ix.unsqueeze(1) + torch.arange(block_size, device='cpu')
            # idx_y = idx_x + 1
            
            # # Copy into the batch-size-1 slice
            # eval_input.copy_(data[idx_x], non_blocking=True)
            # eval_target.copy_(data[idx_y], non_blocking=True)
            
            x, y = get_batch(split)
            eval_input.copy_(x[:eval_batch_size], non_blocking=True)
            eval_target.copy_(y[:eval_batch_size], non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe, fp8_group=amax_reduction_group):
                    _, loss = raw_model(eval_input, eval_target)
            
            loss_sum += loss.item()
        
        out[split] = loss_sum / eval_iters
    
    model.train()
    return out


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
            activation='relu', 
            attention_dropout=dropout, 
            hidden_dropout=dropout, 
            hidden_size=n_embd, 
            ffn_hidden_size=4*n_embd, 
            num_attention_heads=n_head,
            attn_input_format='bshd',  # Critical: our input is (batch, seq, hidden)
            bias=False,
            qk_norm_type='RMSNorm',
            fuse_qkv_params=True,
            seq_length=block_size,
            micro_batch_size=batch_size,
            set_parallel_mode=USE_DDP,
            # num_gqa_groups=n_head//2,
            # fuse_wgrad_accumulation=True,
            
        ) for i in range(n_layer)}) 
        self.ln_f = nn.RMSNorm(n_embd) # final layer norm
        self.lm_head = te.Linear(n_embd, vocab_size,bias=False)
        self.token_embedding_table.weight = self.lm_head.weight # tie weights
    
    def forward(self, idx, targets=None,is_first_microbatch= False):
        B, T = idx.shape
        rotary_pos_emb = self.rope(T)  # Shape: (T, 1, 1, head_dim)
        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        for i in range(n_layer):
            x = self.blocks[f"block_{i}"](x, rotary_pos_emb=rotary_pos_emb,
                                          self_attn_mask_type="causal",
                                          is_first_microbatch = is_first_microbatch)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
with te.fp8_model_init(enabled=USE_FP8_WEIGHTS, recipe=fp8_recipe):
    model = LLM().to(device)


if master_process:
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*model.token_embedding_table.parameters(),*model.lm_head.parameters(),*model.ln_f.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.0004, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]


torch.backends.fp32_precision = "tf32"
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision = "tf32"
torch.backends.cudnn.rnn.fp32_precision = "tf32"
# torch.backends.cuda.matmul.allow_tf32 = True # old api
# torch.backends.cudnn.allow_tf32 = True       # old api
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True




# optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)

optimizer_muon = NorMuon(param_groups[0]['params'],
                                    lr=param_groups[0]['lr'],
                                    weight_decay=param_groups[0]['weight_decay'])

# optimizer_muon = torch.optim.Muon(param_groups[0]['params'],
#                                   lr=param_groups[0]['lr'],
#                                   weight_decay=param_groups[0]['weight_decay'],)
optimizer_adamw = torch.optim.AdamW(param_groups[1]['params'],
                                    lr=param_groups[1]['lr'],
                                    betas=param_groups[1]['betas'],
                                    weight_decay=param_groups[1]['weight_decay'],
                                    capturable=True,
                                    # optim_bits=8,
                                    # fused=True,
                                    # foreach=True
                                    )
    
gradScaler = torch.amp.GradScaler(device='cuda', enabled=USE_AMP)

# After creating optimizers:
warmup_scheduler_muon = LinearLR(optimizer_muon, start_factor=0.1, total_iters=warmup_iters)
main_scheduler_muon = CosineAnnealingLR(optimizer_muon, T_max=max_iters - warmup_iters, eta_min=min_lr)
scheduler_muon = SequentialLR(optimizer_muon, [warmup_scheduler_muon, main_scheduler_muon], milestones=[warmup_iters])

warmup_scheduler_adamw = LinearLR(optimizer_adamw, start_factor=0.1, total_iters=warmup_iters)
main_scheduler_adamw = CosineAnnealingLR(optimizer_adamw, T_max=max_iters - warmup_iters, eta_min=min_lr)
scheduler_adamw = SequentialLR(optimizer_adamw, [warmup_scheduler_adamw, main_scheduler_adamw], milestones=[warmup_iters])


# Wrap with DDP BEFORE compiling (as per TE docs)
if USE_DDP:
    model = DDP(model, device_ids=[ddp_local_rank], process_group=data_parallel_group)
    raw_model = model.module
else:
    raw_model = model

# Compile after wrapping with DDP
# model = torch.compile(model, mode='max-autotune-no-cudagraphs')

# @torch.compile()
def gradscaler_step_adamw():
    gradScaler.step(optimizer_adamw)
    
def gradscaler_step():
    gradScaler.step(optimizer_muon)
    gradscaler_step_adamw()
    scheduler_muon.step()
    scheduler_adamw.step()
    
def optimizer_zero_grad():
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()

def gradscaler_unscale():
    gradScaler.unscale_(optimizer_muon)
    gradScaler.unscale_(optimizer_adamw)


# if USE_CUDA_GRAPH:
    
#     static_input = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
#     static_target = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    
#     print("Warming up for CUDA graph capture...")
#     s = torch.cuda.Stream()
#     s.wait_stream(torch.cuda.current_stream())
#     with torch.cuda.stream(s):
#         for i in range(3): # a few iterations to warm up
            
#             xb, yb = get_batch('train')
#             static_input.copy_(xb)
#             static_target.copy_(yb)
            
#             optimizer_zero_grad()
            
#             with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
#                 with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
#                     _, loss = model(static_input, static_target)
            
#             gradScaler.scale(loss/grad_accum_steps).backward()
#     torch.cuda.current_stream().wait_stream(s)
    
#     print("Capturing CUDA graph...")
#     g = torch.cuda.CUDAGraph()
    
#     optimizer_zero_grad()
    
#     with torch.cuda.graph(g):
#         with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
#             with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
#                 _, static_loss = model(static_input, static_target)
#         gradScaler.scale(static_loss/grad_accum_steps).backward()
#         # Note: optimizer steps are NOT captured - they run eagerly
    
#     print("CUDA graph captured successfully!")
#     print("=" * 60)

t0 = time.time()  
total_training_time = 0  # Track total time (excluding first few iterations)
tlast = t0
grad_norm = 0.0  # Track gradient norm for logging
raw_model = model.module if USE_DDP else model
for iter in range(max_iters):
    t1 = time.time()
    should_print = iter % print_interval == 0 or iter == max_iters - 1
    
    if USE_CUDA_GRAPH:
        pass
        
    #     xb, yb = get_batch('train')
    #     static_input.copy_(xb)
    #     static_target.copy_(yb)
        
    #     g.replay()
        
        
    #     if (iter + 1) % grad_accum_steps == 0:
    #         # Unscale gradients before clipping
    #         gradscaler_unscale()
            
    #         # Get gradient norm before clipping
    #         grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)

    #         gradscaler_step()
    #         gradScaler.update()
            
    #         optimizer_zero_grad()   
    else:    
        xb, yb = get_batch('train')

        # Control gradient synchronization for gradient accumulation
        if USE_DDP: 
            model.require_backward_grad_sync = ((iter + 1) % grad_accum_steps == 0)       
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
            # Use fp8_group for proper FP8 scaling factor synchronization across GPUs (TE best practice)
            with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe, fp8_group=amax_reduction_group):
                _, loss = model(xb, yb, is_first_microbatch=(iter % grad_accum_steps == 0))
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps
        gradScaler.scale(scaled_loss).backward()
        
        if (iter + 1) % grad_accum_steps == 0:
            # Unscale gradients before clipping
            gradscaler_unscale()
            
            # Get gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            gradscaler_step()
            gradScaler.update()
            
            optimizer_zero_grad()
    
    
    # Track total training time (skip first 10 iterations for warmup)
    if iter > 10 and master_process:
        dt = time.time() - t1
        total_training_time += dt
    
    if should_print and master_process:
        t2 = time.time()
        dt = t2 - tlast 
        
        tokens_per_iteration = batch_size * block_size * print_interval * ddp_world_size
        tok_per_sec = int(tokens_per_iteration / dt)
        
        print(f"step {iter}: train loss {loss.detach():.4f}")
        print(f"iter time: {dt:.4f}s | tok/sec: {tok_per_sec:,}")
        print(f"lr: Muon:{scheduler_muon.get_last_lr()[0]:.6f}, AdamW: {scheduler_adamw.get_last_lr()[0]:.6f}")  # Add this line
        print(f"total time: {total_training_time/60:.2f} min")
        free, total = torch.cuda.mem_get_info(device)
        mem_used_GB = (total - free) / 1024 ** 3
        print(f"grad norm: {grad_norm:.4f}")
        print(f"{mem_used_GB:.2f} GB used")
        print('------------')
        tlast = time.time()
    if iter % eval_interval == 0 and iter > 0 and master_process:
        # Reuse training batch tensors for evaluation to save VRAM
        if USE_CUDA_GRAPH:
            # losses = estimate_loss(reuse_input=static_input, reuse_target=static_target)
            pass
        else:
            losses = estimate_loss(reuse_input=xb, reuse_target=yb)
        print('---- eval ----')
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print('------------')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
if master_process:
    print(decode(raw_model.generate(context, max_new_tokens=2000)[0].tolist()))

# Clean up DDP
if USE_DDP:
    destroy_process_group()
