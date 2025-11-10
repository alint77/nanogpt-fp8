import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support,check_mxfp8_support,check_nvfp4_support,get_device_compute_capability
from transformer_engine.common.recipe import Format, DelayedScaling, Float8BlockScaling,Float8CurrentScaling,NVFP4BlockScaling,MXFP8BlockScaling
from normuon import SingleDeviceNorMuon, SingleDeviceNorMuonWithAuxAdam
import bitsandbytes as bnb
import numpy as np

print ("Transformer Engine FP8 support status:")
print("\nFP8 Block Scaling Support:", check_fp8_block_scaling_support())
print("\nMXFP8 Support:", check_mxfp8_support())
print("\nNVFP4 Support:", check_nvfp4_support())
print("\nDevice Compute Capability:", get_device_compute_capability())

USE_FP8 = True
USE_AMP = True
USE_FP8_WEIGHTS = False

# hyperparameters
batch_size = 6 # how many independent sequences will we process in parallel?
block_size = 2048 # what is the maximum context length for predictions?
max_iters = 5000
print_interval = 1
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_layer = 20
n_embd = n_layer*64
n_head = max(1, (n_embd + 127) // 128)
dropout = 0.0
# ------------
print("n_layer:", n_layer, "n_embd:", n_embd, "n_head:", n_head, "batch_size:", batch_size, "block_size:", block_size)
torch.manual_seed(1337)
fp8_recipe = DelayedScaling(disable_rht=True)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = 65536
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


# data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y
@torch.no_grad()
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Vectorized indexing - much faster than list comprehension
    idx_x = ix.unsqueeze(1) + torch.arange(block_size, device='cpu')
    idx_y = idx_x + 1
    x = data[idx_x]  # Fancy indexing: (batch_size, block_size)
    y = data[idx_y]
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)  # Keep on GPU
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda',enabled=USE_AMP, dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):  # Enable FP8!
                    logits, loss = model(X, Y)
            losses[k] = loss  # Keep on GPU, don't call .item() yet
        out[split] = losses.mean().item()  # Single CPU-GPU sync at the end
    model.train()
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):

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
            layer_number=i+1,
            bias=False,
            qk_norm_type='RMSNorm',
            fuse_qkv_params=True,
            seq_length=block_size,
            micro_batch_size=batch_size,
            
        ) for i in range(n_layer)}) 
        self.ln_f = te.LayerNorm(n_embd) # final layer norm
        self.lm_head = te.Linear(n_embd, vocab_size,bias=False)
        # self.token_embedding_table.weight = self.lm_head.weight # tie weights
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        rotary_pos_emb = self.rope(T)  # Shape: (T, 1, 1, head_dim)
        # idx and targets are both (B,T) tensor of integers
        x = self.token_embedding_table(idx) # (B,T,C)
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        for i in range(n_layer):
            x = self.blocks[f"block_{i}"](x, rotary_pos_emb=rotary_pos_emb,
                                          self_attn_mask_type="causal")
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
# with torch.amp.autocast('cuda',enabled=USE_AMP, dtype=torch.bfloat16):
with te.fp8_model_init(enabled=USE_FP8_WEIGHTS, recipe=fp8_recipe):
        model = BigramLanguageModel().to(device)

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
torch.backends.cuda.matmul.fp32_precision = 'tf32'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


# optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
# optimizer_muon = SingleDeviceNorMuon(param_groups[0]['params'],
#                                   lr=param_groups[0]['lr'],
#                                   weight_decay=param_groups[0]['weight_decay'])

optimizer_muon = torch.optim.Muon(param_groups[0]['params'],
                                  lr=param_groups[0]['lr'],
                                  weight_decay=param_groups[0]['weight_decay'],)
optimizer_adamw = bnb.optim.adamw.AdamW(param_groups[1]['params'],
                                    lr=param_groups[1]['lr'],
                                    betas=param_groups[1]['betas'],
                                    weight_decay=param_groups[1]['weight_decay'],
                                    optim_bits=8)

gradScaler = torch.amp.GradScaler(device='cuda', enabled=USE_AMP)

# CUDA Graph setup
USE_CUDA_GRAPH = True
print(f"\nCUDA Graphs enabled: {USE_CUDA_GRAPH}")

if USE_CUDA_GRAPH:
    # Create static tensors for graph capture
    static_input = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    static_target = torch.empty((batch_size, block_size), dtype=torch.long, device=device)
    
    # Warmup: must run on a side stream before capture
    print("Warming up for CUDA graph capture...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            # Get real data for warmup
            xb, yb = get_batch('train')
            static_input.copy_(xb)
            static_target.copy_(yb)
            
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
                    logits, loss = model(static_input, static_target)
            
            gradScaler.scale(loss).backward()
    torch.cuda.current_stream().wait_stream(s)
    
    # Capture the graph
    print("Capturing CUDA graph...")
    g = torch.cuda.CUDAGraph()
    
    # Zero grads before capture with set_to_none=True
    # so backward() will create .grad attributes from the graph's private pool
    optimizer_muon.zero_grad(set_to_none=True)
    optimizer_adamw.zero_grad(set_to_none=True)
    
    with torch.cuda.graph(g):
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
                static_logits, static_loss = model(static_input, static_target)
        gradScaler.scale(static_loss).backward()
        # Note: optimizer steps are NOT captured - they run eagerly
    
    print("CUDA graph captured successfully!")
    print("=" * 60)

t0 = time.time()  # Overall start time
total_training_time = 0  # Track total time (excluding first few iterations)

for iter in range(max_iters):
    t1 = time.time()
    should_print = iter % print_interval == 0 or iter == max_iters - 1
    
    if USE_CUDA_GRAPH:
        # Fill graph's input memory with new batch data
        xb, yb = get_batch('train')
        static_input.copy_(xb)
        static_target.copy_(yb)
        
        # Replay the graph (forward + backward)
        g.replay()
        
        # Run optimizer steps eagerly (not part of the graph)
        gradScaler.step(optimizer_muon)
        gradScaler.step(optimizer_adamw)
        gradScaler.update()
        
        # Use static_loss for logging
        loss = static_loss
    else:
        # Original non-graph path
        optimizer_muon.zero_grad()
        optimizer_adamw.zero_grad()
        xb, yb = get_batch('train')
        
        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.bfloat16):
            with te.fp8_autocast(enabled=USE_FP8, fp8_recipe=fp8_recipe):
                logits, loss = model(xb, yb)
        
        gradScaler.scale(loss).backward()
        gradScaler.step(optimizer_muon)
        gradScaler.step(optimizer_adamw)
        gradScaler.update()
    
    t2 = time.time()
    dt = t2 - t1  
    
    tokens_per_iteration = batch_size * block_size
    tok_per_sec = int(tokens_per_iteration / dt)
    
    # Track total training time (skip first 10 iterations for warmup)
    if iter > 10:
        total_training_time += dt
    
    if should_print:
        print(f"step {iter}: train loss {loss.detach():.4f}")
        print(f"iter time: {dt:.4f}s | tok/sec: {tok_per_sec:,}")
        print(f"total time: {total_training_time/60:.2f} min")
        free, total = torch.cuda.mem_get_info(device)
        mem_used_GB = (total - free) / 1024 ** 3
        print(f"{mem_used_GB:.2f} GB used")
        print('------------')
    if iter % eval_interval == 0 and iter > 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
