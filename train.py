import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import check_fp8_block_scaling_support,check_mxfp8_support,check_nvfp4_support,get_device_compute_capability
from transformer_engine.common.recipe import Format, DelayedScaling, Float8BlockScaling,Float8CurrentScaling,NVFP4BlockScaling
from normuon import SingleDeviceNorMuonWithAuxAdam
import bitsandbytes as bnb

print ("Transformer Engine FP8 support status:")
print("\nFP8 Block Scaling Support:", check_fp8_block_scaling_support())
print("\nMXFP8 Support:", check_mxfp8_support())
print("\nNVFP4 Support:", check_nvfp4_support())
print("\nDevice Compute Capability:", get_device_compute_capability())

# hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 512 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 512
n_head = 16
n_layer = 16
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = 128
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
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Fix: specify attn_input_format='bshd' to match our batch-first input tensors
        # This ensures causal masking is applied correctly
        self.blocks = nn.ModuleDict({f"block_{i}": te.TransformerLayer(
            activation='relu', 
            attention_dropout=dropout, 
            hidden_dropout=dropout, 
            hidden_size=n_embd, 
            ffn_hidden_size=4*n_embd, 
            num_attention_heads=n_head,
            attn_input_format='bshd',  # Critical: our input is (batch, seq, hidden)
            layer_number=i+1
        ) for i in range(n_layer)})
        self.ln_f = te.LayerNorm(n_embd) # final layer norm
        self.lm_head = te.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for i in range(n_layer):
            x = self.blocks[f"block_{i}"](x, self_attn_mask_type="causal")
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

model = BigramLanguageModel()
model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

hidden_weights = [p for p in model.blocks.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.blocks.parameters() if p.ndim < 2]
nonhidden_params = [*model.position_embedding_table.parameters(), *model.token_embedding_table.parameters(), *model.lm_head.parameters()]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.01, weight_decay=0.01),
    dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
torch.backends.cuda.matmul.fp32_precision = 'ieee'
torch.backends.cudnn.conv.fp32_precision = 'tf32'
# optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
fp8_recipe = Float8CurrentScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
t0 = time.time()
for iter in range(max_iters):
    t1 = time.time()
    should_print = iter % eval_interval == 0 or iter == max_iters - 1
    # every once in a while evaluate the loss on train and val sets
    if should_print:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # sample a batch of data
    xb, yb = get_batch('train')
    
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # evaluate the loss
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if should_print:
        t2 = time.time()
        print(f"iter time: {(t2 - t1):.2f}s")
        print(f"total time: {(t2 - t0)/60:.2f} min")
        free, total = torch.cuda.mem_get_info(device)
        mem_used_GB = (total - free) / 1024 ** 3
        print(f"{mem_used_GB:.2f} GB used")
        print('------------')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
