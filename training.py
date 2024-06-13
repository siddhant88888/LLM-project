import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse

parser = argparse.ArgumentParser(description='This is a demo program.')

parser.add_argument('-batch_size', type = str, required= True, help='Please provide a batch size')
args = parser.parse_args()

print(f'batch size: {args.batch_size}')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
batch_size = args.batch_size
block_size = 32

learning_rate = 3e-4
max_iters = 200
eval_iters = 100
eval_interval = 500
n_embd = 384
n_head = 1
n_layer = 1
# This will turn 20% of neurons to zero.
dropout = 0.2

# embedding_vector = [0.1, 0.2, 0.8, 1.1,.......n_embd values(i.e. 384)]
# Every value in the embedding vector will store some information about the 
# word that is being encoded.

chars = ''
with open('openwebtext/vocab.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
vocab_size = len(chars)

string_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_string = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

import mmap
import random


def get_random_chunk(split):
    filename = 'openwebtext/train_split.txt' if split == 'train' else 'openwebtext/val_split.txt'
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
                
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            mm.seek(start_pos)
            block = mm.read(block_size * batch_size-1)

            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r', '')

            data = torch.tensor(encode(decoded_block), dtype = torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)
    # This generates random indexes to pull 8 elements from.
    ix = torch.randint(len(data) - block_size, (batch_size,))
   
    # take tensors from 30233 to 30241(30233+8) and do this for
    # every tensor in ix.
    x = torch.stack([data[i:i+block_size] for i in ix])
    #take tensors from 30234(30233+1) to 30242(30233+8+1) and do this 
    # for every tensor in ix.
    y = torch.stack([data[i+1:i+block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

x,y = get_batch('train')
print(x.device)
print(y.device)
print('inputs:')
print(x)
print('targets:')
print(y)

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

class Head(nn.Module):
    
    def __init__(self, head_size):
        super().__init__()
        # Here k,q and v are calculated to later be scaled and dot product-ed.
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout= nn.Dropout(dropout)
        
    def forward(self, x):
            # What is x?
            # 'x' is the input(A single word/token from a training text) 
            # It is represented by a vector of size 384 == n_embd
            # This is why, this code -> 'nn.Linear(n_embd, head_size, bias = False)' works
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
            # wei == attention weights/scores.
            
        wei = q @ k.transpose(-2, -1) * k.shape[-1]** - 0.5
            # This code multiplies the wei with the masking matrix (lower triangular matrix).
            # This multiplication helps set the wei of words that come after the current position(x^i) to zero.
            # Effectively forcing the model to use only the context it can gather from the word at the current position
            # and the words can come before it.
            
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            # Softmax normalizes the wei's -> It exagerates the difference of the wei's from one another.
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
            
        v = self.value(x)
        out = wei @ v 
        return out
        

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            # Any number below zero is converted to 0 and any number above 
            # stays the same.
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # This makes a certain percentage of neurons dropout.
            # This is done to prevent overfitting.
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x+y)
        y = self.ffwd(x)
        x = self.ln2(x+y)
        return x
        
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Token embedding uses vocab size to give 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # This clever little piece of code passes one block at a time to the nn.Sequential layers.
        # The '*' symbol is responsible for this unpacking.
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        # Every element in n_embd is normalized, scaled(y= Gamma) and shifted(Beta) one at a time.
        # That's what LayerNorm does.
        self.ln_f = nn.LayerNorm(n_embd)
        # Pass n_embd as input to linear layer and produce an outputs = len(vocab_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
                
        
    def forward(self, index, targets=None):
        B, T = index.shape
        
        
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
           
        return index

model =GPTLanguageModel(vocab_size)
print('loading model parameters...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully...')
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f'step: {iter},train loss: {losses["train"]:.3f}, val loss: {losses["val"]:.3f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss

    logits, loss = model.forward(xb, yb)
   
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')

   








