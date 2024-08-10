from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
import torch.nn as nn
from transformers import GPT2LMHeadModel
import tiktoken
import time

#detect device
torch.manual_seed(1337)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"
    torch.mps.manual_seed(1337)

print(f"[+] Using device = {device}")

# this class is very similar to the gpt part 1 Head class and MultiHeadAttention class combined
class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #k,q,v projections for all heads in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularisation
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
       B,T,C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
       #calculate k,q,v for all heads in the batch and then move head forward
       # nh = number of heads, hs = head_size, C = number of channels = nh*hs
       # GPT2 (124M) n_head = 12, hs = 64 so C = 768
       qkv = self.c_attn(x)
       q,k,v = qkv.split(self.n_embd,dim=2)
       k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #B,nh,T,hs
       q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #B,nh,T,hs
       v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) #B,nh,T,hs

       att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
       att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
       att = F.softmax(att, dim=-1)

       y = att @ v # B,nh,T,T * B,nh,T,hs = B,nh,T,hs
       y = y.transpose(1,2).contiguous().view(B,T,C)

       y = self.c_proj(y)
       return y
    
# Feed forward nodes
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc =  nn.Linear(config.n_embd,4 * config.n_embd) #4 here comes from the attention is all you need paper
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50000 BPE, 256 Byte tokens, 1 <|endoftext|> token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        #weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        #initialisation parameters
        self.apply(self.__init_weights)

    def __init_weights(self,module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer)** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"[-] Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls,model_type):
        #load pre-trained gpt2 model weights from huggingface
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        print("\n[+] loading weights from the pre-trained gpt2 model from hugging face: %s" % model_type)

        #n_layer, n_head, n_embd ae taken from the model_type

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 #standard for the GPT models
        config_args['block_size'] = 1024 #standard for GPT2 models

        #ok let's create a GPT2 model from scratch using the above weights
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        #initialise the HF transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        #copy HF parameters by ensuring names and shapes match
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# data loader

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('clean_harry_potter.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        print(f"[+] Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) //(B*T)} batches")

        #state 
        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T 
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T) #inputs
        y = (buf[1:]).view(B,T) # targets

        #advance the position in the tensor
        self.current_position += B*T

        #if loading the next batch would be out of bounds,reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x,y

train_loader = DataLoaderLite(B=4,T=1024)

torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)

#model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
for i in range(50):
    t0=time.time()
    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type = device, dtype=torch.bfloat16):
        logits, loss = model(x,y)
    loss.backward()
    optimizer.step()
    torch.mps.synchronize()
    t1=time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T)/(t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/s = {tokens_per_sec:.2f}")


import sys;sys.exit(0)


#model = GPT.from_pretrained('gpt2')# this is the pre-trained exact model from openAI (hosted in HF). We used their trained model weights
model = GPT(GPTConfig()) #using our randomly initialised model
model.eval()
model.to(device)


num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
x = tokens.to(device)

torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:,-1,:] #logits at the last position
        probs = F.softmax(logits,dim=-1) # get the probabilities
        topk_probs, topk_indices = torch.topk(probs,50,dim=-1) #(top-k sampling of top 50)
        ix = torch.multinomial(topk_probs, 1) #select token from top-k probabilities
        xcol = torch.gather(topk_indices,-1,ix) #gather appropriate indices
        x = torch.cat((x,xcol),dim=1) #append to the sequence

for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print(">",decoded)