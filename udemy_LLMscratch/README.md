## LLM Mastery: Hands-on Code, Align and Master LLMs
- Instructor: Javier Ideami

## Section 1: Introduction to Generative AI

### 1. Welcome to the course

### 2. Why we will start by introducing Generative AI concepts

### 3. Introducing myself

### 4. Generative modelling, Evolution of Generative AI and Overview of applications

### 5. Building Blocks of Machine Creativity: Machine Learning Foundations for Gen AI
- ReLU vs leaky ReLU

### 6. Architectures of Machine Imagination, from GANs to Diffusion and beyond
- 1000x1000 pixels: 1 M dimensions
  - How to fill 1M factors?
  - Probability distribution of images of human faces
- Generative models
  - Generative Adversary Network
    - Very hard to train
    - Hard to tweak hyperparameters
  - Autoregressive
    - Suitable for text generation
    - Needs enormous data to train
  - Variational Autoencoder (VAE)
    - Good for generating images
    - More stable than GAN
    - Sharpness is worse than GAN
  - Diffusion
    - Text to image/video
  - Flow
    - Precise data modeling
    - Voice synthesis
    - Computationally expensive and hard to scale
- GANs
  - Discriminator: real/fake -> find fake/real
  - Generator: noise -> fake
  - They are competing each other
- Autoregressive
  - Multiple transformer layers
  - Learn output the next token in a data sequence
- VAE
  - Encoder: tweak probability distribution
  - Decoder
- Diffusion
  - Diffusion (adding noise)
  - Denoising (reversing noise)
- Flow
  - Transformations
  - Inverse transformations

### 7. Machine Creativity Meets Real-World Impact - Applications of Generative AI

### 8. The Ethics of Machine Creativity: Challenges and Considerations in Generative AI

### 9. Worlds Reimagined: Visions of the Future with Generative AI

### 10. Summary and closing thoughts about this intro of GenAI

## Section 2: Coding a small LLM from scratch, understanding all the key concepts involved 

### 11. Welcome to this section

### 12. Where to do the coding - Intro

### 13. Where to do the coding - Details

### 14. Dealing with challenges, and reminder about coding options

### 15. Setting up the coding environment
- pip install ipdb tqdm sentencepiece wandb

### 16. How Jupyter Notebooks work

### 17. Importing the necessary libraries
```py
import os,sys
import ipdb
from tqdm import tqdm
from datetime import datetime
import platform, shutil
import requests, zipfile, io
import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
#
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
```

### 18. Setting up our base files
```py
files_url="https:/ideami.com/llm_train"
print("Dowloading files using Python")
response = requests.get(files_url)
zipfile.ZiFile(io.BytesIO(response.content)).extractall(".")
```
- Not working

### 19. Setting up the parameters of the architecture
```py
# Architecture parameers
batch_size = 4 # 8 for 4GB GPU memory
context = 512 # limits of the sequence of tokens
embed_size = 384 # dim of embedding
n_layers = 7
n_heads = 7
BIAS = True # allows the activation function to shift
```

### 20. Exploring the crucial hyperparameters
- Gradient vanishing: without gradient information the network cannot learn
- Gradient explosing: too large gradients cause instability in the network computations
```py
# hyperparameters
lr = 3e-4
dropout = 0.05 # regularization - preventing overfitting - randomly turning off a fraction of neurons
weight_decay = 0.01 # or L2 regularization. Adds a penalty to the loss function
grad_clip = 1.0 # prevents exploding gradients 
```

### 21. Key parameters for an effective training process
```py
# training parameters
train_iters = 100000
eval_interval = 50
eval_iters = 3
compile = False
checkpoint_dir = 'models/'
checkpoint_fn = 'latest.pt'
checkpoint_load_fn = 'latest.pt'
dtype = torch.bfloat16
# mode
inference = False
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 22. Introducing Logging
- wandb.ai

### 23. Setting up logging

### 24. Setting up the tokenizer and related functionality
```py
with open("zip/wiki.txt", 'r', encoding = 'utf-8') as f:
  text = f.read()
print(text[10000:10300])
# tokenizer
sp = spm.SentencePieceProcessor(model_file='zip/wiki_tokenizer.model')
vocab_size = sp.get_piece_size()
print(f"Tokenizer vocab_size: {vocab_size} ") # 4096
#
encode = lambda s: sp.Encode(s)
decode = lambda l: sp.Decode(l)
print(encode("Once upon a time")) # [612, 370, 698, 265, 261, 684]
print(decode(encode("Once upon a time"))) # Once upon a time
##
if os.path.exists(f"zip/encoded_data.pt"):
  data = torch.load('zip/encoded_data.pt')
else:
  data = torch.tensor(encode(text),dtype=torch.long)
  torch.save(data, 'zip/encoded_data.pt')
```

### 25. Splitting our data and creating our get batch function
```py
data_size = len(data)
spl = int(0.9*data_size)
train_data= data[:spl]
val_data=data[spl:]
print(f"Total data: {data_size/1e6:.2f} Million | training {len(train_data)/1e6:2f} Million | validation: {len(val_data)/1e6:2f} Million") # 59.21 M, 53.3 M, 5.92 M
#
def get_batch(split):
  # BS = batch size (4 above)/ SL = sequence or context length
  data = train_data if split=="train" else val_data
  inds = torch.randint(len(data)-context, (batch_size,))
  x = torch.stack([data[i:i+context] for i in inds]) # (BS, SL) = (4,512)
  y = torch.stack([data[i+1:i+context+1] for i in inds]) # predicts the next token
  x,y = x.to(device), y.to(device)
  return x,y
x,y= get_batch("train")
print(x.shape, y.shape) # torch.Size([4, 512]) torch.Size([4, 512])
print(x[0][2:5]) # tensor([1688,  280, 3093], device='cuda:0')
```

### 26. The Transformer Architecture
- https://github.com/javismiles/X-Ray-Transformer
- https://medium.com/data-science/x-ray-transformer-dive-into-transformers-training-inference-computations-through-a-single-visual-4e8d50667378

### 27. Declaring the top layers of the LLM
```py
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size,embed_size) # 4096x384
    self.positions = nn.Embedding(context, embed_size) # seq. info, 512x384
    self.blocks = nn.Sequential( *[Block(n_heads) for _ in range(n_layers)]) # * is unpacking operator
    self.ln = nn.LayerNorm(embed_size) # layer normalization
    self.final_linear = nn.linear(embed_size,vocab_size,bias=BIAS) # 384x4096
```

### 28. The forward function of the LLM

### 29. The Cross Entropy Loss with Pytorch

### 30. The Cross Entropy Loss recreated manually
- The entropy is the negative sum of the product of each of the probabilities by the log of that probability

### 31. From Information to Cross-Entropy - Deep Dive

### 32. Completing and verifying the manual cross entropy loss
```py
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size,embed_size) # 4096x384
    self.positions = nn.Embedding(context, embed_size) # seq. info, 512x384
    # TBD
    #self.blocks = nn.Sequential( *[Block(n_heads) for _ in range(n_layers)]) # * is unpacking operator
    self.ln = nn.LayerNorm(embed_size) # layer normalization
    self.final_linear = nn.Linear(embed_size,vocab_size,bias=BIAS) # 384x4096
    self.apply(self._init_weights)
  # parameter initialization
  def _init_weights(self,module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  def forward(self, input, targets=None):
    loss = None
    BS, SL = input.shape # 4x512
    emb = self.embeddings(input) # 4x512x384
    pos = self.positions(torch.arange(SL,device=device))
    x = emb + pos # BS x SL x 384
    # TBD
    #x = self.blocks(x) # BS x SL x 384
    x = self.ln(x) # BS x SL x 384 
    logits = self.final_linear(x) # BS x SL x 4096 (vocab_size)
    if targets is not None:
      BS, SL, VS = logits.shape
      logits = logits.view(BS*SL, VS)
      targets = targets.view(BS*SL)
      loss = F.cross_entropy(logits, targets)
      # manual calculation (softmax)
      counts = logits.exp()
      prob = counts /counts.sum(-1, keepdim=True)
      loss2 = - prob[torch.arange(BS*SL), targets].log().mean()
      # target[3] = 329, prob[3][329]= 0.xxx
      # cross entropy = - log p(x)
    return logits, loss, loss2
x,y = get_batch("train")
model = GPT()
model = model.to(dtype)
model = model.to(device)
logits, loss, loss2 = model(x,y)
print(loss.item(), loss2.item()) # 8.375 8.375 - very bad!!!
```
- Block will be implemented later

### 33. Generating new samples - Intro

### 34. Creating the functionality to generate new samples

### 35. Testing the sample generation functionality
```py
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size,embed_size) # 4096x384
    self.positions = nn.Embedding(context, embed_size) # seq. info, 512x384
    # TBD
    #self.blocks = nn.Sequential( *[Block(n_heads) for _ in range(n_layers)]) # * is unpacking operator
    self.ln = nn.LayerNorm(embed_size) # layer normalization
    self.final_linear = nn.Linear(embed_size,vocab_size,bias=BIAS) # 384x4096
    self.apply(self._init_weights)
  # parameter initialization
  def _init_weights(self,module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  def forward(self, input, targets=None):
    loss = None
    BS, SL = input.shape # 4x512
    emb = self.embeddings(input) # 4x512x384
    pos = self.positions(torch.arange(SL,device=device))
    x = emb + pos # BS x SL x 384
    # TBD
    #x = self.blocks(x) # BS x SL x 384
    x = self.ln(x) # BS x SL x 384 
    logits = self.final_linear(x) # BS x SL x 4096 (vocab_size)
    loss = None
    loss2 = None
    if targets is not None:
      BS, SL, VS = logits.shape
      logits = logits.view(BS*SL, VS)
      targets = targets.view(BS*SL)
      loss = F.cross_entropy(logits, targets)
      # manual calculation (softmax)
      counts = logits.exp()
      prob = counts /counts.sum(-1, keepdim=True)
      loss2 = - prob[torch.arange(BS*SL), targets].log().mean()
      # target[3] = 329, prob[3][329]= 0.xxx
      # cross entropy = - log p(x)
      #if (not torch.allclose(loss,loss2)):
      #  print(f"[Loss diff] Pygtorch:{loss.item()} Manual:{loss2.item()}")
    return logits, loss
  # Generate a new sample
  def generate(self, input, max=500):
    for _ in range(max):
      input = input[:, -context:] # (1, input length until max of SL)
      # x[-3:] 3 items from the end of the list
      logits, _ = self(input) # (1, input length, 4096)
      logits = logits[:, -1,:] # pick last logit (1,4096)
      probs = F.softmax(logits, dim=-1) # (1,4096), -1 for the last dim
      next = torch.multinomial(probs,num_samples=1)
      input = torch.cat((input,next),dim=1)
    return input
x,y = get_batch("train")
model = GPT()
model = model.to(dtype)
model = model.to(device)
@torch.no_grad()
def generate_sample(input):
  t1 = torch.tensor(encode(input), dtype=torch.long, device=device)
  t1 = t1[None,:] # (1,[sizeof the ids])
  newgen = model.generate(t1,max=64)[0].tolist()
  result=decode(newgen)
  print(f"{result}")
generate_sample("Once upon a time") # Once upon a timeewcent down Allummer phil comb sh philfer loverangeren~ railummer popul contin soutoveroor art  defeated building tells Divisionure I Pet bird cultZemsiting within runsware separateartmentimabich those female struickly albumolsosp murder� book contractuz mass Swedatures contro Pet deg eas Offr
```
- Result doesn't make sense as the model is not trained yet    

### 36. Coding the blocks of the LLM architecture
```py
class Block(nn.Module):
  def __init__(self, n_heads):
    super().__init__()
    head_size = embed_size // n_heads
    self.ma = Multihead(n_heads,head_size)
    self.feed_forward = ForwardLayer(embed_size)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)
  def forward(self,x):
    x = x + self.ma(self.ln1(x)) # residual connection, preventing vanishing gradient
    x = x + self.feed_forward(self.ln2(x))
    return x
```

### 37. Communication plus Computation
- The attention mechanisms allow you to work with the context, with their relationships between the different parts of the sequence of data
- It allows you to take every token, every part of the data sequence, and understand how strong is the relationship between that token and all the other tokens  of that data sequence, and so on and so forth with the rest of the tokens of the data sequence.
- It allows you to communicate that relationship, that influence, to reinforce different features of the data sequence depending on all of those relationships.

### 38. Providing computational power to the LLM
```py
class ForwardLayer(nn.Module):
  def __init__(self,mbed_size):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(embed_size,6*embed_size,bias=BIAS),
      nn.GELU(),
      nn.Linear(6*embed_size, embed_size, bias=BIAS),
      nn.Dropout(dropout)
    )
  def forward(self,x):
    x = self.network(x)
    return x
```

### 39. The Multi Head Attention Mechanism
```py
head_size = embed_size // n_heads
print(f"embed {embed_size} n_heads: {n_heads} head_size: {head_size}")
# embed 384 n_heads: 7 head_size: 54
```
- Multihead class:
```py

class Multihead(nn.Module):
  def __init__(self,n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.combine = nn.Linear(head+size * n_heads, embed_size, bias=BIAS)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    x = torch.cat()([head(x) for head in self.heads],dim=-1)
    # each head outputs(BS, SL, head_size)
    x = self.combine(x) # (BS, SL, 384)
    x = self.dropout(x)
    return x
```
- Attention head definition is later

### 40. Attention is all you need

### 41. Coding and understanding the attention head
- `@`: matrix multplication
- `torch.tril()` returns the lower triangular part of an input tensor, setting all elements above the specified diagonal to zero
- `torch.register_buffer()`: provides tensors that are a part of models' persistent state
  - Not trainable parameters (`nn.Parameter`)
  - Automatic device management: using `.to('cuda')` or `.cpu()`, those tensors move automatically
    - Member data don't have such features
  - `.state_dict()` will save those registred parameters and buffers
    - Not Python member data
```py
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.queries = nn.Linear(embed_size,head_size,bias=BIAS)
    self.keys = nn.Linear(embed_size,head_size,bias=BIAS)
    self.values = nn.Linear(embed_size,head_size,bias=BIAS)
    self.register_buffer('tril', torch.tril(torch.ones(context,context)))
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    BS, SL, VS = s.shape
    q=self.queries(x) # BS, SL, 54
    k=self.keys(x) # BS, SL, 54
    v=self.values(x) # BS, SL, 54
    # attention matrix = 512x512
    attn_w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5# BS, SL, SL
    # Q: 512x54
    # K: 54x512
    # Q@K =  512x512
    attn_w = attn_w.masked_fill(self.tril[:SL, :SL]==0, float('-inf'))
    attn_w = F.softmax(attn_w, dim=-1) # BS, SL, SL
    x = attn_w @ v # 512x512 @ 512x54 = 512x54
    return x
```

### 42. Understanding attention - deep manual dive
- Optional content for understanding attention:
```py
x,y = get_batch("train")
x = x.to(device)
y = y.to(device)
embeddings = nn.Embedding(vocab_size, embed_size).to(device)
positions = nn.Embedding(context, embed_size).to(device)
queries = nn.Linear(embed_size,head_size,bias=BIAS).to(device)
keys = nn.Linear(embed_size,head_size,bias=BIAS).to(device)
values = nn.Linear(embed_size,head_size,bias=BIAS).to(device)
tril = torch.tril(torch.ones(context,context)).to(device)
emb = embeddings(x)
pos = positions(torch.arange(context,device=device))
x = emb*pos
q=queries(x)
k=keys(x)
v=values(x)
print(q.shape,k.shape,v.shape) # torch.Size([4, 512, 54]) torch.Size([4, 512, 54]) torch.Size([4, 512, 54])
torch.set_printoptions(precision=4,threshold=1000,edgeitems=3,linewidth=80,profile='default',sci_mode=True)
print(q[0][0][:5]) # tensor([ 1.1169e-01, -3.0581e-01,  2.4552e-01, -3.6700e-01, -3.3566e-01],       device='cuda:0', grad_fn=<SliceBackward0>)
full = q@k.transpose(-2,-1) # 512x54 @ 54x512
a = q[0][5]
b = k.transpose(-2,-1)[0,:,3]
print(a.shape,b.shape) #torch.Size([54]) torch.Size([54])
c = torch.dot(a,b)
print(c) # tensor(-4.6457e+00, device='cuda:0', grad_fn=<DotBackward0>)
#
BS = batch_size 
SL = context
attn_w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5# BS, SL, SL
#attn_w = attn_w.masked_fill(tril[:SL, :SL]==0, float('-inf'))
print(attn_w.shape, v.shape) # torch.Size([4, 512, 512]) torch.Size([4, 512, 54])
print(x[0][7].shape) # torch.Size([384])
attn_scores2 = attn_w[0,7,:] # Shape [512]
# 
result = torch.zeros(54)    
for i in range(54):
  result[i] = torch.dot(attn_scores2, v[0,:,i])
print(result)
```
 
### 43. Review and debugging example
- Final version of the class
```py
class ForwardLayer(nn.Module):
  def __init__(self,mbed_size):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(embed_size,6*embed_size,bias=BIAS),
      nn.GELU(),
      nn.Linear(6*embed_size, embed_size, bias=BIAS),
      nn.Dropout(dropout)
    )
  def forward(self,x):
    x = self.network(x)
    return x
# 
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.queries = nn.Linear(embed_size,head_size,bias=BIAS)
    self.keys = nn.Linear(embed_size,head_size,bias=BIAS)
    self.values = nn.Linear(embed_size,head_size,bias=BIAS)
    self.register_buffer('tril', torch.tril(torch.ones(context,context)))
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    BS, SL, VS = x.shape
    q=self.queries(x) # BS, SL, 54
    k=self.keys(x) # BS, SL, 54
    v=self.values(x) # BS, SL, 54
    # attention matrix = 512x512
    attn_w = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5# BS, SL, SL
    # Q: 512x54
    # K: 54x512
    # Q@K =  512x512
    attn_w = attn_w.masked_fill(self.tril[:SL, :SL]==0, float('-inf'))
    attn_w = F.softmax(attn_w, dim=-1) # BS, SL, SL
    x = attn_w @ v # 512x512 @ 512x54 = 512x54
    return x
#
class Multihead(nn.Module):
  def __init__(self,n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.combine = nn.Linear(head_size * n_heads, embed_size, bias=BIAS)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    x = torch.cat([head(x) for head in self.heads],dim=-1)
    # each head outputs(BS, SL, head_size)
    x = self.combine(x) # (BS, SL, 384)
    x = self.dropout(x)
    return x
#    
class Block(nn.Module):
  def __init__(self, n_heads):
    super().__init__()
    head_size = embed_size // n_heads
    self.ma = Multihead(n_heads,head_size)
    self.feed_forward = ForwardLayer(embed_size)
    self.ln1 = nn.LayerNorm(embed_size)
    self.ln2 = nn.LayerNorm(embed_size)
  def forward(self,x):
    x = x + self.ma(self.ln1(x)) # residual connection, preventing vanishing gradient
    x = x + self.feed_forward(self.ln2(x))
    return x
class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeddings = nn.Embedding(vocab_size,embed_size) # 4096x384
    self.positions = nn.Embedding(context, embed_size) # seq. info, 512x384
    self.blocks = nn.Sequential( *[Block(n_heads) for _ in range(n_layers)]) # * is unpacking operator
    self.ln = nn.LayerNorm(embed_size) # layer normalization
    self.final_linear = nn.Linear(embed_size,vocab_size,bias=BIAS) # 384x4096
    self.apply(self._init_weights)
  # parameter initialization
  def _init_weights(self,module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0,std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
  def forward(self, input, targets=None):
    loss = None
    BS, SL = input.shape # 4x512
    emb = self.embeddings(input) # 4x512x384
    pos = self.positions(torch.arange(SL,device=device))
    x = emb + pos # BS x SL x 384
    x = self.blocks(x) # BS x SL x 384
    x = self.ln(x) # BS x SL x 384 
    logits = self.final_linear(x) # BS x SL x 4096 (vocab_size)
    loss = None
    loss2 = None
    if targets is not None:
      BS, SL, VS = logits.shape
      logits = logits.view(BS*SL, VS)
      targets = targets.view(BS*SL)
      loss = F.cross_entropy(logits, targets)
      # manual calculation (softmax)
      counts = logits.exp()
      prob = counts /counts.sum(-1, keepdim=True)
      loss2 = - prob[torch.arange(BS*SL), targets].log().mean()
      # target[3] = 329, prob[3][329]= 0.xxx
      # cross entropy = - log p(x)
      #if (not torch.allclose(loss,loss2)):
      #  print(f"[Loss diff] Pygtorch:{loss.item()} Manual:{loss2.item()}")
    return logits, loss
  # Generate a new sample
  def generate(self, input, max=500):
    for _ in range(max):
      input = input[:, -context:] # (1, input length until max of SL)
      # x[-3:] 3 items from the end of the list
      logits, _ = self(input) # (1, input length, 4096)
      logits = logits[:, -1,:] # pick last logit (1,4096)
      probs = F.softmax(logits, dim=-1) # (1,4096), -1 for the last dim
      next = torch.multinomial(probs,num_samples=1)
      input = torch.cat((input,next),dim=1)
    return input    
```

### 44. Evaluating the performance with more precision
```py
model = GPT()
model = model.to(dtype)
model = model.to(device)
compile = False # MX450
if compile:
  print("Torch:: compiling model")
  model = torch.compile(model)
print(sum(p.numel() for p in model.parameters()) /1e6, "Million parameters") # 19.8M
@torch.no_grad()
def calculate_loss():
  out={}
  model.eval()
  for split in ['train','eval']:
    l=torch.zeros(eval_iters)
    for i in range(eval_iters):
      x,y = get_batch(split)
      _,loss=model(x,y)
      l[i] = loss
    out[split]=l.mean().item()
  model.train()
  return out
l = calculate_loss()
print(l) # {'train': 8.375, 'eval': 8.375}
```

### 45. Setting up the Optimizer and Scheduler
```py
p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}
weight_decay_p = [ p for n,p in p_dict.items() if p.dim() >=2]
no_weight_decay_p = [ p for n,p in p_dict.items() if p.dim() < 2]
optimizer_groups = [
  {'params':weight_decay_p,'weight_decay': weight_decay},
  {'params':no_weight_decay_p, 'weight_decay':0.0}
]
optimizer=torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr/10)
start_iteration = 0
best_val_loss = float('inf')
```

### 46. Loading checkpoints for Inference or to restart trainings
```py
# loading checkpoints
def load_checkpoint(path):
  print("LLM - loading model")
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  iteration = checkpoint['iteration']
  loss = checkpoint['loss']
  print(f"Loaded iter {iteration} with loss {loss}")
  return iteration, loss
if os.path.exists(f"{checkpoint_dir}/{checkpoint_load_fn}") and load_pretrained:
  start_iteration,loss = load_checkpoint(checkpoint_dir + checkpoint_load_fn)
  best_val_loss = loss
```

### 47. Loading and testing a pre-trained checkpoint
```py
# inference
inference = True
if inference:
  model.eval()
  while True:
    qs = input("Enter text (q to quit):")
    if qs == "":
      continue
    if qs == "q":
      break
    generate_sample(qs)
```

### 48. Coding the learning process - Intro

### 49. The training loop
- `clip_grad_norm_()` vs `clip_grad_norm()`
  - https://stackoverflow.com/questions/71608032/pytorch-clip-grad-norm-vs-clip-grad-norm-what-is-the-differece-when-it-has-und
  - Trailing underscore: performs in-place operations. Modifies the tensor
  - Regular (no underscore): Leaves the original tensor unmodified while returns a new tensor
- When gradient is positive, increasing weight means increasing loss. Decreasing weight will decrease the loss
- When gradient is negative, increasing weight will decrease the loss
- Therefore, regardless of +/- of gradient, **focus on loss**

### 50. Training our LLM
```py
# training loop
wandb_log = False
# Top stop training in the middle, try loop
try:
  for i in tqdm(range(start_iteration,train_iters)):
    xb,yb=get_batch("train")
    logits,loss=model(xb,yb)
    # evaluating loss
    if (i % eval_interval == 0  or i == train_iters-1):
      l = calculate_loss()
      print(f"\n{i}: train loss: {l['train']} / val loss: {l['eval']}")
      generate_sample("Once upon a time")
      if l['eval'] < best_val_loss:
        print("[CHECKPOINT]: Saving with loss: ", best_val_loss)
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict':optimizer.state_dict(),
          'loss':best_val_loss,
          'iteration':i,        
        }, checkpoint_dir + checkpoint_load_fn)
      if wandb_log:
        wandb.log({
          "loss/train":l['train'],
          "loss/val":l['eval'],
          "lr": scheduler.get_last_lr()[0],
        }, step=i)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    scheduler.step()
  if wan6mindb_log:
    wandb.finish()  
except KeyboardInterrupt:
  print("Training interrupted: Cleaning up...")    
finally:
  torch.cuda.empty_cache()
  print("GPU memory released")
  #sys.exit(0)
```
- By training of 1400/100000, traing loss is down to 4.8 
- batch_size=8 yields GPU memory full at MX450
  - Use batch_size=4 for 2GB GPU
- As data is not enormous, loss will not go down much

### 51. Keeping in mind the scale of our LLM

### 52. Training the tokenizer
```py
# train a tokenizer
import sentencepiece as spm
vocab_size = 4096
spm.SentencePieceTrainer.train(
  input='zip/wiki.txt',
  model_prefix="zip/test_wiki_tokenizer",
  model_type="bpe",
  vocab_size = vocab_size,
  self_test_sample_size=8,
  input_format="text",
  character_coverage=0.995,
  num_threads=os.cpu_count()-1,
  split_digits=True,
  allow_whitespace_only_pieces = True,
  byte_fallback=True,
  unk_surface=r" \342\201\207 ",
  normalization_rule_name="identity"
)
print("Tokenizer training completed")
```
- Produces 4096 vocabulary
- Result
```bash
Tokenizer training completed
sentencepiece_trainer.cc(78) LOG(INFO) Starts training with : 
trainer_spec {
  input: zip/wiki.txt
  input_format: text
  model_prefix: zip/test_wiki_tokenizer
  model_type: BPE
  vocab_size: 4096
  self_test_sample_size: 8
  character_coverage: 0.995
  input_sentence_size: 0
  shuffle_input_sentence: 1
  seed_sentencepiece_size: 1000000
  shrinking_factor: 0.75
  max_sentence_length: 4192
  num_threads: 7
  num_sub_iterations: 2
  max_sentencepiece_length: 16
  split_by_unicode_script: 1
  split_by_number: 1
  split_by_whitespace: 1
  split_digits: 1
  pretokenization_delimiter: 
  treat_whitespace_as_suffix: 0
  allow_whitespace_only_pieces: 1
  required_chars: 
  byte_fallback: 1
  vocabulary_output_piece_score: 1
  train_extremely_large_corpus: 0
  seed_sentencepieces_file: 
  hard_vocab_limit: 1
  use_all_vocab: 0
  unk_id: 0
  bos_id: 1
  eos_id: 2
  pad_id: -1
  unk_piece: <unk>
  bos_piece: <s>
  eos_piece: </s>
  pad_piece: <pad>
  unk_surface:  \342\201\207 
  enable_differential_privacy: 0
  differential_privacy_noise_level: 0
  differential_privacy_clipping_threshold: 0
}
normalizer_spec {
...
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=2322 size=3740 all=227149 active=12704 piece=▁Wall
bpe_model_trainer.cc(268) LOG(INFO) Added: freq=2305 size=3760 all=227922 active=13477 piece=▁brain
trainer_interface.cc(689) LOG(INFO) Saving model: zip/test_wiki_tokenizer.model
trainer_interface.cc(701) LOG(INFO) Saving vocabs: zip/test_wiki_tokenizer.vocab
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
```

### 53. Encoding our dataset with the tokenizer

### 54. Conclusions and what comes next

## Section 3: Into the AI Matrix, exploring advanced visualization of the core of your LLM

### 55. Into the AI matrix: Introducing our journey to the core of LLMs
- Visualization of attention matrix

### 56. Advanced visualizations of LLM cores: a trailer

### 57. Deep dive: attention matrices at the start of LLM training
- Attention matrix
  - Lower tridiagonal matrix
  - Only the attention value of a token to the past token only
  - No value to the future token
  - Why top/left element has the highest value?
    - Sum of probability per row is 1.0
    - Top row has only 1 element which has nonzero value
    - Closer to top rows, fewer number of non-zero elements

### 58. Tracking peak attention between a specific pair of tokens
- In the initial stages of training, we often observe a hierarchical pattern: the peak of the connection starts in lower layers and progressivly moves to higher layer aligning with the intuition that more abstract, semantic relationships are processed at higher levels

### 59. Tracking peak attention between semantically meaningful connections

### 60. Exploring a set of semantic and structural connections through the training

### 61. Visualizing the capture of local sequential dependencies

### 62. The importance of understanding the attention matrix
- By learning different attention patterns, the LLM gains the ability to understand context, grasp meaning, and connect ideas across sentences, much like a human would

### 63. Long-term evolution of attention pattern dynamics
- Full training run
  - What happens when training around 50,000?
  - Moving to lower/upper layers
    - Lower layers: found short-cut
    - Upper layers: Complex relationship (?)
- Dimensional reduction
  - Reducing complexity

## Section 4: Understanding the code and concepts of an Advanced LLM

### 64. Welcome to a deep dive through an advanced LLM architecture
- Alignment is required for better interaction
  - Traditionally reinforcement training was performed but very expensive
- Will use pre-trained models
- Comparison of before/after alignment

### 65. Setting up a new environment and hosting the support files
- pip3 install tqdm datasets transformers wandb jupyter flash-attn
- https://ideami.com/llm_align
- A non-aligned LLM simply tries to continue whatever phrases you give it
  - Just extends the input phrase without concern for context
- An aligned LLM is designed to provide more useful and interactive responses
  - Continues the phrase in a helpful and interactive manner
- flash-attn source install
  - pip3 install torch torchvision torchaudio ninja 
  - python -m pip install --upgrade pip wheel setuptools
  - export TORCH_CUDA_ARCH_LIST="7.5"
  - pip3 install flash-attn --no-build-isolation
  - This takes too long (> hours). Download binary from https://github.com/Dao-AILab/flash-attention/releases
    - pip3 uninstall torch torchvision torchaudio
    - pip3 install torch==2.7.1 torchvision torchaudio
    - pip3 install ./flash_attn-2.8.2+cu12torch2.7cxx11abiTRUE-cp313-cp313-linux_x86_64.whl 
- Reinforcement learning from human feedback (RLHF)
  - First, humans review and rate the model's responses to identify what works well and what doesn't
  - Second, this feedback is used to train a reward model that predicts which responses are good
  - Third, the language model is fine-tuned using reinforcement learning, guided by the reward model to generate better and more accurate responses
- The ORPO (Optimized Response Planning Output)   alignment technique works by making the LLM focus on generating more helpful and relevant responses rather thanjust continuing the input phrase blindly, comparing useuf and non-useful responses to create a contrast, and pushing the modelto provide more useful answers
- models/ folder
  - base_model.tp: pretrained model but not aligned
  - aligned_model.pt: after alignment

### 66. Declaring the main parameters of the model
- llm.py:
- swiGLU: Swish activation function

### 67. Main structure and loss calculation

### 68. Advanced generation using extra parameters
- temperature: 0, 1, 2, ...
  - When 0, find the higherest probability
  - For lower temperature (<1), divide the logits with the given temperature, scaling up
  - For higher temperature (>1), divide the logits with the given temperature, scaling down

### 69. The main blocks of the architecture

### 70. Analyzing the computational layers of the LLM
- Swish (SiLU): gentle curve on negative x

### 71. An efficient attention implementation, part 1
- self.n_rep is used inside of function repeat_kv(). This function adjusts the key (k) and value (v) tensor shapes to match the number of query(q) heads when they differ
- When flash_attn is ready, set `self.flash = True`

### 72. An efficient attention implementation, part 2
- Rotary positional embeddings rotate each token's representation in a sequence by an amount unique to its position, by applying a rotation matrix to each token's embedding

### 73. Exploring rotary positional embeddings and other supporting functions

### 74. Analyzing the inference code
- `usr_orpo=False` 
  - When using aligned checkpoint, make it True

### 75. Preparing to run inference on the cloud and locally

### 76. Inference on non-aligned vs aligned versions of the model
- Before alignment: python3 llm.py -num 2 -temp 1 -topk 50
- After alignment: python3 llm.py -align -num 2 -temp 1 -topk 50
- Running jupyter notebook:
  - WeightsUnpickler error: Unsupported global: GLOBAL transformers.models.llama.configuration_llama.LlamaConfig was not an allowed global by default. Please use `torch.serialization.add_safe_globals([transformers.models.llama.configuration_llama.LlamaConfig])` or the `torch.serialization.safe_globals([transformers.models.llama.configuration_llama.LlamaConfig])` context manager to allowlist this global if you trust this class/function.
```py
import os
# Pytorch
import torch
# Architecture
import transformers
# Import Llama based model
from llm import Llama, ModelArgs
use_orpo = False #True  # use aligned checkpoint or not
num_answers = 2 #3
temp = 1
topk= 50
torch.serialization.add_safe_globals([transformers.models.llama.configuration_llama.LlamaConfig])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
tokenizer_path = "tokenizers/tok16384"
model_path = "./models/"    
if use_orpo==True:
        model_inf, context= "aligned_model.pt", 1024  # ORPO is trained with context of 1024
        print("Mode::Using Orpo aligned model")
else:
        model_inf, context= "base_model.pt", 512  # The original was trained with context of 512
        print("Mode::Using pretrained model without alignment")
print(f"Using model {model_inf}")   
# Load model and extract config
checkpoint = torch.load(os.path.join(model_path, model_inf), map_location=device)
config = checkpoint.pop("config")    
# temporary fix if the model was trained and saved with torch.compile
# The _orig_mod. prefix in your model's state dictionary keys is related to
# how PyTorch handles compiled models, specifically when using the torch.compile function
# When torch.compile is used, PyTorch might wrap the original model in a way that modifies
# the names of its parameters and buffers. This wrapping can prepend a prefix like _orig_mod.
# We remove those wrappings to make the checkpoint compatible with the non compiled version of the model
new_dict = dict()
for k in checkpoint.keys():
        if k.startswith("_orig_mod."):
            #print("Removing _orig_mod wrapping")
            new_dict[k.replace("_orig_mod.", "")] = checkpoint[k]
        else:
            new_dict[k] = checkpoint[k]
# Setup tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token
model_args = ModelArgs(
        dim=config.hidden_size, 
        n_layers=config.num_hidden_layers, 
        n_heads=config.num_attention_heads, 
        n_kv_heads=config.num_key_value_heads, 
        vocab_size=config.vocab_size, 
        norm_eps=config.rms_norm_eps, 
        rope_theta=config.rope_theta,
        max_seq_len=context, 
        dropout=config.attention_dropout, 
        hidden_dim=config.intermediate_size,
        attention_bias=config.attention_bias,
        mlp_bias=config.mlp_bias
    )
# Instantiate model, load parms, move to device
model = Llama(model_args)
model.load_state_dict(new_dict)
if device.type == 'cuda':
        model = model.to(torch.bfloat16)
        model = model.to(device)
model.eval()
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1e6:.2f} M parameters")
# Interactive loop
while True:
         qs = input("Enter text (q to quit) >>> ")
         if qs == "":
             continue
         if qs == 'q':
             break  
         # we activate chat template only for ORPO model because it was trained with it
         if use_orpo:
            qs = f"<s> <|user|>\n{qs}</s>\n<s> <|assistant|> "
         x = tokenizer.encode(qs)
         x = torch.tensor(x, dtype=torch.long, device=device)[None, ...]
         for ans in range(num_answers):
            with torch.no_grad():
                y = model.generate(
                    x, 
                    max_new_tokens=256, 
                    temperature=temp, 
                    top_k=topk
                )
            response = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)   

            output = model.clean_response(response)

            print("################## \n")
            print(f"### Answer {ans+1}: {output}")
```
- Using "Once upon a time"
  - Some contents are repeated
- Testing temperature=3
  - Becomes more creative(?)
  - Removes repetition issues
- Test non-aligned vs aligned for "What is a food recipe using apple?" 
  - Set temperature=1

### 77. Further reflections on the inference results

## Section 5: Coding an alignment process from scratch, understanding all the key concepts

### 78. The importance of alignment
- Ref: ORPO: Monolithic Preference Optimization without Reference Model by Hong et al: https://aclanthology.org/2024.emnlp-main.626/

### 79. The pretraining and alignment datasets
- FineWeb-Edu by HuggingFace
- orpo-dpo-mix-40k
  - 40k interactions b/w users and LLM
  - In each sample, we will ahve
    - The prompt: 1 question with no answer. Or multiple questions/answers but without the last answer
    - Chosen answer: the last answer in the chosen version of interaction
    - Rejected answer: the last answer in the rejected version of interaction

### 80. Importing the necessary libraries
- pip3 install datasets
```py
import os,sys
import math
from tqdm import tqdm
from datetime import datetime
import ipdb
from typing import List, Dict, Union
# 
import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers
from datasets import load_dataset, load_from_disk
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.cuda.empty_cache()
# Adjust the amount to show tensors for debugging
torch.set_printoptions(threshold=10)
```

### 81. Setting up the parameters for the alignment process
```py
# training parameters
batch_size = 1
epochs = 3
lr = 6e-5
lr_warmup_steps=100 # lr begins as 0 then grows
context = 1024 
alpha = 0.5 # weighting for the ORPO odds ratio
prompt_max_size = 512 # limit for the prompt part of the interaction
compile = False
dtype = torch.bfloat16
log_iters = 50
# hyperparameters
dropout = 0.
grad_clip = 1.0
weight_decay = 0.0
# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: you will be using {device}")
# Logging
project_name = "aligntest"
wandb_log = False
wandb_project = project_name
wandb_run_name = "aligntest-run"
#wandb_run_name = "aligntest-run" +date.now().strftime("%Y_%m_%d_%H_%M-%S")
if wandb_log:
  import wandb
  wandb.init(project=wandb_project,name=wandb_run_name
```

### 82. Setting up the chat template for the tokenizing process

### 83. Filtering our alignment dataset
```py
# orpo-dpo-mix-40k
dataset_path="./data/orpo_dataset"
dataset_name = "mlabonne/orpo-dpo-mix-40k"
tokenizer_path = "tokenizers/tok16384"
checkpoint_dir = "./models/"
# tokenizer dataset
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
# set our interaction template
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# make padding token equal to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token
if False: #os.path.exists(dataset_path):
  dataset=load_from_disk(dataset_path) # This is finalized data and not working the test at the current chapter
else:
  print("Filtering and tokenizing dataset")
  dataset= load_dataset(dataset_name,split="all")
  # now w will tokenize it
  # optional: filter name of the entries # 37136 vs 36622
  dataset = dataset.filter(lambda r: r["source"] != "toxic-dpo-v0.2")
  # Filter dataset
  # eliminate entries longer than 512 (prompt_max_size). 
  # we want prompt + answer to fit within the total context(1024)
  def filter_dataset(examples):
    prompt_length = tokenizer.apply_chat_template(examples["chosen"][:-1], tokenize=True, 
                                                  add_generation_prompt=True, return_tensors='pt').size(-1) 
    # examples["chosen"][:-1] -> removes the last answer
    if prompt_length < prompt_max_size: # 512
      return True
    else:
      return False
  # excluding prompts that are too long
  datatset = dataset.filter(filter_dataset)
len(dataset) # 38550 -> 43704
dataset[0]['chosen'][:-1]
'''
[{'content': 'The setting is an otherworldly, ...
  'role': 'user'},
 {'content': "As you step onto ...",
  'role': 'assistant'},
 {'content': 'Describe the unique...',
  'role': 'user'},
 {'content': "The Zephyrian ...",
  'role': 'assistant'},
 {'content': 'How does the Zephyrian humanoid perceive and interpret the color and sound patterns of its own species?',
  'role': 'user'}]
'''  
```
- Data from webpage doesn't have right keys. Download using dataset package

### 84. Pre-processing and Tokenizing the alignment dataset
```py
# orpo-dpo-mix-40k
dataset_path="./data/orpo_dataset_new"
dataset_name = "mlabonne/orpo-dpo-mix-40k"
tokenizer_path = "tokenizers/tok16384"
checkpoint_dir = "./models/"
# tokenizer dataset
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
# set our interaction template
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# make padding token equal to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token
if False: #os.path.exists(dataset_path):
  dataset=load_from_disk(dataset_path) # ? wrong data loaded
else:
  print("Filtering and tokenizing dataset")
  dataset= load_dataset(dataset_name,split="all")
  # now w will tokenize it
  # optional: filter name of the entries # 37136 vs 36622
  dataset = dataset.filter(lambda r: r["source"] != "toxic-dpo-v0.2")
  # Filter dataset
  # eliminate entries longer than 512 (prompt_max_size). 
  # we want prompt + answer to fit within the total context(1024)
  def filter_dataset(examples):
    prompt_length = tokenizer.apply_chat_template(examples["chosen"][:-1], tokenize=True, 
                                                  add_generation_prompt=True, return_tensors='pt').size(-1) 
    # examples["chosen"][:-1] -> removes the last answer
    if prompt_length < prompt_max_size: # 512
      return True
    else:
      return False
  # preprocess and tokenize data
  def preprocess_dataset(examples: Union[List,Dict]):
    prmopt = [tokenizer.apply_chat_template(item[:-1],tokenize=False, add_generation_prompt=True) for item in examples['chosen']]
  # excluding prompts that are too long
  datatset = dataset.filter(filter_dataset)
  # preprocessing and tokenize dataset
  dataset = dataset.map(preprocess_dataset, batched=True, num_proc=os.cpu_count()-1, remove_columns=dataset.column_names)
  dataset.save_to_disk(dataset_path)
```

### 85. Debugging and completing the pre-processing function
```py
# orpo-dpo-mix-40k
dataset_path="./data/orpo_dataset_new"
dataset_name = "mlabonne/orpo-dpo-mix-40k"
tokenizer_path = "tokenizers/tok16384"
checkpoint_dir = "./models/"
# tokenizer dataset
tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
# set our interaction template
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

# make padding token equal to the end of sentence token
tokenizer.pad_token = tokenizer.eos_token
if False: #os.path.exists(dataset_path):
  dataset=load_from_disk(dataset_path) # ? wrong data loaded
else:
  print("Filtering and tokenizing dataset")
  dataset= load_dataset(dataset_name,split="all")
  # now w will tokenize it
  # optional: filter name of the entries # 37136 vs 36622
  dataset = dataset.filter(lambda r: r["source"] != "toxic-dpo-v0.2")
  # Filter dataset
  # eliminate entries longer than 512 (prompt_max_size). 
  # we want prompt + answer to fit within the total context(1024)
  def filter_dataset(examples):
    prompt_length = tokenizer.apply_chat_template(examples["chosen"][:-1], tokenize=True, 
                                                  add_generation_prompt=True, return_tensors='pt').size(-1) 
    # examples["chosen"][:-1] -> removes the last answer
    if prompt_length < prompt_max_size: # 512
      return True
    else:
      return False
  # preprocess and tokenize data
  def preprocess_dataset(examples: Union[List,Dict]):
    prompt = [tokenizer.apply_chat_template(item[:-1],tokenize=False, add_generation_prompt=True) for item in examples['chosen']]
    chosen = [tokenizer.apply_chat_template(item,tokenize=False) for item in examples['chosen']]
    rejected = [tokenizer.apply_chat_template(item,tokenize=False) for item in examples['rejected']]
    # Tokenizie now
    # HF tokenizer Dict Format
    # Fields: ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing    
    inputs = tokenizer(prompt,max_length=context, padding='max_length', truncation=True, return_tensors='pt')
    pos_labels = tokenizer(chosen,max_length=context, padding='max_length', truncation=True, return_tensors='pt')
    neg_labels = tokenizer(rejected,max_length=context, padding='max_length', truncation=True, return_tensors='pt')
    inputs['positive_input_ids'] = pos_labels['input_ids']
    inputs['positive_attention_mask'] = pos_labels['attention_mask']
    inputs['negative_input_ids'] = neg_labels['input_ids']
    inputs['negative_attention_mask'] = neg_labels['attention_mask']
    # Prompt: inputs['input_ids'][0] inputs['attention_mask'][0]
    # Positive: inputs['positive_input_ids'][0] inputs['positive_attention_mask'][0]
    # Negative: inputs['negative_input_ids'][0] inputs['negative_attention_mask'][0]
    return inputs
  # excluding prompts that are too long
  datatset = dataset.filter(filter_dataset)
  # preprocessing and tokenize dataset
  dataset = dataset.map(preprocess_dataset, batched=True, num_proc=os.cpu_count()-1, remove_columns=dataset.column_names)
  dataset.save_to_disk(dataset_path)
```
- dataset format has changed. Now `dataset[0]['chosen']` doesn't work

### 86. Splitting the alignment data and creating our dataloaders
```py
# split data
dataset = dataset.shuffle(42).train_test_split(test_size=0.05)
train_data = dataset["train"]
# features: 'input_ids', 'attention_masks'
val_data = dataset["test"]
# features: 'input_ids', 'attention_masks'
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=False, collate_fn=data_collator, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,shuffle=False, collate_fn=data_collator, num_workers=0)
it = iter(train_loader)
batch = next(it)
print(tokenizer.decode(batch['input_ids'][0]))
'''
<|user|>
Is there a specific paper or citable reference from NIST or SWGDE that you can include?</s> 
<|assistant|>
</s></s>...</s>
'''
```

### 87. Setting up the model and optimizer for the alignment training process
```py
# setup architecture
from llm import Llama, ModelArgs
torch.serialization.add_safe_globals([transformers.models.llama.configuration_llama.LlamaConfig])
checkpoint = torch.load(os.path.join(checkpoint_dir,"base_model.pt"))
config = checkpoint.pop("config")
model_args = ModelArgs(
  dim=config.hidden_size,
  n_layers=config.num_hidden_layers,
  n_heads=config.num_attention_heads,
  n_kv_heads= config.num_key_value_heads,
  vocab_size = config.vocab_size,
  norm_eps=config.rms_norm_eps,
  rope_theta=config.rope_theta,
  max_seq_len=context,
  dropout=config.attention_dropout,
  hidden_dim=config.intermediate_size,
  attention_bias=config.attention_bias,
  mlp_bias=config.mlp_bias
)
# dim=768, n_layers=12, n_heads=12, vocab=16384, etc
model = Llama(model_args)
model.load_state_dict(checkpoint)
model=model.to(dtype)
model=model.to(device)
model.train()
if compile:
  model=torch.compile(model)
print(sum(p.numel() for p in model.parameters())/1e6, "Million parameters")
# 138.431232 Million parameters
#optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=device=="cuda", weight_decay=weight_decay)
num_training_steps = len(train_loader)*epochs
print(f"num_training_steps: {num_training_steps}") #124554
```

### 88. Setting up our scheduler function
```py
# scheduler for lr: first 100 steps, we do a warmup in which we increase linearly the lr
# after warming, we decrease it gradually following a cosine curve
def lr_lambda(current_step):
  if current_step < lr_warmup_steps:
    return float(current_step) / float(max(1,lr_warmup_steps))
  progress = float(current_step - lr_warmup_steps) / float(max(1, num_training_steps-lr_warmup_steps))
  return max(0.0, 0.5*(1.0*math.cos(math.pi*float(0.5)*2.0*progress)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda, last_epoch=-1)
```

### 89. Coding the training loop of the alignment process
```py
def compute_logs(prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
  pass
# Alignment training loop
try:
  for e in range(epochs):
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True):
      optimizer.zero_grad(set_to_none=True)
      batch["positive_input_ids"] = batch["positive_input_ids"].to(device)
      batch["positive_attention_mask"] = batch["positive_attention_mask"].to(device)
      batch["negative_input_ids"] = batch["negative_input_ids"].to(device)
      batch["negative_attention_mask"] = batch["negative_attention_mask"].to(device)
      batch["attention_mask"] = batch["attention_mask"].to(device)
      neg_labels = batch['negative_input_ids'].clone()
      pos_labels = batch['positive_input_ids'].clone()
      # Calculating the loss
except KeyboardInterrupt:
  print("Training interrupted: cleaning up")
finally:
  torch.cuda.empty_cache()
  print("GPU memory released")
```

### 90. Coding the alignment loss calculation - part 1
- We encourage LLM to have better probabilities for next token
- We cannot penalize LLM for bad probabilities for next token
- Loss = L_SFT + $\lambda$ L_OR
  - L_SFT = -log P_correct
  - odds(y) = P(y)/ (1- P(y))
  - OR (Odds Ratio) = odds(yw)/odds(yl)
    - yw: y winning
    - yl: y losing
  - L_OR = -$\log \sigma( \log ( {odds(yw) \over odds(yl)} )) $
  - OR(yw, yl) = $\log { { P(yw) \over 1 -P(yw)} \over {P(yl)\over 1- P(yl)}}$ = $\log [P(yw)(1-P(yl))] -  \log [(1-P(yw))P(yl)]$

### 91. Understanding how we will favor aligned responses - Deep Dive

### 92. Coding the alignment loss calculation - part 2

### 93. Coding the alignment loss calculation - part 3

### 94. Adding logging, checkpoint saving and launching the training
- Requirse > 2.4GB in GPU memory
- Running at CPUs
  - nan found. system exited

### 95. Training and testing the alignment, analyzing and expanding the stats
```py
# Declare optimizer, it helps us compute gradients, update parameters, manage learning rate, apply weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused= device == 'cuda', weight_decay=weight_decay)
# betas: control the exponential moving averages of the gradient and its square (essential part of AdamW alg) 
# eps: a small number to add numerical stability in computations
# fused: technique used to improve the performance of computations, by combining multiple operations into a single one 
# Calculate max total number of steps, the length of training loader times number of epochs
num_training_steps = len(train_loader) * epochs  #111408 with default settings - we use BS of 1 by default
# Scheduler for learning rate: first 100 steps we do a warmup in which we increase linearly the LR.
# After warmup, we decrease it gradually following a cosine curve
def lr_lambda(current_step):
    if current_step < lr_warmup_steps:
        return float(current_step) / float(max(1, lr_warmup_steps))
    progress = float(current_step - lr_warmup_steps) / float(max(1, num_training_steps - lr_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
# Compute log probabilities for positive and negative responses, necessary for Log Odds Calculation
def compute_logps(prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits):
    # note: in general we get rid of first element in labels because we want to match each token to the label in next position so 
    # we shift labels one to the left. As a consequence we get rid of last element of logits to equalize dimensions and also
    # because we dont care about the predictions for last token as there is no next token after that
    # create mask with only positions of last answer but starting from the character before the last answer,
    # because we will start predicting from that one
    mask = chosen_attention_mask[:, :-1] - prompt_attention_mask[:, 1:]
    # Gather logits corresponding to the IDs of the tokens of chosen answer
    # torch.gather selects elements of logits based on the indices in index_tensor along the specified dimension dim=2.
    # for example index gives us token 1160. Now we go to logits and from dimension 2 we extract the probability of token 1160
    # IMPORTANT: log_softmax function already incorporates the negative sign inside, so it produces negative log probabilities
    # logits[:,:-1,:] (1,1023,16384)
    # index = (mask * chosen_inputs[:, 1:]).unsqueeze(2)  (1, 1023, 1)
    # final result: per_token_logps: 1,1023
    per_token_logps = torch.gather(logits[:, :-1, :].log_softmax(-1), dim=2, 
                                    index=(mask * chosen_inputs[:, 1:]).unsqueeze(2)).squeeze(2)
    # mask the per_token_logps to leave only positions of last answer, then normalize
    # mask.sum will only sum the active elements of the mask so that you normalize by the total tokens of answer
    return torch.mul(per_token_logps, mask.to(dtype)).sum(dim=1).to(dtype) / mask.sum(dim=1).to(dtype)
# Setup Iterators and update key variables
val_iterator = iter(val_loader)
train_iterator = iter(train_loader)
log_iters = 100
eval_iters= 5 # Use a small number, otherwise things will get too slow
print(f"train loader size: {len(train_loader)}")
print(f"validation loader size: {len(val_loader)}")
print(f"number of training steps: {num_training_steps}")
@torch.no_grad()  # Prevent gradient calculation
# Calculate average of training and validation losses over multiple batches
def calculate_loss():
    global train_iterator, val_iterator
    loss_mean={}
    odds_mean={}
    ratio_mean={}
    model.eval()
    for split in ['train','val']: 
        l=torch.zeros(eval_iters)  # Create a tensor of zeros the size of eval_iters
        o=torch.zeros(eval_iters)  # Create a tensor of zeros the size of eval_iters
        r=torch.zeros(eval_iters)  # Create a tensor of zeros the size of eval_iters
        for i in range(eval_iters):
            try:
                if split == 'val':
                    batch = next(val_iterator)
                else:
                    batch = next(train_iterator)
            except StopIteration:
                if split == 'val':
                    print("####### Resetting Validation Iterator")
                    val_iterator = iter(val_loader)
                    batch = next(val_iterator)
                else:
                    print("####### Resetting Training Iterator")
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)                   
            batch["positive_input_ids"] = batch["positive_input_ids"].to(device) 
            batch["positive_attention_mask"] = batch["positive_attention_mask"].to(device)
            batch["negative_input_ids"] = batch["negative_input_ids"].to(device)
            batch["negative_attention_mask"] = batch["negative_attention_mask"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)       
            neg_labels = batch['negative_input_ids'].clone()
            pos_labels = batch['positive_input_ids'].clone()
            mask = batch['attention_mask'] * batch['positive_attention_mask']  # sets mask to have 1s in only the prompt positions
            pos_labels = pos_labels * mask.logical_not()  # puts 0s where the prompt was, preserve last answer (padding tokens are EOS(2))
            pos_labels[pos_labels == 0] = tokenizer.pad_token_id # replaces 0s with EOS(2)
            neg_labels[neg_labels == tokenizer.pad_token_id] = -100 # change 2 to -100 so that loss calculations ignore prompt and padding
            pos_labels[pos_labels == tokenizer.pad_token_id] = -100 # change 2 to -100 so that loss calculations ignore prompt and padding
            outputs_pos, loss_pos = model(batch['positive_input_ids'], pos_labels)  #  (1,1024) , (1,1024)
            outputs_neg, loss_neg = model(batch['negative_input_ids'], neg_labels)    
            # returns the average of the log probabilities for the positive samples (masking out prompt)
            pos_prob = compute_logps(
                        prompt_attention_mask=batch['attention_mask'], 
                        chosen_inputs=batch["positive_input_ids"], 
                        chosen_attention_mask=batch['positive_attention_mask'], 
                        logits=outputs_pos
                    )
            # returns the average of the log probabilities for the negative samples (masking out prompt)
            neg_prob = compute_logps(
                        prompt_attention_mask=batch['attention_mask'], 
                        chosen_inputs=batch["negative_input_ids"], 
                        chosen_attention_mask=batch['negative_attention_mask'], 
                        logits=outputs_neg
                    )    
            # CALCULATE ORPO ODDS RATIO
            log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
            sig_ratio = F.sigmoid(log_odds) # constrain to be between 0 and 1
            ratio = torch.log(sig_ratio) # apply the final log to the calculation
            # Calculate the Final Total Loss, combination of standard Cross Entropy loss and the weighted Odds Ratio
            loss = torch.mean(loss_pos - (alpha*ratio).mean()).to(dtype=dtype)
            # notice that mean() is useful if batch size is larger than 1  
            l[i]=loss.item()
            o[i]=log_odds.mean().item()
            r[i]=ratio.mean().item()
        loss_mean[split]=l.mean().item()
        odds_mean[split]=o.mean().item()
        ratio_mean[split]=r.mean().item()
    model.train()
    return loss_mean, odds_mean, ratio_mean
l, o, r = calculate_loss()
print(l,o,r)
################################################
################################################
############### ORPO TRAINING ##################
################################################
################################################
try:
    for e in range(epochs):
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True):        
            optimizer.zero_grad(set_to_none=True)  # Reset gradients    
            # Move batch data to device
            batch["positive_input_ids"] = batch["positive_input_ids"].to(device) 
            batch["positive_attention_mask"] = batch["positive_attention_mask"].to(device)
            batch["negative_input_ids"] = batch["negative_input_ids"].to(device)
            batch["negative_attention_mask"] = batch["negative_attention_mask"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)
            # Debug: if anytime you want to look inside batch: #tokenizer.decode(batch["positive_input_ids"][0])
            # Get the token IDs of positive and negative responses
            neg_labels = batch['negative_input_ids'].clone()
            pos_labels = batch['positive_input_ids'].clone()    
            # CALCULATE STANDARD CROSS ENTROPY LOSS (focused on POSITIVE responses)
            # disabling loss on prompt tokens
            # When we calculate the standard loss, we will focus on the loss of the positive responses, how good is the model
            # predicting the next character in the case of the positive, chosen responses. So we want to mask the positive IDs
            # so that they only take into account the ones of the response, and ignore the prompt
            mask = batch['attention_mask'] * batch['positive_attention_mask']  # sets mask to have 1s in only the prompt positions
            # in our case the line above is similar to just mask = batch['attention_mask'] (because all our batch sequences have same length)
            pos_labels = pos_labels * mask.logical_not()  # puts 0s where the prompt was, preserve last answer (padding tokens are EOS(2))
            pos_labels[pos_labels == 0] = tokenizer.pad_token_id # replaces 0s with EOS(2)
            neg_labels[neg_labels == tokenizer.pad_token_id] = -100 # change 2 to -100 so that loss calculations ignore prompt and padding
            pos_labels[pos_labels == tokenizer.pad_token_id] = -100 # change 2 to -100 so that loss calculations ignore prompt and padding
            # Run model for positive response
            outputs_pos, loss_pos = model(batch['positive_input_ids'], pos_labels)  #  (1,1024) , (1,1024)
            #positive input ids have all IDs including last answer and the padding has EOS 2
            #pos_labels have everything set to -100 except the IDs of the last answer
            # we don't use the negative loss for anything, that's why we didn't do a similar preparation here, but we use the
            # output negative logits for the per token log probability calculations of the negative responses
            outputs_neg, loss_neg = model(batch['negative_input_ids'], neg_labels)    
            # CALCULATE PER TOKEN LOG PROBS, necessary to calculate ORPO ODDS ratio
            # returns the average of The log probabilities for the positive samples (masking out prompt)
            pos_prob = compute_logps(
                prompt_attention_mask=batch['attention_mask'], 
                chosen_inputs=batch["positive_input_ids"], 
                chosen_attention_mask=batch['positive_attention_mask'], 
                logits=outputs_pos
            )
            # returns the average of The log probabilities for the negative samples (masking out prompt)
            neg_prob = compute_logps(
                prompt_attention_mask=batch['attention_mask'], 
                chosen_inputs=batch["negative_input_ids"], 
                chosen_attention_mask=batch['negative_attention_mask'], 
                logits=outputs_neg
            )    
            # CALCULATE ORPO ODDS RATIO
            log_odds = (pos_prob - neg_prob) - (torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob)))
            sig_ratio = F.sigmoid(log_odds) # constrain to be between 0 and 1
            ratio = torch.log(sig_ratio) # apply the final log to the calculation
            # Calculate the Final Total Loss, combination of standard Cross Entropy loss and the weighted Odds Ratio
            loss = torch.mean(loss_pos - (alpha*ratio).mean()).to(dtype=dtype)
            # notice that mean() is useful if batch size is larger than 1
            # log info every few iterations
            if i%log_iters == 0:
                # Calculate average losses 
                loss_m, log_odds_m, ratio_m = calculate_loss()
                print(f"Epochs [{e}/{epochs}] Step: [{i}/{len(train_loader)}], train loss: {loss_m['train']:.4f}, val loss: {loss_m['val']:.4f}, Odds Ratio: {log_odds_m['train']:.4f}, val Odds Ratio: {log_odds_m['val']:.4f}")
                if wandb_log:
                    wandb.log({
                        "train_loss": loss_m['train'],
                        "val_loss": loss_m['val'],
                        "train_log_odds": log_odds_m['train'],
                        "val_log_odds": log_odds_m['val'],
                        "train_ratio": (alpha*ratio_m['train']),
                        "val_ratio": (alpha*ratio_m['val']),
                        #"pos_prob": pos_prob.mean().item(),
                        #"neg_prob": neg_prob.mean().item(),                        
                        #"lr": scheduler.get_last_lr()[0],
                    }, 
                    step = (e*len(train_loader) + i))
                if torch.isnan(loss):
                    if wandb_log:   
                        wandb.finish()
                    torch.cuda.empty_cache()
                    sys.exit()
            loss.backward() # Calculate gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip) # Clip gradients
            optimizer.step() # Update model parameters
            scheduler.step() # Update learning rate
        # At the end of each epoch, save a checkpoint
        sd = model.state_dict()
        sd['config'] = config
        torch.save(sd, os.path.join(checkpoint_dir, f'{project_name}_{e+1}.pt'))
except KeyboardInterrupt:
    print("Training interrupted. Cleaning up...")
finally:
    # Release GPU memory
    torch.cuda.empty_cache()
    print("GPU memory released.")
if wandb_log:   
    wandb.finish()
torch.cuda.empty_cache()
```

### 96. Adding new code to calculate more precise training and validation losses

### 97. Comparing training and validation charts - Deep Dive

### 98. Alignment wrap-up
- Reinforcement training: Do this way
- Alignment: Don't do that way

### 99. The path towards alignment

### 100. Congrats, summary, and what's next

## Section 6: Origam + AI: Learning key insights about neural networks and AI with Origami

### 101. Welcome to this original origami based journey to the core of AI

### 102. In Search of the Magical Mappings of Creativity, using Origami!

### 103. The Search for the Perfect Mapping: datasets and dimensionality

### 104. From Linearity to Complexity: Neural Networks and the Nonlinearities of Life

### 105. Bending the Rules: Non-Linear transformations and the key to complexity

### 106. Not Too Tight, Not Too Loose - Finding the perfect fit

### 107. How increasing the dimensionality impacts the latent complexity of the network

### 108. The Power of Depth: Creating Sophisticated Mappings with AI networks

### 109. From high dimensional manifolds to dynamic and ever changing latent spaces
- Continuosly adopting network

### 110. Advanced digital representations of the latent complexity of neural networks

### 111. Visualizing the Journey: Loss Landscapes and the Search for Optimal Weights

### 112. Example of the dynamic Loss Landscape of a generative adversarial network

### 113. Lucy - Real Time Visualization of the changing weights of a neural network

### 114. Charting the hidden depths: a recap of our transformative latent space journey

### 115. Summary of the last sections

## Section 7: Activating the Generative Model of your own mind

### 116. Introducing our final journey

### 117. A guided visualization experience to exercise the generative model in your head

### 118. Intro to the journey to the center of the neuron

### 119. The container, the salty ocean and the 150000 cortical columns

### 120. Visualizing the pyramidal neuron

### 121. The Synapse, visualizing the input-output interface

### 122. Biological vs Artificial Neurons: Inputs, Outputs, Speed, etc

### 123. Learning in biological and artificial neurons

### 124. Planning, decision making and world models

### 125. Efficiency: sparsity in biological vs artificial networks

### 126. Consciousness: within the neurons

### 127. The future, towards AGI / ASI

### 128. Conclusion and congratulations

    
