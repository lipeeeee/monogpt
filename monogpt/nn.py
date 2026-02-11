import time
import torch
from transformer import *
from torch import nn

class MONOGPT(nn.Module):
  def __init__(self, vocab_size: int, verbose:bool=False):
    super().__init__()
    self.verbose = verbose

    self.CONTEXT_SIZE:int = 1024
    self.EMBEDDING_DIM:int = 512
    self.BLOCK_NUM:int = 6 # number of transformer blocks
    self.HEADS_PER_BLOCK:int = 8 # number of attention heads in each block
    self.VOCAB_SIZE:int = vocab_size # how large our vocab is(number of tokens)

    if verbose:
      print(f"MONOGPT Initialized with:\
\n\tCONTEXT_SIZE={self.CONTEXT_SIZE}\
\n\tVOCAB_SIZE={self.VOCAB_SIZE}\
\n\tEMBEDDING_DIM={self.EMBEDDING_DIM}\
\n\tBLOCK_NUM={self.BLOCK_NUM}\
\n\tHEADS_PER_BLOCK={self.HEADS_PER_BLOCK}\
\n\tTOTAL_TRANSFORMER_HEADS={self.HEADS_PER_BLOCK * self.BLOCK_NUM}")

    self.define()
    
  def define(self):
    self.token_embedding = nn.Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM)
    self.positional = nn.Embedding(self.CONTEXT_SIZE, self.EMBEDDING_DIM)
    self.blocks = [Block(self.HEADS_PER_BLOCK, self.EMBEDDING_DIM, self.CONTEXT_SIZE) for _ in range(self.BLOCK_NUM)]
    self.blocks = nn.ModuleList(self.blocks)
    self.blocks = nn.Sequential(*self.blocks)
    self.layer_norm_final = nn.LayerNorm(self.EMBEDDING_DIM)
    self.lm_head = nn.Linear(self.EMBEDDING_DIM, self.VOCAB_SIZE)

  def forward(self, x, targets=None):
    B, T = x.shape
    loss = None

    # embedding
    token_embed = self.token_embedding(x)
    pos_embed = self.positional(torch.arange(T, device=x.device))
    embed = token_embed + pos_embed # (B, T, C)

    # compute logits
    logits = self.blocks(embed)
    logits = self.layer_norm_final(logits)
    logits = self.lm_head(logits)

    if targets is not None:
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = nn.functional.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, context:list, temp:float=1.0):
    """Provides softmax distribuition over the next token (1, VOCAB_SIZE)"""
    if len(context) > self.CONTEXT_SIZE:
      context = context[-self.CONTEXT_SIZE:] # keep only last CONTEXT_SIZE elems

    context = torch.tensor(context, dtype=torch.long).unsqueeze(0) # unsqueeze does (CONTEXT) -> (1, CONTEXT)
    context = context.to(next(self.parameters()).device) # moves to same device as self's params
    logits, _ = self(context, targets=None)
    logits = logits[:, -1, :] # (1, T, VOCAB_SIZE) -> (1, VOCAB_SIZE)
    logits = logits / temp
    
    probs = nn.functional.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item() # int token
  
  def generate_n(self, n:int, context:list, temp:float=1.0):
    """Generates n new tokens"""
    assert n > 0
    generated:list[int] = []
    curr_context = context.copy()

    for _ in range(n):
      new_tok = self.generate(context, temp)
      generated.append(new_tok)

      curr_context.append(new_tok)
    
    return generated

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device):
  """ x = [0, 1, 2, 3]
      y = [1, 2, 3, 4] (Next token prediction)"""
  # Generate random starting indices in the data
  # We subtract block_size to ensure we don't go out of bounds
  ix = torch.randint(len(data) - block_size, (batch_size,))
  
  # Stack the 1D chunks into a 2D batch tensor
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])

  if device == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  else:
    x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad() # dont backprop here
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device):
  out = {}
  model.eval()
    
  for split, data in [('train', train_data), ('val', val_data)]:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(data, block_size, batch_size, device)
      _, loss = model(X, targets=Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
        
  model.train()
  return out

# takes in (dataset, tokenzier, vocab_size=8192, verbose=False)
def train(dataset: str, tokenizer, 
          max_iters: int = 5000,
          batch_size: int = 32,
          learning_rate: float = 3e-4,
          eval_interval: int = 500,
          percent_training: float = 0.9, # 90% for training
          verbose: bool = True):

  device:str = "cuda" if torch.cuda.is_available() else "cpu"
  if verbose: print(f"Using device: {device}")

  # encode dataset
  if verbose: print(f"Starting to encode dataset with tokenizer(size={tokenizer.vocab_size})")
  tokenized_dataset = tokenizer.encode(dataset)
  if verbose: print(f"Passing tokenized dataset(len={len(tokenized_dataset)}) to tensor")
  data_tensor = torch.tensor(tokenized_dataset, dtype=torch.long, device="cpu") # force cpu
  if verbose: print(f"Dataset is on: {data_tensor.device}")
  n = int(percent_training * len(data_tensor))
  train_data = data_tensor[:n]
  val_data = data_tensor[n:]
  if verbose: print(f"Data Tensor Shape: {data_tensor.shape}, using {percent_training*100}% for training")
  
  # initialize model & optim
  m = MONOGPT(tokenizer.vocab_size, verbose).to(device)
  optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)
  t0 = time.time() # first time mention because idk

  # training loop
  if verbose: print(f">>>> Starting training")
  for iter in range(max_iters):
    if (iter % eval_interval == 0 or iter == max_iters -1):
      losses = estimate_loss(m, train_data, val_data, 200, m.CONTEXT_SIZE, batch_size, device)
      dt = time.time() - t0
      if verbose: print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (took {dt:.2f} sec)")
      t0 = time.time()

    xb, xy = get_batch(train_data, m.CONTEXT_SIZE, batch_size, device)
    logits, loss = m(xb, targets=xy)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

  if verbose: print(f">>>> Finished training")
  return m
  
if __name__ == "__main__":
  from utils import load_dataset_from_dir
  from tokenizer import BytePairEncoding
  dataset = load_dataset_from_dir("data")
  bpe = BytePairEncoding()

  # bpe.train(dataset, vocab_size=8192)
  # bpe.save("models/tinyshakespeare")
  bpe.load("models/tinyshakespeare.model")

  m = train(dataset, bpe, batch_size=16)