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