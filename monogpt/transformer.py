# https://arxiv.org/pdf/1706.03762
import torch
from torch import nn

class AttentionHead(nn.Module):
  def __init__(self, embedding_dim:int, head_size:int, block_size:int, bias:bool=False):
    super().__init__()
    self.register_buffer("tril_mask", torch.tril(torch.ones(block_size, block_size)))
    self.head_size = head_size
    self.define(embedding_dim, bias)

  def define(self, embedding_dim, bias):
    self.query = nn.Linear(embedding_dim, self.head_size, bias=bias)
    self.key = nn.Linear(embedding_dim, self.head_size, bias=bias)
    self.value = nn.Linear(embedding_dim, self.head_size, bias=bias)

  def forward(self, x:torch.Tensor):
    B, T, C = x.shape

    q:torch.Tensor = self.query(x)
    k:torch.Tensor = self.key(x)
    v:torch.Tensor = self.value(x)

    wei = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
    wei = wei.masked_fill(self.tril_mask[:T, :T] == 0, float("-inf"))
    wei = nn.functional.softmax(wei, -1) @ v
    return wei

class MultiAttentionHead(nn.Module):
  def __init__(self, h:int, embedding_dim:int, block_size:int, bias:bool=False):
    super().__init__()

    head_size = embedding_dim // h
    self.define(h, embedding_dim, head_size, block_size, bias)

  def define(self, number_heads, embedding_dim, head_size, block_size, bias):
    self.heads = [AttentionHead(embedding_dim, head_size, block_size, bias) for _ in range(number_heads)]
    self.heads = nn.ModuleList(self.heads)

    self.linear = nn.Linear(embedding_dim, embedding_dim)

  def forward(self, x):
    B, T, C = x.shape

    head_outputs = [head(x) for head in self.heads]
    out = torch.cat(head_outputs, dim=-1)
    return self.linear(out)
    