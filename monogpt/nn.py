from torch import nn

class MONOGPT(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()

    self.TIME_SIZE:int = 1024
    self.EMBEDDING_DIM:int = 512
    self.VOCAB_SIZE = vocab_size

    self.define()
    
  def define(self):
    self.token_embedding = nn.Embedding(self.VOCAB_SIZE, self.EMBEDDING_DIM)
    self.positional = nn.Embedding(self.TIME_SIZE, self.EMBEDDING_DIM)

  def forward(self):
    pass