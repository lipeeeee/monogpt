def get_stats(ids:list, counts:dict|None=None) -> dict[tuple, int]:
  """ids=[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}"""
  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]): # iterate consecutive elements
      counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids:list, pair:tuple, idx:int) -> list[int]:
  """ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]"""
  newids = []
  i = 0
  while i < len(ids):
      # if not at the very last position AND the pair matches, replace it
      if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
          newids.append(idx)
          i += 2
      else:
          newids.append(ids[i])
          i += 1
  return newids

class Tokenizer:
  def __init__(self):
    self.merges:dict[tuple[int, int], int] = {} # dict[(token(int)_1, token(int)_2), new_token]
    self.pattern:str = ""
    self.special_tokens:dict[str, int] = {} # dict[str, token]
    self.vocab:dict = self.build_vocab()

  def build_vocab(self) -> dict: # dict[int, byte] 
    vocab = { idx: bytes([idx]) for idx in range(256) } # first 256 ascii 
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    for special, idx in self.special_tokens.items():
      vocab[idx] = special.encode("utf-8")
    return vocab

  def train(self, text:str, vocab_size:int, verbose:bool=False):
    raise NotImplementedError

  def encode(self, text: str):
    raise NotImplementedError

  def decode(self, ids: list):
    raise NotImplementedError

  def save(self, file_path: str):
    raise NotImplementedError

  def load(self, file_path: str):
    raise NotImplementedError

# https://en.wikipedia.org/wiki/Byte-pair_encoding
class BytePairEncoding(Tokenizer):
  def __init__(self):
    super().__init__()

  def train(self, text:str, vocab_size:int, verbose:bool=False):
    assert vocab_size >= 256, "Vocab must have at least 256 characters for storing ASCII entries"
    num_merges = vocab_size - 256

    # input text preprocessing
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255

    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)

        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
          print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

    self.merges = merges
    self.vocab = vocab

  def decode(self, ids:list[int]) -> str:
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode("utf-8", errors="replace")
    return text

  def encode(self, text:str) -> list[int]:
    text_bytes = text.encode("utf-8") # raw bytes
    ids = list(text_bytes) # list of integers in range 0..255

    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

        if pair not in self.merges:
            break

        idx = self.merges[pair]
        ids = merge(ids, pair, idx)
    return ids

if __name__ == "__main__":
  t = "There was a brown fox, the brown fox hopped the fence often haha"
  bpe = BytePairEncoding()
  bpe.train(t, 256+3, verbose=True)
