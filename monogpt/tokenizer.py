import unicodedata

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

def replace_control_characters(s: str) -> str:
  # we don't want to print control characters
  # which distort the output (e.g. \n or much worse)
  # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
  # http://www.unicode.org/reports/tr44/#GC_Values_Table
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch) # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
  # pretty print a token, escaping control characters
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s

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

  def save(self, file_prefix: str):
    """
      Saves two files: file_prefix.vocab and file_prefix.model
      This is inspired (but not equivalent to!) sentencepiece's model saving:
      - model file is the critical one, intended for load()
      - vocab file is just a pretty printed version for human inspection only
    """
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      f.write("minbpe v1\n")
      f.write(f"{self.pattern}\n")

      # write the special tokens, first the number of them, then each one
      f.write(f"{len(self.special_tokens)}\n")
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")

      # the merges dict
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")

    vocab_file = file_prefix + ".vocab"
    inverted_merges = {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in self.vocab.items():
        # note: many tokens may be partial utf-8 sequences
        # and cannot be decoded into valid strings. Here we're using
        # errors='replace' to replace them with the replacement char �.
        # this also means that we couldn't possibly use .vocab in load()
        # because decoding in this way is a lossy operation!
        s = render_token(token)
        if idx in inverted_merges:
          # if this token has children, render it nicely as a merge
          idx0, idx1 = inverted_merges[idx]
          s0 = render_token(self.vocab[idx0])
          s1 = render_token(self.vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else:
          # otherwise this is leaf token, just print it
          # (this should just be the first 256 tokens, the bytes)
          f.write(f"[{s}] {idx}\n")

  def load(self, model_file: str):
    """Inverse of save() but only for the model file"""
    assert model_file.endswith(".model")
    merges = {}
    special_tokens = {}
    idx = 256
    with open(model_file, 'r', encoding="utf-8") as f:
      version = f.readline().strip()
      assert version == "minbpe v1"
      self.pattern = f.readline().strip()

      num_special = int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)

      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1

    self.merges = merges
    self.special_tokens = special_tokens
    self.vocab = self.build_vocab()

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
