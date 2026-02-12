<div align="center">
  <h1>monogpt</h1>
  
  **Train** your own language model with your own data!
</div>

---
This was heavily inspired and based on the [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) whitepaper.

---

### Monogpt has
- Custom **Transformer** architecture (Decoder-only GPT)
- Custom **BytePairEncoding Tokenizer** training pipeline
- Hackable **Attention** implementation with stackable blocks
- Inference script(**wrapper.py**) made to be extremely accessible
- It can reach **GPT2 level** if trained for long enough with the right data

It is a lightweight Transformer implementation for GPT training and inference in PyTorch. 
Designed to be a drop-in solution: given raw text -> output a generative model. It handles the entire pipeline from raw data ingestion to generative AI with an inference script.

---
### Using monogpt to create models
Simply create **"data/"** folder, put data that you want the model to learn from and just run **wrapper.py**!
```bash
ubuntu@ubuntu:~$ python3 wrapper.py

>>>> Loading entire dataset to RAM...
Loaded 269352791 characters from 2 files.
>>>> Loaded 269352791 characters.
>>>> Using Hugging faces tokenizer! (i love rust implementations)
>>>> Found and loaded tokenizers weights on ./build/wrapper_w_tokenizer.json.
>>>> No existing model found at ./build/monogpt_v1.pth.
>>>> Starting fresh training run...
Using device: cuda
>>>> Starting to encode dataset with tokenizer(size=16000)
>>>> Finished tokenizing dataset (took 402.57 sec)
>>>> Saved computed tokenized dataset to ./build/tokenized_dataset.bin
>>>> Passing tokenized dataset(len=64673413) to tensor
>>>> Dataset is on: cpu
>>>> Data Tensor Shape: torch.Size([64673413]), using 90.0% for training
MONOGPT Initialized with:
        CONTEXT_SIZE=512
        VOCAB_SIZE=16000
        EMBEDDING_DIM=384
        BLOCK_NUM=6
        HEADS_PER_BLOCK=6
        TOTAL_TRANSFORMER_HEADS=36
```

---
### Technical Neural Net info
(Most are the same that was used on [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) whitepaper)
- Positional Embeddings: Learnable absolute positions
- Attention: Causal Self-Attention (Masked trigonometry)
- Activation: GELU (Gaussian Error Linear Unit)
- Optimization: AdamW with Weight Decay
- Regularization: Dropout & LayerNorm

---
### Transformer Hyperparams and Training
- The current version of monogpt is **perfect** for small-medium datasets. Meaning the neural net is not complex enough that it will start overfitting and taking long to train.
- But it still has some complexity/is deep. The current network params take 2GB-10GB of VRAM. Mainly because of **AdamW** optimizer and **PyTorch** just being greedy with RAM.

It's very hackable so:
- **lower** the params for: Faster training(**lower VRAM aswell**) and support lower complexity datasets (won't learn very niche information on data).
- **increase** the params for: More complex datasets and potential to learn more connections between words, resulting in industry-standard and even GPT2 level results.

```python
# nn.py
class MONOGPT(nn.Module):
  def __init__(self, vocab_size: int, verbose:bool=False):
    super().__init__()
    self.verbose = verbose

    self.CONTEXT_SIZE:int = 512 # how much the network will see before predicting next token
    self.EMBEDDING_DIM:int = 384 # how much dimension will each token be represented in
    self.BLOCK_NUM:int = 6 # number of transformer blocks
    self.HEADS_PER_BLOCK:int = 6 # number of attention heads in each block
    self.DROPOUT: float = 0.2 # just normal feedforward block dropout(defined in paper)
````

---
### Full Instalation and Usage Instructions
<h6>!you should install a cuda compiled pytorch for compute speed!</h6>

```bash
# 1. Install dependencies
pip install torch numpy
pip install tokenizers # if you want faster tokenization; else it will automatically use monogpts implementation

# 2. Add your text files to /data
cp my_book.txt ./data/

# 3. Run the pipeline (Train -> Chat)
python wrapper.py
```
