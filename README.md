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

It is a lightweight Transformer implementation for GPT training and inference in PyTorch. 
Designed to: given raw text -> output a generative model. It handles the entire pipeline from raw data ingestion to generative AI with an inference script.

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
### Full Instalation and Usage Instructions
```bash
# 1. Install dependencies
pip install torch numpy
pip install tokenizers # if you want faster tokenization; else it will automatically use monogpts implementation

# 2. Add your text files to /data
cp my_book.txt ./data/

# 3. Run the pipeline (Train -> Chat)
python wrapper.py
```
