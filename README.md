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
