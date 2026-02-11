"""
  Wrapper around monogpt where you can train your own language model easily
  pip install tokenizers # is recommended since they use rust impl for BPE (monogpt's python impl is slower)

  **The current setup in nn.py uses 8GB-10GB VRAM, if you don't have the required VRAM consider lowering hyperparams for training**
 
  To run: python3 wrapper.py [args]
"""

import os
import glob
import torch
from monogpt.nn import MONOGPT, train
from monogpt.tokenizer import BytePairEncoding
from monogpt.utils import load_dataset_from_dir

# wrapper settings
data_folder: str = "./data/"
build_folder: str = "./build/" # for storing weights & caching tokenizer stuff
tokenizer_weigths_path: str = build_folder + "wrapper_w_tokenizer"
tokenized_dataset_path: str = build_folder + "tokenized_dataset.bin"
nn_weights_path: str = build_folder + "monogpt_v1.pth" # Weights will be saved to this folder after training
vocab_size: int = 16000 # number of tokens, lower for faster tokenization traingin
verbose: bool = True # keep true is recommended
dataset: str|None = None # DO NOT TOUCH THIS, IT IS COMPUTED IN WRAPPER SCRIPT

# Training Hyperparams | Change them to alter training performance (Tweak to the dataset needs)
max_iters: int = 5000
batch_size: int = 32 # this affects vram alot
learning_rate: float = 3e-4
eval_interval: int = 500
percent_training: float = 0.9 # data division; 90% for training

if not torch.cuda.is_available():
  print("WARNING: torch isnt compiled with cuda.")

# See what data we have
if not os.path.exists(data_folder):
  raise FileNotFoundError(f"Data folder not found. Create a folder on \"{data_folder}\" and add .txt files for training.")

# figure out which tokenizer to use
try:
  from tokenizers import Tokenizer, models, pre_tokenizers, trainers
  print(f">>>> Using Hugging face's tokenizer! (i love rust implementations)")
  tokenizer_weigths_path = tokenizer_weigths_path + ".json" # hf uses .json
  if os.path.exists(tokenizer_weigths_path):
    tokenizer = Tokenizer.from_file(tokenizer_weigths_path)
    print(f">>>> Found and loaded tokenizer's weights on {tokenizer_weigths_path}.")
  else:
    print(f">>>> Did not find pre-computed tokenizer's weights, training them now on dataset with vocab_size={vocab_size}.")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=[])
    all_files: list = glob.glob("data/*.txt")
    assert (all_files is not None) and (len(all_files) > 0), f"Something went wrong loading dataset from \"{data_folder}\", is the folder empty?"
    tokenizer.train(all_files, trainer)
    tokenizer.save(tokenizer_weigths_path) # actual training, already prints to console
    print(f">>>> Saved tokenizer's weights to \"{tokenizer_weigths_path}\".")
except ImportError:
  print(f">>>> Since Hugging face's tokenizers are not installed(pip install tokenizers), using monogpt's slower tokenization...")
  tokenizer = BytePairEncoding() 
  if os.path.exists(tokenizer_weigths_path + ".model"):
    tokenizer.load(tokenizer_weigths_path + ".model")
    print(f">>>> Found and loaded tokenizer's weights on {tokenizer_weigths_path + '.model'}.")
  else:
    print(f">>>> Did not find pre-computed tokenizer's weights, training them now on dataset with vocab_size={vocab_size}.")
    print(f"This might take a while... (lower vocab size, minimum 256, for faster compute)")

    print(f">>>> Loading entire dataset to RAM...")
    dataset: str = load_dataset_from_dir(data_folder)
    assert (dataset is not None) and (dataset != ""), f"Something went wrong loading dataset from \"{data_folder}\", is the folder empty?"
    print(f">>>> Loaded {len(dataset)} characters.")

    tokenizer.train(dataset, vocab_size=vocab_size, verbose=verbose)
    tokenizer.save(tokenizer_weigths_path) # this saves .vocab and .model
    print(f">>>> Saved tokenizer's weights to \"{tokenizer_weigths_path + ".model"}\" and \"{tokenizer_weigths_path + ".vocab"}\".")

# make sure tokenizer actual returns anything | This line LIKELY WILL NEVER result in an error but just to make sure we test for tokenizer anyway
assert (tokenizer is not None) and (tokenizer.encode("hi") is not None), "Something went wrong with setting up tokenizers. If not using hugging face's tokenizer\
  do: pip install tokenizers. To attempt and fix this unknown error"

# test if we actually have data computed at this point (either by RAM or DISK)
assert (dataset is not None) or (os.path.exists(tokenized_dataset_path)), f"Something went wrong.\
  We must have either dataset loaded in RAM or a pre-computed tokenized dataset in DISK: \"{tokenized_dataset_path}\"."

# call pre-done training loop
model: MONOGPT = train(dataset=dataset, tokenized_dataset_path=tokenized_dataset_path,
                       max_iters=max_iters,
                       batch_size=batch_size,
                       learning_rate=learning_rate,
                       eval_interval=eval_interval,
                       percent_training=percent_training,
                       save_to=nn_weights_path,
                       verbose=verbose)