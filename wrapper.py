"""
  Wrapper around monogpt where you can train your own language model easily
  pip install tokenizers # is recommended since they use rust impl for BPE (monogpt's python impl is slower)

  **The current setup in nn.py uses 4GB-10GB VRAM(depending on dataset), if you don't have the required VRAM consider lowering nn.py's network architecture**
 
  To run: python3 wrapper.py
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
vocab_size: int = 16000 # number of tokens, lower for faster tokenization training; !!! LOWER IF UR DATASET IS SMALLER!!!
verbose: bool = True # keep true is recommended
dataset: str|None = None # DO NOT TOUCH THIS, IT IS COMPUTED IN WRAPPER SCRIPT

# Training Hyperparams | Change them to alter training performance (Tweak to the dataset needs)
max_iters: int = 5000
batch_size: int = 32 # this affects vram alot
learning_rate: float = 3e-4
eval_interval: int = 500
percent_training: float = 0.9 # data division; 90% for training

device:str = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
  print("WARNING: torch isnt compiled with cuda.")

# See what data we have
if not os.path.exists(data_folder):
  raise FileNotFoundError(f"Data folder not found. Create a folder on \"{data_folder}\" and add .txt files for training.")

if (not os.path.exists(nn_weights_path)) or (not os.path.exists(tokenized_dataset_path)):
  print(f">>>> Loading entire dataset to RAM...")
  dataset: str = load_dataset_from_dir(data_folder)
  assert (dataset is not None) and (dataset != ""), f"Something went wrong loading dataset from \"{data_folder}\", is the folder empty?"
  print(f">>>> Loaded {len(dataset)} characters.")

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
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=[], special_tokens=["<|endoftext|>", "<|pad|>", "<|unk|>"])
    all_files: list = glob.glob("data/*.txt")
    assert (all_files is not None) and (len(all_files) > 0), f"Something went wrong loading dataset from \"{data_folder}\", is the folder empty(only .txt files are parsed)?"
    tokenizer.train(all_files, trainer)
    tokenizer.save(tokenizer_weigths_path) # actual training, already prints to console
    print(f">>>> Saved tokenizer's weights to \"{tokenizer_weigths_path}\".")
except ImportError:
  print(f">>>> Since Hugging face's tokenizers are not installed(pip install tokenizers), using monogpt's slower tokenization...")
  tokenizer = BytePairEncoding() 
  # TODO: Handle special tokens
  # tokenizer.special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
  if os.path.exists(tokenizer_weigths_path + ".model"):
    tokenizer.load(tokenizer_weigths_path + ".model")
    print(f">>>> Found and loaded tokenizer's weights on {tokenizer_weigths_path + '.model'}.")
  else:
    print(f">>>> Did not find pre-computed tokenizer's weights, training them now on dataset with vocab_size={vocab_size}.")
    print(f"This might take a while... (lower vocab size, minimum 256, for faster compute)")
    tokenizer.train(dataset, vocab_size=vocab_size, verbose=verbose)
    tokenizer.save(tokenizer_weigths_path) # this saves .vocab and .model
    print(f">>>> Saved tokenizer's weights to \"{tokenizer_weigths_path + ".model"}\" and \"{tokenizer_weigths_path + ".vocab"}\".")

# make sure tokenizer actual returns anything | This line LIKELY WILL NEVER result in an error but just to make sure we test for tokenizer anyway
assert (tokenizer is not None) and (tokenizer.encode("hi") is not None), "Something went wrong with setting up tokenizers. If not using hugging face's tokenizer\
  do: pip install tokenizers. To attempt and fix this unknown error"

# test if we actually have data computed at this point (either by RAM or DISK)
assert (dataset is not None) or (os.path.exists(tokenizer_weigths_path)), f"Something went wrong.\
  We must have either dataset loaded in RAM or a pre-computed tokenized dataset in DISK: \"{tokenized_dataset_path}\"."

# if weights exist in disk we dont bother training
if os.path.exists(nn_weights_path):
    print(f">>>> Found pre-trained model at {nn_weights_path}!")
    print(">>>> Loading weights instead of training...")
    
    model = MONOGPT(tokenizer.get_vocab_size(), verbose=True)
    checkpoint = torch.load(nn_weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    print(">>>> Model loaded successfully.")
else:
    print(f">>>> No existing model found at {nn_weights_path}.")
    print(">>>> Starting fresh training run...")
    
    # Run the training loop (which returns the trained model)
    model: MONOGPT = train(dataset, tokenizer,
                       tokenized_dataset_path=tokenized_dataset_path,
                       max_iters=max_iters,
                       batch_size=batch_size,
                       learning_rate=learning_rate,
                       eval_interval=eval_interval,
                       percent_training=percent_training,
                       save_to=nn_weights_path,
                       verbose=verbose
                       ).to(device)


####### Using the model to chat
print("\n" + "="*40)
print("CHAT MODE")
print("="*40)
print("Type 'quit' to exit.\n")

# prompt engineering
sys_prompt = (
  "The following is a conversation between a User and a helpful Assistant. "
  "The Assistant is polite, concise, and smart.\n\n"
)
history = [] 
MAX_HISTORY = 3 # Keep last 3 exchanges to save context window

model.eval()
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
      break

    full_prompt = sys_prompt 
    for turn in history[-MAX_HISTORY:]:
      full_prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
    full_prompt += f"User: {user_input}\nAssistant:"
    
    encoded = tokenizer.encode(full_prompt)
    if not isinstance(encoded, list): encoded = encoded.ids # hf support

    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
      output_ids = model.generate_n(n=100, context=x.tolist()[0], temp=0.7)
    
    decoded_text = tokenizer.decode(output_ids)
    response = decoded_text[len(full_prompt):] 
    if "User:" in response:
      response = response.split("User:")[0]
    response = response.strip()

    clean_text = decoded_text.replace("Ġ", " ").replace("Ċ", "\n")
    print(f"MonoGPT: {clean_text}\n")
    history.append({'user': user_input, 'bot': clean_text})