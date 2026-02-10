import os

def load_data(file_path: str) -> str:
  """Reads a text file and returns the entire content as a string."""
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()
      print(f"Loaded {len(text)} characters from {file_path}")
      return text
  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    return ""

def load_dataset_from_dir(directory: str) -> str: # warning:loads everything to RAM
  """Reads all .txt files in a directory and combines them into one string."""
  full_text = []

  # Walk through the directory
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".txt"):
        path = os.path.join(root, file)
        try:
          with open(path, 'r', encoding='utf-8') as f:
            full_text.append(f.read())
        except Exception as e:
          print(f"Skipping {path}: {e}")

  combined = "\n".join(full_text)
  print(f"Loaded {len(combined)} characters from {len(full_text)} files.")
  return combined