import os
from huggingface_hub import hf_hub_download

# Function to read the token from a file
def read_token(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.readline().strip()
    except FileNotFoundError:
        raise ValueError(f"Token file not found: {file_path}")

# Define the model name and file
model_name = "cjpais/llava-v1.6-34B-gguf"
model_file = "llava-v1.6-34b.Q4_K_M.gguf"

# Read the Hugging Face access token from the file
token_file_path = 'hf_token.txt'
HF_TOKEN = read_token(token_file_path)
if HF_TOKEN is None:
    raise ValueError(f"Hugging Face token is not set. Please ensure {token_file_path} contains your token.")

# Download the model from Hugging Face Hub
model_path = hf_hub_download(
    model_name,
    filename=model_file,
    local_dir='models/',  # Download the model to the "models" folder
    token=HF_TOKEN
)

print("My model path:", model_path)
