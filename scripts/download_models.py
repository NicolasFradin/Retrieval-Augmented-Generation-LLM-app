import os
import urllib.request

"""
There is multiple way to download LLM models: 
- Download on HuggingFace
- Llama repo with Meta access
- Frameworks (GPT4All, 0llama...)

Here we are downloading through HuggingFace
"""


def download_file(file_link, filename):
    # Checks if the file already exists before downloading
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(file_link, filename)
        print("File downloaded successfully.")
    else:
        print("File already exists.")

# Downloading GGML model from HuggingFace
ggml_model_path = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf"
filename = '../models/llama-2-7b.Q4_K_M.gguf'
download_file(ggml_model_path, filename)

ggml_model_path = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/blob/main/zephyr-7b-beta.Q4_K_M.gguf"
filename = '../models/zephyr-7b-beta.Q4_K_M.gguf'
download_file(ggml_model_path, filename)

# Fine-tuned Llama-2-Chat is optimized for dialogue use cases
ggml_model_path = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf"
filename = '../models/llama-2-7b-chat.Q4_K_M.gguf'
download_file(ggml_model_path, filename)

