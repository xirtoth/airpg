import json

# Read the current notebook
with open('github_version.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the install cell (cell 3, index 3)
nb['cells'][3]['source'] = [
    "# Uninstall conflicting packages\n",
    "!pip uninstall -y transformers diffusers accelerate huggingface-hub -q\n",
    "!pip install -q --force-reinstall numpy==1.26.4\n",
    "!pip install -q torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118\n",
    "!pip install -q huggingface-hub==0.23.2 safetensors sentencepiece protobuf\n",
    "!pip install -q --no-deps transformers==4.41.0\n",
    "!pip install -q --no-deps diffusers==0.27.0\n",
    "!pip install -q accelerate==0.30.0 gradio\n",
    "\n",
    'print("✅ All packages installed!")\n',
    'print("⚠️ MUST RESTART: Runtime → Restart runtime")'
]

# Write the fixed notebook
with open('colab_setup.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Fixed!")
