import json

notebook_path = 'colab_setup.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0: Header
nb['cells'][0]['source'] = [
    "# AI Fantasy RPG - Google Colab Setup (Updated: Feb 7, 2026 - Version 15)\n",
    "\n",
    "Run this notebook on Google Colab with GPU enabled for the best experience!\n",
    "\n",
    "**Setup Instructions:**\n",
    "1. Go to Runtime → Change runtime type → Select GPU (T4 recommended)\n",
    "2. Run all cells in order\n",
    "3. The game will launch in a web interface with a public URL\n"
]

# Update Cell 2: Git Clone logic
nb['cells'][2]['source'] = [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Completely fresh start - avoids all git conflicts\n",
    "if os.path.exists('airpg'):\n",
    "    print(\"Removing old airpg folder...\")\n",
    "    shutil.rmtree('airpg')\n",
    "\n",
    "print(\"Cloning fresh repository...\")\n",
    "!git clone https://github.com/xirtoth/airpg.git\n",
    "%cd airpg\n",
    "\n",
    "# Verify we have the latest config\n",
    "!grep \"LLM_MODEL\" config_colab.py"
]

# Update Cell 3: Dependencies (Version 15)
nb['cells'][3]['source'] = [
    "# Forced dependency resolution fix (VERSION 15 - Strict Compatibility)\n",
    "!pip -q install --upgrade pip\n",
    "!pip -q install \"numpy>=2.0.0,<2.1.0\" # Satisfy both Python 3.12 and pre-installed numba/tensorflow\n",
    "!pip -q install \\\n",
    "  \"transformers>=4.45.0\" \\\n",
    "  \"diffusers>=0.30.0\" \\\n",
    "  \"accelerate>=0.30.0\" \\\n",
    "  \"gradio>=5.0.0\" \\\n",
    "  \"bitsandbytes>=0.43.0\" \\\n",
    "  \"sentencepiece\" \\\n",
    "  \"protobuf\" \\\n",
    "  \"huggingface-hub>=0.25.0\" \\\n",
    "  \"opencv-python\" \\\n",
    "  \"opencv-python-headless\" \\\n",
    "  \"datasets\" \\\n",
    "  \"transformers[sentencepiece]\"\n",
    "\n",
    "print(\"✅ Version 15 Installs done — NOW: Runtime → Restart runtime\")"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook successfully updated to Version 15!")
