import json

notebook_path = 'colab_setup.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update Cell 0: Header
nb['cells'][0]['source'] = [
    "# AI Fantasy RPG - Google Colab Setup (Updated: Feb 7, 2026 - Version 20)\n",
    "\n",
    "Run this notebook on Google Colab with GPU enabled for the best experience!\n",
    "\n",
    "**Setup Instructions:**\n",
    "1. Go to **Runtime → Disconnect and delete runtime**\n",
    "2. Run all cells in order\n",
    "3. The game will launch in a web interface with a public URL\n"
]

# Update Cell 2: Git Clone logic
nb['cells'][2]['source'] = [
    "import os\n",
    "import shutil\n",
    "\n",
    "# Ensure we are in /content root and not a nested folder\n",
    "if os.getcwd() != '/content':\n",
    "    %cd /content\n",
    "\n",
    "# Completely fresh start\n",
    "if os.path.exists('airpg'):\n",
    "    print(\"Removing old airpg folder...\")\n",
    "    shutil.rmtree('airpg')\n",
    "\n",
    "print(\"Cloning fresh repository...\")\n",
    "!git clone https://github.com/xirtoth/airpg.git\n",
    "%cd airpg\n"
]

# Update Cell 3: Dependencies (Version 20)
nb['cells'][3]['source'] = [
    "# Forced dependency resolution fix (VERSION 20 - Force Reinstall)\n",
    "!pip -q install --upgrade pip\n",
    "\n",
    "# 1. Install core components using pre-compiled binaries where possible\n",
    "!pip -q install --force-reinstall \"sentencepiece\" \"protobuf\" \"numpy>=2.0.0,<2.1.0\" \"datasets\"\n",
    "\n",
    "# 2. Install main libraries with force-reinstall to override cached versions\n",
    "!pip -q install --force-reinstall \\\n",
    "  \"transformers[sentencepiece]>=4.48.0\" \\\n",
    "  \"diffusers>=0.31.0\" \\\n",
    "  \"accelerate>=0.30.0\" \\\n",
    "  \"gradio>=5.1.0\" \\\n",
    "  \"bitsandbytes>=0.43.0\" \\\n",
    "  \"huggingface-hub>=0.25.0\" \\\n",
    "  \"opencv-python-headless\"\n",
    "\n",
    "print(\"✅ Version 20 Installs done — NOW: Runtime → Restart session\")"
]

# Update Cell 4: Verification
nb['cells'][4]['source'] = [
    "# Step 1.1: Verify EVERYTHING (Run after Restart)\n",
    "try:\n",
    "    from transformers import MT5Tokenizer\n",
    "    import sentencepiece\n",
    "    print(\"✅ sentencepiece and MT5Tokenizer are WORKING!\")\n",
    "except Exception as e:\n",
    "    print(f\"❌ ERROR: Tokenizer still broken: {e}\")\n",
    "\n",
    "import numpy, transformers, gradio\n",
    "print(f\"Numpy Version: {numpy.__version__}\")\n",
    "print(f\"Transformers Version: {transformers.__version__}\")\n",
    "print(f\"Gradio Version: {gradio.__version__}\")"
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook successfully updated to Version 20!")
