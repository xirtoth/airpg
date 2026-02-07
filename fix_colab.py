import json

# Read the current notebook
with open('colab_setup.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title
nb['cells'][0]['source'] = [
    "# AI Fantasy RPG - Google Colab Setup (Updated: Feb 7, 2026 - Version 5)\n",
    "\n",
    "Run this notebook on Google Colab with GPU enabled for the best experience!\n",
    "\n",
    "**Setup Instructions:**\n",
    "1. Go to Runtime \u2192 Change runtime type \u2192 Select GPU (T4 recommended)\n",
    "2. Run all cells in order\n",
    "3. The game will launch in a web interface with a public URL"
]

# Update the install cell (cell 3, index 3)
nb['cells'][3]['source'] = [
    "# Forced dependency resolution fix\n",
    "!pip -q install --upgrade pip\n",
    "!pip -q uninstall -y huggingface-hub transformers tokenizers datasets gradio numpy opencv-python opencv-python-headless || true\n",
    "!pip -q install \\\n",
    "  \"numpy==1.26.4\" \\\n",
    "  \"huggingface-hub==0.33.5\" \\\n",
    "  \"transformers==4.41.0\" \\\n",
    "  \"tokenizers==0.19.1\" \\\n",
    "  \"datasets==2.13.0\" \\\n",
    "  \"gradio==5.50.0\" \\\n",
    "  \"diffusers==0.27.2\" \\\n",
    "  \"accelerate==0.22.0\" \\\n",
    "  \"opencv-python\" \\\n",
    "  \"opencv-python-headless\"\n",
    "\n",
    "print(\"✅ Installs done — NOW: Runtime → Restart runtime\")\n"
]

# Insert version check after install cell
if len(nb['cells']) <= 4 or "Verify Versions" not in str(nb['cells'][4]):
    nb['cells'].insert(4, {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1.1: Verify Versions (Run after Restart)\n",
            "import numpy, transformers, huggingface_hub, gradio\n",
            "print(\"numpy:\", numpy.__version__)\n",
            "print(\"transformers:\", transformers.__version__)\n",
            "print(\"huggingface_hub:\", huggingface_hub.__version__)\n",
            "print(\"gradio:\", gradio.__version__)\n"
        ]
    })

# Write the fixed notebook
with open('colab_setup.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Fixed!")
