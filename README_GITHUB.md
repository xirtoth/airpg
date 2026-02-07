# AI Fantasy RPG 🎮⚔️

An interactive AI-powered fantasy RPG that uses Large Language Models for storytelling and Stable Diffusion for generating beautiful scene images!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/airpg/blob/main/colab_setup.ipynb)

## 🚀 Quick Start on Google Colab (Recommended)

**The easiest way to run this game is on Google Colab with upgraded models!**

1. Click the "Open in Colab" badge above
2. Go to Runtime → Change runtime type → Select GPU (T4)
3. Run all cells
4. Play with much better models than possible on local hardware!

### Why Colab?
- 🎯 **Free GPU** with 15GB VRAM (vs. 6GB locally)
- 🚀 **Upgraded Models**: Mistral 7B + SDXL-Turbo (vs. TinyLlama + SD 1.5)
- 📸 **Better Images**: 768x768 high quality (vs. 384x384)
- ⚡ **Faster Generation**: Better GPU = faster turns
- 🔗 **Public URL**: Share your game with friends!

## 🎯 Features

- 🎮 **AI Dungeon Master**: Advanced LLM narrates your adventure
- 🎨 **Scene Visualization**: SDXL generates stunning images for each scene
- 🗣️ **Natural Language**: Type actions in plain English
- 🌟 **Unlimited Possibilities**: The AI responds to anything you describe
- 🔄 **Multiple Model Options**: Choose speed vs. quality based on your needs

## 🖥️ Local Installation (Advanced Users)

If you have a powerful GPU (6GB+ VRAM), you can run locally:

### Requirements
- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM
- CUDA installed
- ~15GB disk space for models

### Setup

1. **Clone this repository**
```bash
git clone https://github.com/YOUR_USERNAME/airpg.git
cd airpg
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the game**
```bash
# Web UI (recommended):
python game_ui.py

# Command line version:
python game.py
```

## ⚙️ Configuration

### For Colab (config_colab.py)
Default configuration uses upgraded models:
- **LLM**: Mistral 7B Instruct (excellent storytelling)
- **Image**: SDXL-Turbo (fast, high quality)
- **Resolution**: 768x768
- **Generation Speed**: ~10 seconds per turn

### For Local (config.py)
Optimized for 6GB VRAM:
- **LLM**: TinyLlama 1.1B (lightweight)
- **Image**: SD 1.5 (compatible)
- **Resolution**: 384x384
- **Generation Speed**: ~15-20 seconds per turn

## 🎮 How to Play

1. Start the game and wait for the AI to generate an opening scene
2. Read the narration and view the generated image
3. Type what you want to do (e.g., "I draw my sword and enter the cave")
4. The AI responds to your action and generates a new scene
5. Continue your adventure!

### Example Actions
- "I carefully examine the mysterious artifact"
- "I attempt to negotiate with the dragon"
- "I cast a fireball at the approaching enemies"
- "I search the room for hidden passages"

Be creative! The AI understands natural language and will respond to any action you describe.

## 📊 Model Comparison

| Aspect | Local (6GB) | Colab Free (15GB) |
|--------|-------------|-------------------|
| LLM Size | 1.1B params | 7B params |
| Story Quality | Basic | Excellent |
| Image Resolution | 384x384 | 768x768 |
| Image Quality | Good | Stunning |
| Speed (per turn) | 15-20s | 8-13s |
| Cost | Free (your GPU) | Free (Google's GPU) |

## 🛠️ Troubleshooting

### Colab Issues
- **Out of Memory**: Use Phi-2 instead of Mistral 7B
- **Slow Generation**: Already optimized with SDXL-Turbo
- **Session Timeout**: Free tier has 12-hour limit, just restart

### Local Issues
- **CUDA not available**: Install CUDA toolkit and PyTorch with CUDA
- **Out of Memory**: Reduce image size in config.py
- **Slow performance**: Use smaller models or upgrade GPU

## 🔗 Useful Links

- [Detailed Colab Setup Guide](COLAB_SETUP.md)
- [Model Configuration Options](config_colab.py)
- [Troubleshooting Guide](COLAB_SETUP.md#troubleshooting)

## 📝 Credits

Built with:
- [Transformers](https://huggingface.co/transformers) by Hugging Face
- [Diffusers](https://huggingface.co/diffusers) by Hugging Face  
- [Gradio](https://gradio.app/) for the web interface
- [PyTorch](https://pytorch.org/) for deep learning

Models used:
- Mistral 7B Instruct by Mistral AI
- SDXL-Turbo by Stability AI
- Alternative options from various open-source contributors

## 📄 License

This project is open source and available for personal and educational use.

## 🤝 Contributing

Feel free to open issues or submit pull requests with improvements!

---

**Enjoy your AI-powered fantasy adventure! 🐉⚔️🏰**
