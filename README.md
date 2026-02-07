# AI Fantasy RPG

An interactive text-based RPG game powered by AI! The game uses a Large Language Model to narrate your adventure and Stable Diffusion to generate images of each scene.

## Features

- 🎮 **AI Dungeon Master**: LLM narrates and responds to your actions
- 🎨 **Scene Visualization**: Stable Diffusion generates images for each scene
- 🗣️ **Natural Language Input**: Type what you want to do in plain English
- 🌟 **Unlimited Possibilities**: The AI responds to any action you describe

## Requirements

- Python 3.8+
- NVIDIA GPU with 6GB+ VRAM (GTX 1060 6GB or better)
- CUDA installed
- ~15GB disk space for models

## Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

4. **Install PyTorch with CUDA** (if not already installed):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Edit `config.py` to customize:
- **LLM_MODEL**: The language model to use for narration
  - Default: `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` (quantized for 6GB VRAM)
  - Alternative: `TheBloke/Llama-2-7B-Chat-GPTQ`
- **DIFFUSION_MODEL**: The Stable Diffusion model for images
  - Default: `stabilityai/stable-diffusion-2-1-base`
  - For lower VRAM: `runwayml/stable-diffusion-v1-5`

## Running the Game

```powershell
python game.py
```

**First run**: Models will be downloaded automatically (this takes time and disk space!)

## How to Play

1. The AI describes the starting scene
2. Type what you want to do (e.g., "I examine the map on the table")
3. The AI narrates what happens next
4. An image of the scene is generated and displayed
5. Repeat!

### Example Actions

- `I talk to the bartender`
- `I approach the hooded figure cautiously`
- `I examine the map carefully`
- `I draw my sword and prepare for battle`
- `I search the room for hidden passages`
- `I order an ale and listen to the tavern gossip`

### Commands

- `quit` or `exit` - End the game
- `help` - Show tips and help

## Tips for Best Experience

1. **Be specific**: "I carefully examine the ancient runes on the door" is better than "I look"
2. **Roleplay**: Describe your character's thoughts and feelings
3. **Explore**: The AI will respond to creative actions
4. **Combat**: You can engage in battles - just describe your actions!

## Performance Notes

### For GTX 1060 6GB:
- LLM response: ~10-30 seconds
- Image generation: ~20-40 seconds
- Total turn time: ~30-70 seconds

### Memory Optimization:
The game uses several optimizations for 6GB VRAM:
- Quantized LLM models (GPTQ)
- Attention slicing for Stable Diffusion
- XFormers memory efficient attention (if available)
- Models are loaded in float16

### If you run out of VRAM:
1. Edit `config.py`:
   - Use `runwayml/stable-diffusion-v1-5` instead of SD 2.1
   - Reduce `IMAGE_WIDTH` and `IMAGE_HEIGHT` to 384x384
   - Reduce `INFERENCE_STEPS` to 20
2. Close other GPU applications
3. Reduce `MAX_NEW_TOKENS` for shorter responses

## Troubleshooting

### "CUDA out of memory"
- Close other applications using GPU
- Use smaller models in `config.py`
- Reduce image resolution

### Models downloading slowly
- Models are large (7-10GB). First download takes time.
- They're cached in `~/.cache/huggingface/`

### "No module named 'transformers'"
```powershell
pip install -r requirements.txt
```

### Images not displaying
- Images are saved to `generated_images/` folder
- Open them manually if auto-display fails

## Generated Content

- **Images**: Saved in `generated_images/` folder
- **Naming**: `scene_XXX_TIMESTAMP.png`

## License

This is a personal project. Use and modify as you wish!

## Credits

- Built with Transformers, Diffusers, and PyTorch
- Powered by open-source LLMs and Stable Diffusion

Enjoy your adventure! 🗡️🐉
