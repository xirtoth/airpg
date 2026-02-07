# Running AI Fantasy RPG on Google Colab

This guide will help you run the AI Fantasy RPG on Google Colab with **upgraded models** that take full advantage of Colab's better GPU!

## 🚀 Quick Start

### Option 1: Using the Notebook (Recommended)

1. **Upload to Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `colab_setup.ipynb` (File → Upload notebook)
   - Or manually upload all project files

2. **Enable GPU**
   - Click Runtime → Change runtime type
   - Select GPU (T4 is free and works great!)
   - Click Save

3. **Run All Cells**
   - Click Runtime → Run all
   - Wait for models to download (5-10 minutes first time)
   - Click the public URL when it appears!

### Option 2: Manual Setup

If you prefer to set up manually, follow these steps:

#### 1. Create a new Colab notebook

#### 2. Install dependencies
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers diffusers accelerate safetensors Pillow gradio xformers bitsandbytes
```

#### 3. Upload your files
Upload these files to Colab:
- `game_ui_colab.py` (create from the notebook)
- `config_colab.py` (create from the notebook)

#### 4. Run the game
```python
!python game_ui_colab.py
```

## 🎮 Model Upgrades

### What's Different from Local Version?

| Component | Local (6GB VRAM) | Colab (15GB VRAM) | Improvement |
|-----------|------------------|-------------------|-------------|
| **LLM** | TinyLlama 1.1B | Mistral 7B | 6x larger, much smarter |
| **Image Model** | SD 1.5 (512px) | SDXL-Turbo (768px) | 2x resolution, better quality |
| **Generation Speed** | Slower | Faster (better GPU) | ~2-3x faster |
| **Image Steps** | 12 steps | 4 steps (Turbo!) | 3x faster images |
| **Story Quality** | Basic | Advanced narratives | Much better! |

### Available Model Configurations

#### For Story Generation (LLM):

1. **Mistral 7B Instruct** (Default - Best Balance)
   - Excellent quality storytelling
   - Good speed
   - Model: `mistralai/Mistral-7B-Instruct-v0.2`

2. **Zephyr 7B Beta** (Great Alternative)
   - Very good instruction following
   - Creative responses
   - Model: `HuggingFaceH4/zephyr-7b-beta`

3. **Phi-2** (Fastest)
   - Still much better than TinyLlama
   - Uses less VRAM
   - Model: `microsoft/phi-2`

4. **Llama 2 7B Chat** (Requires HuggingFace Token)
   - Excellent for creative storytelling
   - Need to authenticate with HF
   - Model: `meta-llama/Llama-2-7b-chat-hf`

#### For Image Generation:

1. **SDXL-Turbo** (Default - Best Speed/Quality)
   - High resolution (768x768)
   - Only needs 4 steps
   - Fast generation
   - Model: `stabilityai/sdxl-turbo`

2. **SDXL Base** (Best Quality)
   - Highest quality images
   - Needs 20+ steps (slower)
   - Model: `stabilityai/stable-diffusion-xl-base-1.0`

3. **SD 2.1 Base** (Good Balance)
   - Good quality at 512x512
   - Faster than SDXL
   - Model: `stabilityai/stable-diffusion-2-1-base`

## 🔧 Configuration Tips

### Change Models On-The-Fly

Add this cell to your notebook to change models:

```python
import config_colab as config

# Change LLM
config.LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"

# Change image model  
config.DIFFUSION_MODEL = "stabilityai/sdxl-turbo"
config.INFERENCE_STEPS = 4

# Adjust image size
config.IMAGE_WIDTH = 768
config.IMAGE_HEIGHT = 768
```

### Save Images to Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Update config
config.IMAGES_DIR = '/content/drive/MyDrive/airpg_images'
config.SAVE_IMAGES = True
```

### Optimize for Speed

```python
# Faster story generation
config.MAX_NEW_TOKENS = 200
config.TEMPERATURE = 0.7

# Faster images
config.INFERENCE_STEPS = 4
config.IMAGE_WIDTH = 512
config.IMAGE_HEIGHT = 512
```

### Optimize for Quality

```python
# Better story generation
config.MAX_NEW_TOKENS = 500
config.TEMPERATURE = 0.9

# Better images
config.DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
config.INFERENCE_STEPS = 25
config.GUIDANCE_SCALE = 7.5
config.IMAGE_WIDTH = 1024
config.IMAGE_HEIGHT = 1024
```

## ⚠️ Troubleshooting

### Out of Memory Error

If you get CUDA out of memory:

1. **Use smaller models:**
```python
config.LLM_MODEL = "microsoft/phi-2"
config.DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"
```

2. **Reduce image size:**
```python
config.IMAGE_WIDTH = 512
config.IMAGE_HEIGHT = 512
```

3. **Clear GPU memory:**
```python
import torch
torch.cuda.empty_cache()
```

### Models Loading Slowly

- First run downloads ~10GB of models (normal, be patient)
- Models are cached, subsequent runs are much faster
- Use Colab Pro for faster downloads and better GPU

### Session Timeout

Colab free sessions timeout after ~12 hours of inactivity:
- Save important images to Drive
- Session will restart, but models stay cached
- Just re-run the last cell to restart

### Quality Issues

If story/images aren't good enough:
- Try different model combinations
- Increase `MAX_NEW_TOKENS` for longer stories
- Increase `INFERENCE_STEPS` for better images
- Adjust `TEMPERATURE` (0.7-0.9) for creativity

## 💡 Pro Tips

1. **Share Your Adventure**: The Gradio interface creates a public URL valid for 72 hours - share with friends!

2. **Save Progress**: Screenshot your adventure or save images to Drive

3. **Experiment**: Try different model combinations to find your favorite

4. **Cost**: Colab is free! Pro version ($10/month) gets you:
   - Better GPUs (A100)
   - Longer sessions
   - No interruptions

4. **GPU Usage**: Check GPU usage with:
```python
!nvidia-smi
```

## 📊 Expected Performance

On Colab T4 GPU (free tier):

| Task | Time |
|------|------|
| Initial model loading | 5-10 minutes (first time) |
| Story generation | 3-5 seconds |
| Image generation (SDXL-Turbo) | 5-8 seconds |
| Image generation (SDXL Base) | 15-20 seconds |
| Total turn time | 8-13 seconds |

## 🎯 Recommended Setup

For the best experience, use:

```python
# Balanced - Fast and High Quality
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DIFFUSION_MODEL = "stabilityai/sdxl-turbo"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
INFERENCE_STEPS = 4
MAX_NEW_TOKENS = 350
```

This gives you:
- ✅ High-quality storytelling
- ✅ Beautiful 768x768 images
- ✅ Fast generation (~10 seconds per turn)
- ✅ Fits comfortably in 15GB VRAM

## 🌟 Enjoy Your Adventure!

You're now running a much more powerful version of the game than possible on most local hardware. Have fun exploring AI-generated fantasy worlds!

Need help? Check the troubleshooting section or try different model combinations.
