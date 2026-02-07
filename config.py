"""Configuration settings for the AI RPG game."""

# Model configurations
LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Instruction-tuned chat model
# GPT-Neo-2.7B is too heavy with SD on 6GB VRAM

DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"  # SD 1.5 for better compatibility
# Alternative: "stabilityai/stable-diffusion-2-1-base" for better quality

# Generation parameters
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_P = 0.95

# Image parameters
IMAGE_WIDTH = 384  # Increased from 256
IMAGE_HEIGHT = 384  # Increased from 256
INFERENCE_STEPS = 12  # Increased from 7 for better quality
GUIDANCE_SCALE = 7.0  # Slightly higher for better adherence to prompt

# Game settings
GAME_TITLE = "AI Fantasy RPG"
IMAGES_DIR = "generated_images"
SAVE_IMAGES = True
