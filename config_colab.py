"""Configuration settings for the AI RPG game - Colab Optimized Version."""

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DIFFUSION_MODEL = "stabilityai/sdxl-turbo"

# Generation parameters
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_P = 0.95

# Image parameters
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
INFERENCE_STEPS = 2
GUIDANCE_SCALE = 0.0

# Game settings
SAVE_IMAGES = True
IMAGES_DIR = "generated_images"
GAME_TITLE = "AI Fantasy RPG"
IMAGES_DIR = "generated_images"
SAVE_IMAGES = True
