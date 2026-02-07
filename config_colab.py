"""Configuration settings for the AI RPG game - Colab Optimized Version."""

# LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Best quality
LLM_MODEL = "microsoft/phi-2"  # Faster, works well on free Colab

# DIFFUSION_MODEL = "stabilityai/sdxl-turbo"  # Best quality
DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"  # More compatible

# Generation parameters
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_P = 0.95

# Image parameters
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5

# Game settings
GAME_TITLE = "AI Fantasy RPG"
IMAGES_DIR = "generated_images"
SAVE_IMAGES = True
