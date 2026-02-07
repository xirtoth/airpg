"""
AI Fantasy RPG Game - Web UI Version
An interactive RPG with a beautiful web interface powered by Gradio
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
from PIL import Image
import datetime
import gradio as gr
import config_colab as config

# Don't set custom cache paths for Colab


class AIRPGGame:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.llm_model = None
        self.tokenizer = None
        self.diffusion_pipe = None
        self.conversation_history = []
        self.turn_count = 0
        self.current_image = None
        
        # Create images directory
        if config.SAVE_IMAGES:
            os.makedirs(config.IMAGES_DIR, exist_ok=True)
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load both LLM and Stable Diffusion models."""
        print("\n🔄 Loading AI models (this may take a minute)...")
        
        # Load LLM with 4-bit quantization for Mistral 7B to fit in Colab VRAM
        try:
            print(f"Loading LLM: {config.LLM_MODEL}")
            
            # Setup 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL,
                trust_remote_code=True
            )
            
            # Add padding token if missing (Mistral needs this)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("✓ LLM loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise
        
        # Load Stable Diffusion
        try:
            print(f"Loading Stable Diffusion: {config.DIFFUSION_MODEL}")
            self.diffusion_pipe = AutoPipelineForText2Image.from_pretrained(
                config.DIFFUSION_MODEL,
                torch_dtype=torch.float16,
                variant="fp16" if "turbo" in config.DIFFUSION_MODEL else "main",
                use_safetensors=True
            ).to(self.device)
            
            # If not turbo, use a faster scheduler
            if "turbo" not in config.DIFFUSION_MODEL:
                self.diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.diffusion_pipe.scheduler.config
                )
            
            self.diffusion_pipe.enable_attention_slicing()
            
            try:
                self.diffusion_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            print("✓ Stable Diffusion loaded successfully!")
        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            self.diffusion_pipe = None
    
    def generate_story_response(self, player_action, progress=gr.Progress()):
        """Generate the next part of the story based on player action."""
        progress(0.3, desc="AI is thinking...")
        
        # Build conversation history
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]
            for i, entry in enumerate(recent_history):
                action = entry['action']
                response = entry['response'][:200]
                if action != "START":
                    conversation_context += f"Player: {action}\nStory: {response}\n\n"
                else:
                    conversation_context += f"Story: {response}\n\n"
        
        # Mistral Instruct Format
        prompt = f"""[INST] You are a professional fantasy RPG dungeon master. 
Continue the story based on the player's action. Write 2-3 atmospheric sentences.
Context:
{conversation_context}

Player's action: {player_action} [/INST] Story:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if "Story:" in response:
                response = response.split("Story:")[-1].strip()
            else:
                response = response.split("[/INST]")[-1].strip()
            
            # Clean up incomplete sentences at the end
            if response and not response[-1] in '.!?"':
                last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_period > 50:
                    response = response[:last_period + 1]
            
            return response
        except Exception as e:
            print(f"Error generating story: {e}")
            return f"The world shimmers as a magical anomaly occurs... (Error: {e})"
    
    def generate_image_prompt(self, scene_description):
        """Use the LLM to generate a visual prompt for Stable Diffusion."""
        prompt = f"""[INST] You are a visual artist. Create a short, highly descriptive image prompt (15-20 words) for a fantasy RPG scene based on this description:
"{scene_description}"
Focus only on visual elements: lighting, environment, and objects. Do not include story or character names. [/INST] Visual Prompt:"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.4,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Visual Prompt:" in result:
                result = result.split("Visual Prompt:")[-1].strip()
            else:
                result = result.split("[/INST]")[-1].strip()
                
            # Clean up any "Story:" or other meta text
            result = result.replace("Story:", "").strip()
            return result
        except Exception as e:
            print(f"Error generating visual prompt: {e}")
            # Fallback to keyword extraction
            return self._fallback_image_prompt(scene_description)

    def _fallback_image_prompt(self, scene_description):
        """Keyword-based fallback for image prompt generation."""
        blacklist = {'you', 'your', 'the', 'and', 'with', 'that', 'this', 'from', 'into'}
        words = scene_description.lower().replace('.', ' ').replace(',', ' ').split()
        keywords = [w for w in words if len(w) > 4 and w not in blacklist]
        return ", ".join(keywords[:8])
    
    def generate_scene_image(self, scene_description, progress=gr.Progress()):
        """Generate an image based on the scene description."""
        if not self.diffusion_pipe:
            return None
        
        # Clear VRAM cache occasionally
        if self.turn_count % 3 == 0:
            torch.cuda.empty_cache()
        
        progress(0.6, desc="Creating image prompt...")
        
        # Use simple keyword extraction
        keywords = self.generate_image_prompt(scene_description)
        
        # Add fantasy RPG style
        enhanced_prompt = f"fantasy RPG art, {keywords}, detailed digital painting, dramatic lighting, high quality"
        
        # Negative prompt (SDXL Turbo doesn't need much, others do)
        is_turbo = "turbo" in config.DIFFUSION_MODEL
        negative_prompt = "blurry, low quality, distorted, deformed, text, watermark" if not is_turbo else None
        
        print("\n" + "="*60)
        print("🎨 IMAGE PROMPT:")
        print(f"Positive: {enhanced_prompt}")
        print(f"Is Turbo: {is_turbo}")
        print("="*60 + "\n")
        
        progress(0.7, desc="Generating scene image...")
        
        try:
            with torch.no_grad():
                result = self.diffusion_pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=config.INFERENCE_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    height=config.IMAGE_HEIGHT,
                    width=config.IMAGE_WIDTH
                )
                image = result.images[0]
            
            if config.SAVE_IMAGES:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config.IMAGES_DIR}/scene_turn{self.turn_count:03d}_{timestamp}.png"
                image.save(filename)
                self.current_image = filename
                return filename  # Return path to file
            
            return image  # Fallback to PIL object
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def process_action(self, player_action, history, progress=gr.Progress()):
        """Process player action and return updated chat history and image."""
        if history is None:
            history = []
            
        if not player_action or not player_action.strip():
            return history, self.current_image
        
        player_action = player_action.strip()
        self.turn_count += 1
        
        # Generate story response
        progress(0.1, desc="Processing your action...")
        response = self.generate_story_response(player_action, progress)
        
        # Save to internal logic history
        self.conversation_history.append({
            'action': player_action,
            'response': response
        })
        
        # Generate image
        progress(0.5, desc="Visualizing the scene...")
        self.current_image = self.generate_scene_image(response, progress)
        
        # Update Chatbot (Gradio 5 messages format)
        history.append({"role": "user", "content": player_action})
        history.append({"role": "assistant", "content": response})
        
        progress(1.0, desc="Done!")
        return history, self.current_image
    
    def start_new_game(self):
        """Start a new game with initial scene."""
        import random
        
        self.conversation_history = []
        self.turn_count = 0
        
        # Random starting scenarios (Simplified for prompt brevity)
        starting_scenarios = [
            "You awaken in a dimly lit tavern. The smell of ale and roasted meat fills the air. A hooded figure in the corner watches you intently.",
            "You find yourself at the edge of a dark forest. Ancient trees loom overhead. A worn path leads deeper into the woods.",
            "You stand before massive castle gates. Storm clouds gather overhead. The guards at the gate eye you suspiciously.",
            "You awaken on a sandy beach, waves lapping at your feet. The wreckage of a ship lies scattered along the shore.",
            "You're in a bustling marketplace filled with exotic merchants. Suddenly, a cloaked figure grabs your arm and whispers 'Follow me'."
        ]
        
        initial_scene = random.choice(starting_scenarios)
        
        self.conversation_history.append({
            'action': 'START',
            'response': initial_scene
        })
        
        # Generate initial image
        self.current_image = self.generate_scene_image(initial_scene)
        
        # Return history in messages format
        return [{"role": "assistant", "content": initial_scene}], self.current_image


# Initialize game
print("Initializing AI Fantasy RPG...")
game = AIRPGGame()

# Create Gradio interface
with gr.Blocks(title="AI Fantasy RPG") as demo:
    gr.Markdown(
        """
        # 🎮 AI Fantasy RPG
        ## An Epic Adventure Powered by AI
        
        Your choices shape the story. The AI narrates your adventure and generates images of each scene.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Your Adventure",
                height=500,
                show_label=True,
                avatar_images=("👤", "🎲"),
                type="messages"
            )
            
            with gr.Row():
                action_input = gr.Textbox(
                    label="What do you do?",
                    placeholder="Type your action here... (e.g., 'I examine the mysterious map')",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("🎲 Take Action", variant="primary", scale=1)
            
            gr.Markdown(
                """
                ### 💡 Example Actions:
                - "I approach the hooded figure cautiously"
                - "I examine the map carefully"
                - "I order an ale from the bartender"
                - "I draw my sword and prepare for battle"
                """
            )
        
        with gr.Column(scale=1):
            image_output = gr.Image(
                label="Scene Visualization",
                height=400,
                show_label=True
            )
            
            new_game_btn = gr.Button("🔄 Start New Game", variant="secondary")
            
            gr.Markdown(
                """
                ### 📊 Game Info
                - **Turn**: Updates with each action
                - **Images**: Auto-saved to `generated_images/`
                - **Speed**: ~30-60 seconds per turn
                """
            )
    
    # Event handlers
    def submit_action(action, history):
        return game.process_action(action, history)
    
    submit_btn.click(
        fn=submit_action,
        inputs=[action_input, chatbot],
        outputs=[chatbot, image_output]
    ).then(
        fn=lambda: "",
        outputs=action_input
    )
    
    action_input.submit(
        fn=submit_action,
        inputs=[action_input, chatbot],
        outputs=[chatbot, image_output]
    ).then(
        fn=lambda: "",
        outputs=action_input
    )
    
    new_game_btn.click(
        fn=game.start_new_game,
        outputs=[chatbot, image_output]
    )
    
    # Start with initial scene
    demo.load(
        fn=game.start_new_game,
        outputs=[chatbot, image_output]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Starting AI Fantasy RPG Web Interface...".center(60))
    print("="*60)
    print("\n🌐 The game will open in your web browser")
    print("📱 You can also access it from your phone on the same network!\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        inbrowser=True
    )
