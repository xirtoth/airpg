"""
AI Fantasy RPG Game
An interactive text-based RPG powered by LLM and Stable Diffusion
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import datetime
import config


class AIRPGGame:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cpu":
            print("WARNING: CUDA not available. This will be very slow!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
        
        self.llm_model = None
        self.tokenizer = None
        self.diffusion_pipe = None
        self.conversation_history = []
        self.turn_count = 0
        
        # Create images directory
        if config.SAVE_IMAGES:
            os.makedirs(config.IMAGES_DIR, exist_ok=True)
        
    def load_llm(self):
        """Load the language model for game narration."""
        print("\n Loading LLM model (this may take a minute)...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL,
                trust_remote_code=True
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            print("✓ LLM model loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("\nTrying alternative model...")
            # Fallback to a simpler model
            try:
                model_name = "facebook/opt-1.3b"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                ).to(self.device)
                print("✓ Alternative LLM model loaded!")
            except Exception as e2:
                print(f"Failed to load any LLM model: {e2}")
                sys.exit(1)
    
    def load_diffusion(self):
        """Load Stable Diffusion for image generation."""
        print("\n Loading Stable Diffusion model...")
        try:
            self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
                config.DIFFUSION_MODEL,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimize for lower VRAM
            self.diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.diffusion_pipe.scheduler.config
            )
            self.diffusion_pipe = self.diffusion_pipe.to(self.device)
            self.diffusion_pipe.enable_attention_slicing()
            
            # Enable memory efficient attention if available
            try:
                self.diffusion_pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
            
            print("✓ Stable Diffusion loaded successfully!")
        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            print("Continuing without image generation...")
            self.diffusion_pipe = None
    
    def generate_story_response(self, player_action):
        """Generate the next part of the story based on player action."""
        # Build the prompt for the LLM
        system_prompt = """You are a creative dungeon master for a fantasy RPG game. 
Describe what happens in vivid detail (2-3 paragraphs). 
Include sensory details, atmosphere, and consequences of actions.
Keep responses engaging and immersive."""

        # Format conversation history
        context = ""
        if self.conversation_history:
            for entry in self.conversation_history[-3:]:  # Last 3 turns for context
                context += f"Player: {entry['action']}\n{entry['response']}\n\n"
        
        prompt = f"""{system_prompt}

{context}Player action: {player_action}

Describe what happens next:"""

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs.input_ids,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if "Describe what happens next:" in response:
            response = response.split("Describe what happens next:")[-1].strip()
        
        return response
    
    def generate_scene_image(self, scene_description):
        """Generate an image based on the scene description."""
        if not self.diffusion_pipe:
            print("\n[Image generation not available]")
            return None
        
        print("\n Generating scene image...")
        
        # Create a concise image prompt from the scene description
        prompt_parts = scene_description.split('.')[:2]  # First 2 sentences
        image_prompt = ' '.join(prompt_parts)
        
        # Add style keywords for better fantasy RPG images
        enhanced_prompt = f"fantasy RPG scene, {image_prompt}, detailed, atmospheric, digital art"
        
        try:
            # Generate image
            with torch.no_grad():
                image = self.diffusion_pipe(
                    enhanced_prompt,
                    num_inference_steps=config.INFERENCE_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    height=config.IMAGE_HEIGHT,
                    width=config.IMAGE_WIDTH
                ).images[0]
            
            # Save image
            if config.SAVE_IMAGES:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config.IMAGES_DIR}/scene_{self.turn_count:03d}_{timestamp}.png"
                image.save(filename)
                print(f"✓ Image saved: {filename}")
                
                # Try to display the image
                try:
                    image.show()
                except:
                    print("  (Could not auto-display image)")
            
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def start_game(self):
        """Initialize and start the game."""
        print("=" * 60)
        print(f"  {config.GAME_TITLE}".center(60))
        print("=" * 60)
        print("\nInitializing game systems...")
        
        self.load_llm()
        self.load_diffusion()
        
        print("\n" + "=" * 60)
        print("  GAME READY".center(60))
        print("=" * 60)
        
        # Initial scene
        print("\n" + "=" * 60)
        print("  THE ADVENTURE BEGINS".center(60))
        print("=" * 60)
        
        initial_scene = """You awaken in a dimly lit tavern. The smell of ale and roasted meat fills the air. 
A hooded figure in the corner watches you intently. The bartender, a burly dwarf, 
polishes a mug while eyeing you curiously. Your hand instinctively reaches for the 
sword at your side. You notice a mysterious map on your table."""
        
        print(f"\n{initial_scene}\n")
        
        # Generate initial image
        self.generate_scene_image(initial_scene)
        
        self.conversation_history.append({
            'action': 'START',
            'response': initial_scene
        })
        
        self.game_loop()
    
    def game_loop(self):
        """Main game loop."""
        print("\n" + "-" * 60)
        print("Type 'quit' or 'exit' to end the game")
        print("Type 'help' for tips")
        print("-" * 60)
        
        while True:
            print("\n" + "=" * 60)
            self.turn_count += 1
            print(f"  Turn {self.turn_count}".center(60))
            print("=" * 60)
            
            # Get player input
            player_action = input("\nWhat do you do? > ").strip()
            
            if not player_action:
                print("Please enter an action.")
                continue
            
            if player_action.lower() in ['quit', 'exit']:
                print("\nThanks for playing! Your adventure ends here.")
                break
            
            if player_action.lower() == 'help':
                self.show_help()
                continue
            
            # Generate story response
            print("\n[AI is thinking...]")
            try:
                response = self.generate_story_response(player_action)
                
                print("\n" + "-" * 60)
                print(response)
                print("-" * 60)
                
                # Save to history
                self.conversation_history.append({
                    'action': player_action,
                    'response': response
                })
                
                # Generate scene image
                self.generate_scene_image(response)
                
            except Exception as e:
                print(f"\nError generating response: {e}")
                print("Please try again.")
    
    def show_help(self):
        """Show help information."""
        print("\n" + "=" * 60)
        print("  HELP".center(60))
        print("=" * 60)
        print("""
This is an AI-powered RPG where you can do anything!

Tips:
- Be specific with your actions (e.g., "I examine the map carefully")
- Try dialogue: "I ask the bartender about the hooded figure"
- Explore: "I walk over to the mysterious hooded figure"
- Combat: "I draw my sword and challenge them to a duel"
- Investigate: "I search the room for hidden passages"

The AI will respond to any action you describe. Be creative!
        """)
        print("=" * 60)


def main():
    """Main entry point."""
    try:
        game = AIRPGGame()
        game.start_game()
    except KeyboardInterrupt:
        print("\n\nGame interrupted. Goodbye!")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
