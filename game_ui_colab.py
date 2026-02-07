"""
AI Fantasy RPG Game - Web UI Version
An interactive RPG with a beautiful web interface powered by Gradio
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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
        
        cache_dir = "D:/huggingface_cache"
        
        # Load LLM (simpler loading for smaller models)
        try:
            print(f"Loading LLM: {config.LLM_MODEL}")
            print(f"Cache directory: {cache_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLM_MODEL,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                cache_dir=cache_dir
            )
            print("✓ LLM loaded successfully!")
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise
        
        # Load Stable Diffusion
        try:
            print(f"Loading Stable Diffusion: {config.DIFFUSION_MODEL}")
            self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
                config.DIFFUSION_MODEL,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
                cache_dir=cache_dir
            )
            
            self.diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.diffusion_pipe.scheduler.config
            )
            self.diffusion_pipe = self.diffusion_pipe.to(self.device)
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
        
        # Build conversation history with last 3 turns for better memory
        conversation_context = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 turns
            for i, entry in enumerate(recent_history):
                action = entry['action']
                response = entry['response'][:200]  # Keep more context
                if action != "START":
                    conversation_context += f"Player: {action}\nStory: {response}\n\n"
                else:
                    conversation_context += f"Story: {response}\n\n"
        
        # TinyLlama chat format with full context
        if conversation_context:
            messages = f"""<|system|>
You are a fantasy RPG storyteller. Continue the story based on what happened before and the player's new action. Keep consistency with previous events. Write 2-3 sentences.</s>
<|user|>
Previous story:
{conversation_context}
Player's new action: {player_action}

What happens next?</s>
<|assistant|>
"""
        else:
            messages = f"""<|system|>
You are a fantasy RPG storyteller. Describe what happens in 2-3 sentences.</s>
<|user|>
{player_action}</s>
<|assistant|>
"""

        inputs = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=768).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs.input_ids,
                max_new_tokens=250,  # Increased for complete responses
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id else 0,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("</s>")[0]] if self.tokenizer.eos_token_id else [0],
                repetition_penalty=1.15,
                no_repeat_ngram_size=3  # Prevent repetitive phrases
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up - extract only assistant response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        if "<|user|>" in response:
            response = response.split("<|user|>")[0].strip()
        if "</s>" in response:
            response = response.split("</s>")[0].strip()
        
        # Remove any remaining prompt text and meta-phrases
        meta_phrases = [
            "Player action:", "What happens next?", "Here's", "Here is",
            "Updated version:", "Continuation:", "Updated:", "Version:"
        ]
        for phrase in meta_phrases:
            if phrase in response:
                parts = response.split(phrase)
                # Take the part after the meta-phrase if it's at the start
                if len(parts) > 1 and len(parts[0]) < 20:
                    response = parts[1].strip()
                    break
        
        # Clean up incomplete sentences at the end
        if response and not response[-1] in '.!?"':
            # Find last complete sentence
            last_period = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_period > 50:  # Only truncate if we have enough text
                response = response[:last_period + 1]
        
        # Ensure minimum length
        if len(response) < 50:
            response = f"As you {player_action}, the world around you shifts. The air grows thick with anticipation. Something significant is about to happen."
        
        return response
    
    def generate_image_prompt(self, scene_description):
        """Extract visual keywords directly from the story text."""
        
        # Blacklist of non-visual words
        blacklist = {
            'you', 'your', 'the', 'and', 'with', 'that', 'this', 'from', 'into', 
            'are', 'is', 'was', 'were', 'has', 'have', 'had', 'will', 'would', 
            'could', 'should', 'can', 'may', 'might', 'must', 'shall',
            'player', 'continue', 'story', 'where', 'they', 'them', 'their',
            'left', 'scene', 'between', 'keywords', 'here', 'there', 'when',
            'what', 'how', 'why', 'which', 'who', 'whose', 'whom',
            'description', 'visual', 'words', 'prompt', 'happens', 'next'
        }
        
        # Extract words from first 2-3 sentences
        sentences = scene_description.split('.')[:3]
        text = ' '.join(sentences).lower()
        
        # Remove punctuation and split
        for char in ',.!?;:"\'-()[]{}':
            text = text.replace(char, ' ')
        
        words = text.split()
        
        # Filter: keep nouns/adjectives (words 4+ chars, not in blacklist, not too common)
        keywords = []
        for word in words:
            if (len(word) >= 4 and 
                word not in blacklist and
                not word.endswith('ing') and  # Remove gerunds
                not word.endswith('ly')):     # Remove adverbs
                if word not in keywords:  # Avoid duplicates
                    keywords.append(word)
                if len(keywords) >= 10:  # Max 10 keywords
                    break
        
        # If we got too few keywords, add some from location detection
        if len(keywords) < 4:
            locations = ['tavern', 'forest', 'castle', 'cave', 'dungeon', 'mountain', 
                        'beach', 'market', 'city', 'village', 'tower', 'temple']
            for loc in locations:
                if loc in scene_description.lower():
                    keywords.insert(0, loc)
                    break
        
        # Ensure we have something
        if len(keywords) < 2:
            keywords = ['fantasy', 'medieval', 'adventure', 'scene']
        
        return ', '.join(keywords[:10])
    
    def generate_scene_image(self, scene_description, progress=gr.Progress()):
        """Generate an image based on the scene description."""
        if not self.diffusion_pipe:
            return None
        
        # Clear VRAM cache before first generation
        if self.turn_count == 0:
            torch.cuda.empty_cache()
        
        progress(0.6, desc="Creating image prompt...")
        
        # Use LLM to extract keywords from its own story
        keywords = self.generate_image_prompt(scene_description)
        
        # Add fantasy RPG style
        enhanced_prompt = f"fantasy RPG art, {keywords}, detailed digital painting, dramatic lighting, high quality"
        
        # Negative prompt to avoid common issues
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, text, watermark, signature, amateur"
        
        # Print the prompt to console for debugging
        print("\n" + "="*60)
        print("🎨 IMAGE PROMPT:")
        print(f"Positive: {enhanced_prompt}")
        print(f"Negative: {negative_prompt}")
        print("="*60 + "\n")
        
        progress(0.7, desc="Generating scene image...")
        
        try:
            with torch.no_grad():
                image = self.diffusion_pipe(
                    enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=config.INFERENCE_STEPS,
                    guidance_scale=config.GUIDANCE_SCALE,
                    height=config.IMAGE_HEIGHT,
                    width=config.IMAGE_WIDTH
                ).images[0]
            
            if config.SAVE_IMAGES:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{config.IMAGES_DIR}/scene_{self.turn_count:03d}_{timestamp}.png"
                image.save(filename)
            
            return image
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def process_action(self, player_action, history, progress=gr.Progress()):
        """Process player action and return updated chat history and image."""
        if not player_action or not player_action.strip():
            return history, self.current_image
        
        player_action = player_action.strip()
        self.turn_count += 1
        
        # Generate story response
        progress(0.1, desc="Processing your action...")
        response = self.generate_story_response(player_action, progress)
        
        # Save to history
        self.conversation_history.append({
            'action': player_action,
            'response': response
        })
        
        # Generate image
        self.current_image = self.generate_scene_image(response, progress)
        
        # Update chat history
        history.append((player_action, response))
        
        progress(1.0, desc="Done!")
        return history, self.current_image
    
    def start_new_game(self):
        """Start a new game with initial scene."""
        import random
        
        self.conversation_history = []
        self.turn_count = 0
        
        # Random starting scenarios
        starting_scenarios = [
            """You awaken in a dimly lit tavern. The smell of ale and roasted meat fills the air. 
A hooded figure in the corner watches you intently. The bartender, a burly dwarf, 
polishes a mug while eyeing you curiously. Your hand instinctively reaches for the 
sword at your side. You notice a mysterious map on your table.""",
            
            """You find yourself at the edge of a dark forest. Ancient trees loom overhead, their branches 
creating a canopy that blocks out most of the sunlight. Strange sounds echo from within the woods. 
A worn path leads deeper into the forest, while behind you lies a small village with smoke rising 
from its chimneys. In your pack, you feel the weight of a mystical amulet.""",
            
            """You stand before massive castle gates. Storm clouds gather overhead, and thunder rumbles in the 
distance. The guards at the gate eye you suspiciously. A royal messenger approaches, breathless and urgent. 
"You must come quickly," they say, "The king requires your assistance immediately." Your armor feels heavy, 
and your sword gleams with an unnatural light.""",
            
            """You awaken on a sandy beach, waves lapping at your feet. The wreckage of a ship lies scattered 
along the shore. Your clothes are soaked, but miraculously, your weapons remained strapped to your side. 
In the distance, you see smoke rising from what appears to be a jungle. A sealed bottle has washed up 
beside you, containing what looks like a treasure map.""",
            
            """You're in a bustling marketplace filled with exotic merchants and strange creatures. The air is 
thick with the scent of spices and magic. A street urchin bumps into you and quickly disappears into the 
crowd - you notice your coin purse is lighter. Suddenly, a cloaked figure grabs your arm and whispers, 
"They're looking for you. Follow me if you want to live." """
        ]
        
        initial_scene = random.choice(starting_scenarios)
        
        self.conversation_history.append({
            'action': 'START',
            'response': initial_scene
        })
        
        # Generate initial image
        self.current_image = self.generate_scene_image(initial_scene)
        
        return [("🎮 Game Started", initial_scene)], self.current_image


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
                autoscroll=True
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
