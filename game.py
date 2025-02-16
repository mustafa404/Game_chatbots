import pygame
from pygame.locals import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

# Initialize Pygame
pygame.init()

# Screen Settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AI Chat Bot NPCs with LLMs")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Player Settings
player_size = 50
player_x = 100
player_y = 100
player_speed = 1

# NPC Settings
npc1_size = 50  # Flan-T5 NPC
npc1_x = 50
npc1_y = 50

npc2_size = 50  # DistilGPT-2 NPC
npc2_x = SCREEN_WIDTH - 100
npc2_y = 50

# Dialogue Settings
font = pygame.font.SysFont(None, 30)
dialogue = []
input_text = ""
show_dialogue = False
active_npc = None

# Load Flan-T5 Model and Tokenizer
model_name_t5 = "google/flan-t5-small" 
tokenizer_t5 = AutoTokenizer.from_pretrained(model_name_t5)
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name_t5)
model_t5.eval()

# Load DistilGPT-2 Model and Tokenizer
model_name_gpt2 = "distilgpt2"
tokenizer_gpt2 = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2 = AutoModelForCausalLM.from_pretrained(model_name_gpt2)
model_gpt2.eval()

# Use GPU if available
device = torch.device('cpu')
model_t5.to(device)
model_gpt2.to(device)

# Function for Flan-T5 Response
def get_response_t5(prompt):
    input_text = f"User: {prompt} NPC:"
    input_ids = tokenizer_t5.encode(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model_t5.generate(
            input_ids,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
    
    response = tokenizer_t5.decode(output_ids[0], skip_special_tokens=True)
    clean_output = response.split("NPC:")[0].strip()

    return clean_output

# Function for DistilGPT-2 Response
def get_response_gpt2(prompt):
    input_ids = tokenizer_gpt2.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model_gpt2.generate(
            input_ids,
            max_length=25,
            num_beams=4,
            early_stopping=False
        )
    
    response = tokenizer_gpt2.decode(output_ids[0], skip_special_tokens=True)
    return response

# Game Loop
running = True
while running:
    screen.fill(WHITE)
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        
        if event.type == KEYDOWN:
            if event.key == K_e and not show_dialogue:
                # Check proximity to NPCs
                if abs(player_x - npc1_x) < 60 and abs(player_y - npc1_y) < 60:
                    show_dialogue = True
                    active_npc = "t5"
                elif abs(player_x - npc2_x) < 60 and abs(player_y - npc2_y) < 60:
                    show_dialogue = True
                    active_npc = "gpt2"
            
            elif event.key == K_RETURN and show_dialogue:
                # Send player input to the active NPC's LLM and get response
                if input_text.strip() != "":
                    dialogue.append(f"You: {input_text}")
                    if active_npc == "t5":
                        npc_response = get_response_t5(input_text)
                    elif active_npc == "gpt2":
                        npc_response = get_response_gpt2(input_text)
                    dialogue.append(f"NPC: {npc_response}")
                    input_text = ""
            
            elif event.key == K_ESCAPE:
                # End conversation
                show_dialogue = False
                input_text = ""
                dialogue.clear()

            elif show_dialogue:
                # Handle text input
                if event.key == K_BACKSPACE:
                    input_text = input_text[:-1]
                else:
                    input_text += event.unicode
    
    # Player Movement
    keys = pygame.key.get_pressed()
    if keys[K_LEFT] and player_x > 0:
        player_x -= player_speed
    if keys[K_RIGHT] and player_x < SCREEN_WIDTH - player_size:
        player_x += player_speed
    if keys[K_UP] and player_y > 0:
        player_y -= player_speed
    if keys[K_DOWN] and player_y < SCREEN_HEIGHT - player_size:
        player_y += player_speed

    
    # Draw Player and NPCs
    pygame.draw.rect(screen, BLUE, (player_x, player_y, player_size, player_size))  # Player
    pygame.draw.rect(screen, GREEN, (npc1_x, npc1_y, npc1_size, npc1_size))  # Flan-T5 NPC
    pygame.draw.rect(screen, RED, (npc2_x, npc2_y, npc2_size, npc2_size))   # DistilGPT-2 NPC
    
    # Draw Dialogue Box
    if show_dialogue:
        pygame.draw.rect(screen, WHITE, (50, 400, 700, 180))
        pygame.draw.rect(screen, BLACK, (50, 400, 700, 180), 2)
        
        # Display Conversation History
        y_offset = 420
        for line in dialogue[-4:]:  
            text = font.render(line, True, BLACK)
            screen.blit(text, (60, y_offset))
            y_offset += 30
        
        # Display Input Text
        text = font.render(f"> {input_text}", True, BLACK)
        screen.blit(text, (60, 550))
    
    pygame.display.update()

pygame.quit()
pygame.display.quit()
pygame.font.quit()
pygame.mixer.quit()
