#!/usr/bin/env python3
"""
chat.py – External Multi-Model Chat Client for CacheAI

This script connects to the CacheAI API Server (from app.py) and provides two modes:
1. Interactive Chat: Talk to a single loaded model.
2. Multi-Model Debate: Watch two models talk to each other, maintaining their own 
   conversational context from their respective perspectives.

Prerequisites:
- CacheAI server must be running (Start it from the CacheAI GUI).
- At least one model must be loaded via CacheAI for Mode 1.
- At least two models must be loaded via CacheAI for Mode 2.
- `requests` library installed (pip install requests).
"""

import requests
import json
import sys
import time
import logging

# Configuration: Match this to your CacheAI server settings
BASE_URL = "http://localhost:8081"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CacheAI-Chat")

# ANSI color codes for terminal readability
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"


class CacheAIClient:
    """Client to interact with the CacheAI OpenAI-compatible API."""

    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list:
        """Fetch the list of currently loaded models from the server."""
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5)
            r.raise_for_status()
            data = r.json().get("data", [])
            models = [m["id"] for m in data]
            logger.info(f"Discovered loaded models: {models}")
            return models
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to CacheAI server at {self.base_url}.")
            print(f"{Colors.RED}Error: Cannot connect to CacheAI server at {self.base_url}.{Colors.RESET}")
            print(f"{Colors.GREY}Make sure the CacheAI GUI is running and the API Server is started.{Colors.RESET}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            print(f"{Colors.RED}Error fetching models: {e}{Colors.RESET}")
            return []

    def chat(self, model_id: str, messages: list, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Send a chat completion request to a specific model."""
        try:
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            logger.debug(f"Sending payload to {model_id}: {messages[-1]['content'][:50]}...")
            # LLM generation can take a while, set a long timeout
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=300.0)
            r.raise_for_status()
            response_text = r.json()["choices"][0]["message"]["content"]
            logger.debug(f"Received response from {model_id}: {response_text[:50]}...")
            return response_text
        except requests.exceptions.Timeout:
            logger.warning(f"Request to {model_id} timed out.")
            return "[Error: Request timed out. The model might be taking too long to generate.]"
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            return f"[Error: {str(e)}]"


def print_header():
    """Print a fancy header."""
    print(f"\n{Colors.CYAN}{'='*50}")
    print(f"{Colors.BOLD} CacheAI External Chat Client{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*50}{Colors.RESET}\n")


def get_model_choice(client: CacheAIClient, prompt: str) -> str:
    """Helper to list models and let the user pick one."""
    models = client.list_models()
    if not models:
        print(f"{Colors.RED}No models loaded! Please load models in the CacheAI GUI first.{Colors.RESET}")
        logger.error("No models available for selection.")
        sys.exit(1)
    
    print(f"{Colors.YELLOW}{prompt}{Colors.RESET}")
    for i, m in enumerate(models, 1):
        print(f"  {Colors.BOLD}{i}.{Colors.RESET} {m}")
    
    while True:
        try:
            choice = int(input(f"{Colors.GREEN}Enter number: {Colors.RESET}"))
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                logger.info(f"User selected model: {selected_model}")
                return selected_model
            print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")


def mode_single_chat(client: CacheAIClient):
    """Standard 1-on-1 chat with a single model."""
    model_id = get_model_choice(client, "Select a model to chat with:")
    print(f"\n{Colors.CYAN}Chatting with {Colors.BOLD}{model_id}{Colors.RESET}. Type 'quit' to exit.\n")
    logger.info(f"Starting single chat session with {model_id}")
    
    messages = []
    
    while True:
        user_input = input(f"{Colors.GREEN}You: {Colors.RESET}")
        if user_input.strip().lower() in ['quit', 'exit']:
            logger.info("Exiting single chat mode.")
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print(f"{Colors.YELLOW}{model_id}: {Colors.RESET}", end="", flush=True)
        response_text = client.chat(model_id, messages)
        print(response_text)
        
        messages.append({"role": "assistant", "content": response_text})


def mode_multi_model_debate(client: CacheAIClient):
    """Two models talk to each other, maintaining their own context windows."""
    print(f"\n{Colors.MAGENTA}--- Multi-Model Conversation Mode ---{Colors.RESET}")
    model_a = get_model_choice(client, "Select Model A (The Initiator):")
    model_b = get_model_choice(client, "Select Model B (The Responder):")
    
    topic = input(f"\n{Colors.GREEN}Enter the topic/start message for the conversation: {Colors.RESET}")
    rounds = 5
    try:
        rounds = int(input(f"How many exchange rounds? (default 5): ") or "5")
    except ValueError:
        rounds = 5

    logger.info(f"Starting debate: {model_a} vs {model_b} on topic: '{topic}' for {rounds} rounds.")
    print(f"\n{Colors.CYAN}Starting debate between {Colors.BOLD}{model_a}{Colors.RESET}{Colors.CYAN} and {Colors.BOLD}{model_b}{Colors.RESET}{Colors.CYAN}...{Colors.RESET}\n")
    
    # We maintain separate history lists for each model.
    # Because Model A sees Model B as 'user', and vice versa.
    history_a = []
    history_b = []
    
    current_message = topic

    for i in range(rounds * 2):
        is_model_a_turn = (i % 2 == 0)
        active_model = model_a if is_model_a_turn else model_b
        active_history = history_a if is_model_a_turn else history_b
        
        # Append the other model's output as the user message for the current model
        active_history.append({"role": "user", "content": current_message})
        
        # Add a system prompt to guide the multi-agent behavior
        if len(active_history) == 1:
            active_history.insert(0, {"role": "system", "content": "You are engaging in a conversation with another AI. Respond naturally and stay on topic."})

        logger.info(f"Round {(i//2)+1} | Requesting generation from {active_model}...")
        print(f"{Colors.GREY}[Round {(i//2)+1} | {active_model} thinking...]{Colors.RESET}")
        response = client.chat(active_model, active_history)
        
        # Save the response in the active model's history as 'assistant'
        active_history.append({"role": "assistant", "content": response})
        
        # Print the response
        color = Colors.BLUE if is_model_a_turn else Colors.MAGENTA
        print(f"{color}{active_model}: {Colors.RESET}{response}\n")
        
        # The response becomes the next message for the other model
        current_message = response

    logger.info("Multi-model conversation completed.")
    print(f"{Colors.CYAN}Conversation ended after {rounds} rounds.{Colors.RESET}")


def main():
    print_header()
    client = CacheAIClient(BASE_URL)

    while True:
        print(f"\n{Colors.BOLD}Select Mode:{Colors.RESET}")
        print(f"  1. Single Model Chat")
        print(f"  2. Multi-Model Conversation (Debate)")
        print(f"  3. Refresh Models")
        print(f"  4. Exit")
        
        choice = input(f"{Colors.GREEN}Enter choice (1-4): {Colors.RESET}").strip()
        
        if choice == '1':
            mode_single_chat(client)
        elif choice == '2':
            mode_multi_model_debate(client)
        elif choice == '3':
            models = client.list_models()
            print(f"{Colors.CYAN}Currently loaded models: {models if models else 'None'}{Colors.RESET}")
        elif choice == '4' or choice.lower() == 'quit':
            print(f"{Colors.GREY}Goodbye!{Colors.RESET}")
            logger.info("Chat client exiting.")
            break
        else:
            print(f"{Colors.RED}Invalid choice.{Colors.RESET}")


if __name__ == "__main__":
    main()