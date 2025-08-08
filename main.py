#!/usr/bin/env python3
"""
Vietnamese Law Chatbot - Main Application
"""

import argparse
import sys
from pathlib import Path

from src.config.settings import get_config, configure_gpu_settings, setup_environment
from src.utils.gpu_utils import ensure_gpu_available, print_gpu_stats
from src.model.model_loader import (
    prepare_model_for_training, 
    save_model_locally,
    push_model_to_hub
)
from src.data.dataset import prepare_training_data
from src.training.trainer import train_model
from src.inference.inference import  ChatbotInference , run_interactive_chat
#add
from unsloth import FastLanguageModel
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage
from langchain.schema import HumanMessage 


def train_chatbot():
    """Train the Vietnamese Law Chatbot."""
    print("=== Vietnamese Law Chatbot Training ===")
    
    # Get configuration
    config = get_config()
    
    # Check GPU availability
    gpu_name = ensure_gpu_available()
    print(f"Using GPU: {gpu_name}")
    
    # Configure GPU-specific settings
    configure_gpu_settings(config, gpu_name)
    
    # Setup environment
    setup_environment(config)
    
    print(f"Training parameters:")
    print(f"  Max steps: {config.training.max_steps}")
    print(f"  Load in 4bit: {config.model.load_in_4bit}")
    print(f"  Dataset: {config.training.dataset_id}")
    
    # Print initial GPU stats
    print_gpu_stats()
    
    # Prepare model for training
    print("\nLoading and configuring model...")
    model, tokenizer = prepare_model_for_training(config.model)
    
    # Prepare training data
    print("Loading and processing dataset...")
    train_dataset = prepare_training_data(config.training.dataset_id, tokenizer)
    
    # Train model
    print("Starting training...")
    trainer_stats = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config.training,
        api_config=config.api,
        max_seq_length=config.model.max_seq_length
    )
    
    print("Training completed!")
    print(f"Training stats: {trainer_stats}")
    
    # Save model locally
    print(f"\nSaving model locally as '{config.api.model_name}'...")
    save_model_locally(model, tokenizer, config.api.model_name)
    
    # Push to Hugging Face Hub if enabled
    if config.api.enable_hf:
        print("Pushing model to Hugging Face Hub...")
        push_model_to_hub(model, tokenizer, config.api)
    else:
        print("Hugging Face upload disabled (no token provided)")
    
    print("Training pipeline completed successfully!")


def run_inference(model_path: str = None):
    """Run inference with trained model."""
    print("=== Vietnamese Law Chatbot Inference ===")
    
    # Get configuration
    config = get_config()
    
    # Check GPU availability
    gpu_name = ensure_gpu_available()
    print(f"Using GPU: {gpu_name}")
    
    # Configure GPU-specific settings
    configure_gpu_settings(config, gpu_name)
    
    # Load model
    model_name = model_path or config.api.model_name
    print(f"Loading model from: {model_name}")
    
    try:
        # For local model, we need to load it differently
        if Path(model_name).exists():
            # Load from local path
    
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=config.model.max_seq_length,
                dtype=config.model.dtype,
                load_in_4bit=config.model.load_in_4bit,
            )
        else:
            # Load from HF Hub
            model, tokenizer = prepare_model_for_training(config.model)
        
        # Prepare for inference
        model = FastLanguageModel.for_inference(model)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens= 512 ,   #MAX_NEW_TOKENS
            temperature= 0.7 ,      #TEMPERATURE
            top_k= 50 ,             #TOP_K
            top_p= 0.9              #TOP_P
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create inference engine
        inference_engine = ChatbotInference( llm )
        
        # Run interactive chat
        run_interactive_chat(inference_engine)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model exists or run training first.")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Vietnamese Law Chatbot - Training and Inference"
    )
    parser.add_argument(
        "mode",
        choices=["train", "inference", "chat"],
        help="Mode to run: train the model or run inference"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model for inference (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "train":
            train_chatbot()
        elif args.mode in ["inference", "chat"]:
            run_inference(args.model_path)
        else:
            print("Invalid mode. Use 'train' or 'inference'")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
