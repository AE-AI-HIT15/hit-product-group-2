# Vietnamese Law Chatbot

A modular application for fine-tuning and deploying a Vietnamese law chatbot using Unsloth and LoRA.

## Project Structure

```
finetune/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # Dataset loading and processing
│   ├── model/
│   │   ├── __init__.py
│   │   └── model_loader.py      # Model loading and PEFT configuration
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py           # Training pipeline
│   ├── inference/
│   │   ├── __init__.py
│   │   └── inference.py         # Inference and chat interface
│   └── utils/
│       ├── __init__.py
│       └── gpu_utils.py         # GPU utilities
├── main.py                      # Main application entry point
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export HF_TOKEN="your_huggingface_token"
export COMET_API_KEY="your_comet_api_key"
export COMET_PROJECT_NAME="your_project_name"
```

## Usage

### Training

Train the model:
```bash
python main.py train
```

This will:
- Load the base model (T-VisStar-7B-v0.1)
- Configure LoRA parameters
- Load and process the Vietnamese law dataset
- Train the model
- Save the model locally
- Push to Hugging Face Hub (if token provided)

### Inference

Run interactive chat:
```bash
python main.py chat
```

Or:
```bash
python main.py inference
```

Use a specific model path:
```bash
python main.py inference --model-path /path/to/model
```

## Configuration

The application uses dataclasses for configuration management:

- `ModelConfig`: Base model, LoRA parameters, sequence length
- `TrainingConfig`: Training parameters, dataset settings
- `InferenceConfig`: Inference parameters
- `APIConfig`: API tokens and settings

Configuration can be customized by modifying `src/config/settings.py` or setting environment variables.

## Features

- **Modular Design**: Clear separation of concerns
- **GPU Auto-detection**: Automatically configures settings based on available GPU
- **Flexible Configuration**: Easy to customize via dataclasses
- **Interactive Chat**: Real-time conversation interface
- **Model Saving**: Local and Hugging Face Hub support
- **Logging**: Comet ML integration for experiment tracking

## Model Details

- **Base Model**: T-VisStar-7B-v0.1
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: Vietnamese law Q&A dataset
- **Prompt Format**: Alpaca-style prompts in Vietnamese

## Requirements

- NVIDIA GPU with CUDA support
- Python 3.8+
- See `requirements.txt` for package dependencies