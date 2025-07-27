from transformers import TrainingArguments, PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from datasets import Dataset
from typing import Dict, Any

from ..config.settings import TrainingConfig, APIConfig


def create_training_arguments(config: TrainingConfig, api_config: APIConfig) -> TrainingArguments:
    """Create training arguments from configuration.
    
    Args:
        config: Training configuration
        api_config: API configuration for logging
        
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        output_dir=config.output_dir,
        report_to="comet_ml" if api_config.enable_comet else "none",
    )


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    config: TrainingConfig,
    api_config: APIConfig,
    max_seq_length: int
) -> SFTTrainer:
    """Create SFT trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        config: Training configuration
        api_config: API configuration
        max_seq_length: Maximum sequence length
        
    Returns:
        SFTTrainer instance
    """
    training_args = create_training_arguments(config, api_config)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        packing=config.packing,
        args=training_args,
    )
    
    return trainer


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    config: TrainingConfig,
    api_config: APIConfig,
    max_seq_length: int
) -> Dict[str, Any]:
    """Train the model.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        config: Training configuration
        api_config: API configuration
        max_seq_length: Maximum sequence length
        
    Returns:
        Training statistics
    """
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
        api_config=api_config,
        max_seq_length=max_seq_length
    )
    
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed!")
    
    return trainer_stats