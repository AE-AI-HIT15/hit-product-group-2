from datasets import load_dataset, Dataset
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

from ..config.settings import ALPACA_PROMPT


def formatting_prompts_func(examples: Dict[str, List[Any]], tokenizer: PreTrainedTokenizer) -> Dict[str, List[str]]:
    """Format examples using the Alpaca prompt template.
    
    Args:
        examples: Dictionary containing 'question' and 'context' keys
        tokenizer: Tokenizer to get the EOS token
        
    Returns:
        Dictionary with formatted 'text' key
    """
    inputs = examples["question"]
    outputs = examples["context"]
    texts = []
    
    for input_text, output in zip(inputs, outputs):
        # Handle context if it's a list
        if isinstance(output, list):
            output_text = " ".join(output)
        else:
            output_text = output

        # Format prompt and add EOS token
        formatted_text = ALPACA_PROMPT.format(
            input_text.strip(), 
            output_text.strip()
        ) + tokenizer.eos_token
        texts.append(formatted_text)

    return {"text": texts}


def load_and_process_dataset(dataset_id: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Load and process the dataset for training.
    
    Args:
        dataset_id: HuggingFace dataset identifier
        tokenizer: Tokenizer for formatting
        
    Returns:
        Processed dataset
    """
    dataset = load_dataset(dataset_id)
    
    # Apply formatting function
    processed_dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True,
    )
    
    return processed_dataset


def prepare_training_data(dataset_id: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Prepare training data from dataset.
    
    Args:
        dataset_id: HuggingFace dataset identifier
        tokenizer: Tokenizer for formatting
        
    Returns:
        Training dataset
    """
    dataset = load_and_process_dataset(dataset_id, tokenizer)
    return dataset["train"]


def prepare_validation_data(dataset_id: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    """Prepare validation data from dataset if available.
    
    Args:
        dataset_id: HuggingFace dataset identifier
        tokenizer: Tokenizer for formatting
        
    Returns:
        Validation dataset or None if not available
    """
    dataset = load_and_process_dataset(dataset_id, tokenizer)
    
    if "validation" in dataset:
        return dataset["validation"]
    elif "test" in dataset:
        return dataset["test"]
    else:
        return None