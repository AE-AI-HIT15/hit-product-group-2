from datasets import load_dataset, Dataset
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

ALPACA_PROMPT = """Dưới đây là hướng dẫn mô tả một nhiệm vụ, kết hợp với thông tin đầu vào cung cấp thêm ngữ cảnh. Hãy viết phản hồi hoàn thành yêu cầu một cách phù hợp.

### Hướng dẫn:
Bạn là một trợ lý thông minh, hãy trả lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan. Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
### Câu hỏi:
{}

### Trả lời:
{}"""


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