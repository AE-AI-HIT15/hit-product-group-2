from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple
from huggingface_hub import HfApi
#add
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


from ..config.settings import ModelConfig, APIConfig


def load_llm_pipeline(model_name: str, config: ModelConfig ):
    """Load model and setup conversation chain"""
    #global tokenizer, model, pipe, llm
    try:
        # Load model and tokenizer
        model, tokenizer = load_model(model_name, config)
        # Prepare model for inference
        model = FastLanguageModel.for_inference(model)
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens= 512 ,   #MAX_NEW_TOKENS
            temperature= 0.7 ,      #TEMPERATURE
            top_k= 50 ,             #TOP_K
            top_p= 0.9              #TOP_P
        )
        
        # Setup LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return llm
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def load_model(model_name: str, config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the base language model and tokenizer.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    
    return model, tokenizer

def configure_peft_model(model: PreTrainedModel, config: ModelConfig) -> PreTrainedModel:
    """Configure PEFT (LoRA) for the model.
    
    Args:
        model: Base model
        config: Model configuration
        
    Returns:
        PEFT-configured model
    """
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.random_state,
        use_rslora=config.use_rslora,
        loftq_config=config.loftq_config,
    )
    
    return peft_model


def prepare_model_for_training(config: ModelConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Prepare model and tokenizer for training.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (configured_model, tokenizer)
    """
    # Load base model
    model, tokenizer = load_model(config.base_model, config)
    
    # Configure PEFT
    model = configure_peft_model(model, config)
    
    return model, tokenizer


# def prepare_model_for_inference(model: PreTrainedModel) -> PreTrainedModel:
#     """Prepare model for inference.
    
#     Args:
#         model: Trained model
        
#     Returns:
#         Model optimized for inference
#     """
#     FastLanguageModel.for_inference(model)
#     return model


def save_model_locally(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    model_name: str,
    save_method: str = "merged_16bit"
) -> None:
    """Save model locally.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        model_name: Name for saving
        save_method: Saving method
    """
    model.save_pretrained_merged(
        model_name,
        tokenizer,
        save_method=save_method,
    )


def push_model_to_hub(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: APIConfig,
    save_method: str = "merged_16bit"
) -> None:
    """Push model to Hugging Face Hub.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: API configuration
        save_method: Saving method
    """
    if not config.enable_hf:
        print("Hugging Face token not available. Skipping upload.")
        return
    
    api = HfApi()
    user_info = api.whoami(token=config.hf_token)
    huggingface_user = user_info["name"]
    
    repo_name = f"{huggingface_user}/{config.model_name}"
    print(f"Pushing model to: {repo_name}")
    
    model.push_to_hub_merged(
        repo_name,
        tokenizer=tokenizer,
        save_method=save_method,
        token=config.hf_token,
    )