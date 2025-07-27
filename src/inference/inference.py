from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer
import torch
from typing import Optional

from ..config.settings import ALPACA_PROMPT, InferenceConfig


class ChatbotInference:
    """Vietnamese Law Chatbot Inference class."""
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        config: InferenceConfig
    ):
        """Initialize inference class.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            config: Inference configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.text_streamer = TextStreamer(tokenizer) if config.streaming else None
    
    def format_prompt(self, instruction: str) -> str:
        """Format instruction using the Alpaca prompt template.
        
        Args:
            instruction: User instruction/question
            
        Returns:
            Formatted prompt
        """
        return ALPACA_PROMPT.format(instruction, "")
    
    def generate_response(
        self, 
        instruction: str, 
        streaming: Optional[bool] = None, 
        trim_input_message: bool = False
    ) -> str:
        """Generate response for given instruction.
        
        Args:
            instruction: User instruction/question
            streaming: Whether to use streaming (overrides config if provided)
            trim_input_message: Whether to trim input message from output
            
        Returns:
            Generated response
        """
        message = self.format_prompt(instruction)
        inputs = self.tokenizer([message], return_tensors="pt").to("cuda")
        
        use_streaming = streaming if streaming is not None else self.config.streaming
        
        if use_streaming and self.text_streamer:
            output_tokens = self.model.generate(
                **inputs, 
                streamer=self.text_streamer, 
                max_new_tokens=self.config.max_new_tokens, 
                use_cache=self.config.use_cache
            )
        else:
            output_tokens = self.model.generate(
                **inputs, 
                max_new_tokens=self.config.max_new_tokens, 
                use_cache=self.config.use_cache
            )
        
        output = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        
        if trim_input_message:
            return output[len(message):]
        else:
            return output
    
    def chat(self, instruction: str) -> str:
        """Simple chat interface.
        
        Args:
            instruction: User instruction/question
            
        Returns:
            Chatbot response
        """
        return self.generate_response(instruction, streaming=False, trim_input_message=True)


def create_inference_engine(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    config: InferenceConfig
) -> ChatbotInference:
    """Create inference engine.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Inference configuration
        
    Returns:
        ChatbotInference instance
    """
    return ChatbotInference(model, tokenizer, config)


def run_interactive_chat(inference_engine: ChatbotInference) -> None:
    """Run interactive chat session.
    
    Args:
        inference_engine: Inference engine
    """
    print("Vietnamese Law Chatbot - Interactive Mode")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.strip():
            try:
                response = inference_engine.chat(user_input)
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please enter a question.")


def generate_single_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    instruction: str,
    config: InferenceConfig
) -> str:
    """Generate a single response without creating an inference engine.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        instruction: User instruction
        config: Inference configuration
        
    Returns:
        Generated response
    """
    inference_engine = create_inference_engine(model, tokenizer, config)
    return inference_engine.chat(instruction)