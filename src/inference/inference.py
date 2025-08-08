import torch
from typing import Optional
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

from ..config.settings import count_chat, conversation_chains
from ..utils.postprocessing import processor_text


class ChatbotInference:
    """Vietnamese Law Chatbot Inference class."""

    def __init__( self, llm : HuggingFacePipeline ):
        """Initialize inference class.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            config: Inference configuration
        """
        self.llm = llm      
        
    def get_conversation_chain(self , user_id: str) -> ConversationChain:
        if user_id not in conversation_chains:
            memory = ConversationBufferMemory()
            conversation_chains[user_id] = ConversationChain(llm= self.llm, memory=memory, verbose=True)
            count_chat[user_id] = 2
            conversation_chains[user_id].memory.chat_memory.messages = [
                HumanMessage(content='Bạn là ai?') ,
                AIMessage(content='Mình là một trợ lý AI pháp luật được huấn luyện bởi HITman.AI, chuyên giải thích các thuật ngữ, khái niệm, quy định trong lĩnh vực pháp luật Việt Nam giống như một cuốn từ điển pháp luật.'),
                HumanMessage(content='Vui lòng chỉ trả lời đúng nội dung pháp luật, và tập trung vào câu hỏi ở dòng cuối cùng , Không được dùng từ human') ,
                AIMessage(content='Mình là một AI Pháp luật được huấn luyện bởi HITman.AI , rất mong được hhỗ trợ' )
            ]
        else : count_chat[user_id] += 1
        if count_chat[user_id] >= 7:
            count_chat[user_id] -= 1 
            data = conversation_chains[user_id].memory.chat_memory.messages
            del data[4:6]
    
        return conversation_chains[user_id]
    
    def generate_response( self, question: str, user_id: str) -> str:
        """Generate response using conversation memory"""
        try:
            conversation = self.get_conversation_chain(user_id)
            #clear_conversation_memory(user_id)
            data = conversation_chains[user_id].memory.chat_memory.messages 
        
            response = conversation.predict(input = question)
            response = processor_text(response) 
            data[count_chat[user_id]*2 + 1] = AIMessage(content= response )
                    
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Xin lỗi, hiện mình gặp sự cố. Vui lòng thử lại sau."
        

def generate_single_response(
    llm : HuggingFacePipeline,
    user_message : str ,
    user_id : str
    ) -> str:
    """Generate a single response from the model."""
    
    inference_engine = ChatbotInference(llm)
    respone = inference_engine.generate_response(user_message, user_id)
    
    return respone


#bug 
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
                response = inference_engine.generate_response(user_input, "user1")
                print(f"Bot: {response}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please enter a question.")
            