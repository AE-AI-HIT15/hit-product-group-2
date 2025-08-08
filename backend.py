from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4
import logging
import json
import time

from src.config.settings import get_config, MODEL_NAME , conversation_chains, count_chat
from src.model.model_loader import load_llm_pipeline 
from src.inference.inference import generate_single_response
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.utils.streaming import generate_streaming_response

app = FastAPI(title="Vietnamese Law Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
config = get_config()
# Load model on startup
llm = load_llm_pipeline(MODEL_NAME, config.model)
#model_inference = load_model(MODEL_NAME, config.model)


class ChatRequest(BaseModel):
    user_message: str
    user_id: str = None
    bot_id: str = "vietnamese_law_bot"
    stream: bool = False

class ChatResponse(BaseModel):
    content: str
    user_id: str
    bot_id: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Main chat endpoint with conversation memory"""
    try:
        logger.info(f"Received chat request from user: {request.user_id}")    
        if not request.user_message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
            

        user_id = request.user_id or str(uuid4())
        
        response_content = generate_single_response(
            llm ,
            request.user_message,
            user_id , 
        )
        
        return ChatResponse(
            content=response_content,
            user_id= user_id,
            bot_id=request.bot_id
        )
            
    except Exception as e:
        print(e)
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": True
    }
    
@app.post("/api/memory/reset", status_code=204)
def reset_memory( user_id : str ):
    if not req.user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")
    
    conversation_chains.pop(user_id, None)
    count_chat.pop(user_id, None)
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)