from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uuid import uuid4
import logging
import json
import time

from src.config.settings import get_config, MODEL_NAME
from src.model.model_loader import load_model, prepare_model_for_inference
from src.inference.inference import generate_single_response
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
model, tokenizer = load_model(MODEL_NAME, config.model)
model_inference = prepare_model_for_inference(model)


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
    try:
        logger.info(f"Received chat request from user: {request.user_id}")
        if not request.user_message.strip():
            raise HTTPException(status_code=400, detail="Empty message")
        # Tạo user_id nếu chưa có
        user_id = request.user_id or str(uuid4())
        message = request.user_message
        if request.stream:
            return StreamingResponse(
                generate_single_response(model_inference, tokenizer, message, config.inference),
                media_type="text/plain"
            )
        else:
            response_content = generate_single_response(model_inference, tokenizer, message, config.inference)
            return ChatResponse(
                content=response_content,
                user_id=user_id,
                bot_id=request.bot_id
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)