import json
import time
import logging

logger = logging.getLogger(__name__)

def generate_streaming_response(full_response: str):
    """
    Sinh streaming chunks dựa trên full_response đã có.
    
    Yields Server-Sent Events (SSE) với định dạng:
      data: {"content": "...", "done": bool}\n\n
    """
    try:
        words = full_response.split()
        for i, word in enumerate(words):
            chunk = {
                "content": word + " ",
                "done": i == len(words) - 1
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.05)
        # Kết thúc stream
        yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        error_chunk = {"error": "Failed to generate response", "done": True}
        yield f"data: {json.dumps(error_chunk)}\n\n"
