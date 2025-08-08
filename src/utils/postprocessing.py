import random

def processor_text(text: str) -> str:
    """
    Xử lý chuỗi đầu ra thô của model.
    Args:    
    text:    Chuỗi đầu ra thô của model, có định dạng "… AI: <reply> Human: …"
    Returns: Chuỗi đã cắt và chèn gợi ý.
    """
    # Danh sách câu gợi ý
    hints = [
        "Bạn có muốn biết rằng…?",
        "Bạn có hứng thú tìm hiểu về…?",
        "Bạn có quan tâm đến… không?",
        "Bạn có muốn khám phá…?"
    ]

    idx = text.rfind("AI: ")
    if idx == -1:
        return text

    start = idx + len("AI: ")
    processed = text[start:].lstrip()

    return processed.replace("Human: ", random.choice(hints) + "\n   Khi ")
