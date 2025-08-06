import streamlit as st
import requests
import logging
import time
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8002"

def send_chat_request(user_message, user_id, bot_id, stream=False):
    url = f"{API_BASE_URL}/api/chat"
    payload = {
        "user_message": user_message,
        "user_id": user_id,
        "bot_id": bot_id,
        "stream": stream
    }
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API error: {e}")
        return {"content": f"❌ Lỗi: {str(e)}"}

def check_health():
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "unhealthy"}

def response_generator(user_message, user_id, bot_id):
    resp = send_chat_request(user_message, user_id, bot_id, stream=False)
    content = resp.get("content", "")
    for word in content.split():
        yield word + " "
        time.sleep(0.03)

# --- UI ---
st.set_page_config(page_title="Vietnamese Law Chatbot", page_icon="⚖️", layout="wide")
st.title("⚖️ Vietnamese Law Chatbot")
st.markdown("*Trợ lý AI cho các câu hỏi pháp luật Việt Nam*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    health = check_health()
    if health.get("status") == "healthy":
        st.success("✅ Server đang hoạt động")
        st.info(f"Model: {health.get('model', '')}")
    else:
        st.error("❌ Server không hoạt động")
    enable_streaming = st.checkbox("Bật hiệu ứng streaming", value=False)
    if st.button("🗑️ Xóa lịch sử trò chuyện"):
        st.session_state.messages = []
        st.rerun()

# Khởi tạo user_id và bot_id mặc định (ẩn khỏi giao diện)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid4())
bot_id = "vietnamese_law_bot"

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi pháp luật của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if enable_streaming:
            response = st.write_stream(response_generator(prompt, st.session_state.user_id, bot_id))
        else:
            with st.spinner("Đang xử lý..."):
                resp = send_chat_request(prompt, st.session_state.user_id, bot_id)
                response = resp.get("content", "Không có phản hồi từ bot.")
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Vietnamese Law Chatbot - Powered by AI</small>
    </div>
    """, unsafe_allow_html=True
)