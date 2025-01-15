import streamlit as st
import requests
import os
from config import *
import time


##############################
# Utility to read text files
##############################
def read_text_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

##############################
# Continually read updates
##############################
def load_live_updates():
    summary = read_text_file(SUMMARY_FILE)
    status = read_text_file(STATUS_REPORT_FILE)
    objectives = read_text_file(STORE_OBJECTIVE_FILE)
    warnings = read_text_file(STORE_WARNINGS_FILE)
    long_term_memory = read_text_file(LONG_TERM_MEMORY_FILE)
    return summary, status, objectives, warnings, long_term_memory

##############################
# Initialize session state
##############################

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts: [{"role": "user"/"assistant", "content": "..."}]

##############################
# Layout
##############################
st.title("Video QA - Demo UI")

# Section to show the live text updates
with st.sidebar:
    st.header("Live Updates")


    interval = 1.0  # Interval in seconds
    last_refresh = time.time()
    summary, status, objectives, warns, long_term_memory = load_live_updates()

    while time.time() - last_refresh < interval:
        # Check if the refresh interval has passed
        if time.time() - last_refresh >= interval:
            summary, status, objectives, warns, long_term_memory = load_live_updates()
            last_refresh = time.time()

    summary, status, objectives, warns, long_term_memory = load_live_updates()
    st.subheader("Summary")
    st.write(summary or "No summary yet.")
    st.subheader("Status")
    st.write(status or "No status yet.")
    st.subheader("Objectives")
    st.write(objectives or "No objectives yet.")
    st.subheader("Warnings")
    st.write(warns or "No warnings yet.")
    st.subheader("Long Term Memory")
    st.write(long_term_memory or "No memory yet.")
    
    # A button to manually refresh the sidebar info
    if st.button("Refresh Info"):
        load_live_updates()

##############################
# Chat / Q&A
##############################
st.subheader("Q&A Chat")

# Input area for user question
question = st.text_input("Ask a question about the video:")
if st.button("Send Question"):
    if question.strip():
        # Send to your Flask backend
        flask_url = "http://127.0.0.1:5000/query"  # Adjust if running on different host/port
        payload = {"question": question}
        try:
            resp = requests.post(flask_url, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                video_path = data.get("video_path", "")
                image_paths = data.get("image_paths", [])
                
                # Store Q & A in session state
                st.session_state["chat_history"].append({"role": "user", "content": question})
                st.session_state["chat_history"].append({"role": "assistant", "content": answer,
                                                      "video_path": video_path,
                                                      "image_paths": image_paths})
            else:
                st.warning(f"Backend error: HTTP {resp.status_code}")
        except Exception as e:
            st.error(f"Request failed: {e}")
    else:
        st.warning("Please enter a question.")

# Display chat history
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"**User**: {msg['content']}")
    else:
        st.markdown(f"**Assistant**: {msg['content']}")
        
        # If there are images
        if "image_paths" in msg and msg["image_paths"]:
            st.write("Images returned:")
            for img_path in msg["image_paths"]:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                else:
                    st.write(f"Image not found: {img_path}")
        
        # If there's a video
        if "video_path" in msg and msg["video_path"]:
            if os.path.exists(msg["video_path"]):
                st.write("Video returned:")
                st.video(msg["video_path"])
            else:
                st.write(f"Video not found: {msg['video_path']}")
