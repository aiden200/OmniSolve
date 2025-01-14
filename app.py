import streamlit as st
import cv2
import time
import threading
import queue
from datetime import datetime
from MMDuet.extract_timestamps import TimestampExtracter
from question_generation.prompts import *

# -----------------------------------------
# Placeholder for your real extractor class
# -----------------------------------------


# Example usage of your snippet:
# for timestamp, response, informative_score, relevance_score, frame, additional_info \
#        in timestampExtracter.start_chat(video_url):
#     if response:
#         # do something ...
#         pass

timestampExtracter = TimestampExtracter(DEFAULT_MM_PROMPT)

# -----------------------------------
# Thread function to read the yields
# -----------------------------------
def run_timestamp_extractor(video_url, output_queue):
    """
    Continuously read from timestampExtracter.start_chat(video_url)
    and store results in output_queue.
    """
    # Your “start_time”, “prev_context”, etc. might come from session state or real logic
    start_time = 0
    counter = 0
    prev_context = []
    qa = []  # Example placeholder

    for timestamp, response, informative_score, relevance_score, frame, additional_info in timestampExtracter.start_chat(video_url):
        # We'll just push everything to a queue so the main thread can handle it
        output_queue.put((timestamp, response, informative_score, relevance_score, frame, additional_info))

        # If your real logic wants to do something special when response is not empty:
        if response:
            end_time = timestamp - 1
            # example of calling your own function
            # ------------------------------------
            # YOUR CODE HERE
            # e.g.,
            # vid_output_file_path = f"{new_folder}/{counter}_video.mp4"
            # question_output_file_path = f"{new_folder}/{counter}_question.json"
            # text_output_file_path = f"{new_folder}/{counter}_description.text"
            #
            # curr_context = self.video_processor.qa_over_part_video(
            #     video_url,
            #     start_time,
            #     end_time,
            #     vid_output_file_path,
            #     question_output_file_path,
            #     text_output_file_path,
            #     qa=qa,
            #     prev_context=prev_context
            # )
            # prev_context.append(curr_context)
            # start_time = end_time + 1
            # counter += 1
            st.session_state["timestamps"].append(timestamp)
            # For the example, let's just print something
            # print(f"Detected response from {start_time} to {end_time}: {response}")

# -----------------------
# Streamlit UI
# -----------------------
def main():
    st.title("Async Timestamp Extractor App")

    # ------------------------
    # Session State variables
    # ------------------------
    if "objectives" not in st.session_state:
        st.session_state["objectives"] = "Enter your objectives here"
    if "warnings" not in st.session_state:
        st.session_state["warnings"] = "Enter your warnings here"
    if "qa_pairs" not in st.session_state:
        st.session_state["qa_pairs"] = []  # list of (question, answer)
    if "timestamps" not in st.session_state:
        st.session_state["timestamps"] = []
    if "current_summary" not in st.session_state:
        st.session_state["current_summary"] = "No summary yet"

    # -------------
    # Left Sidebar
    # -------------
    st.sidebar.header("Controls")
    st.sidebar.write("Enter the video URL and manage Objectives/Warnings/Q&A")

    # Input for video URL
    video_url = st.sidebar.text_input("Video URL", value="question_generation/trimmed_video.mp4")

    # Objectives & Warnings
    st.sidebar.subheader("Edit Objectives")
    new_objectives = st.sidebar.text_area("Objectives", value=st.session_state["objectives"])
    if st.sidebar.button("Update Objectives"):
        st.session_state["objectives"] = new_objectives

    st.sidebar.subheader("Edit Warnings")
    new_warnings = st.sidebar.text_area("Warnings", value=st.session_state["warnings"])
    if st.sidebar.button("Update Warnings"):
        st.session_state["warnings"] = new_warnings

    st.sidebar.write("---")

    # Q&A form
    st.sidebar.subheader("Add Q&A")
    q_input = st.sidebar.text_input("Question")
    a_input = st.sidebar.text_input("Answer")
    if st.sidebar.button("Add Q&A"):
        if q_input and a_input:
            st.session_state["qa_pairs"].append((q_input, a_input))
        else:
            st.warning("Please enter both question and answer")

    # -------------
    # Main Layout
    # -------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Incoming Video Frames")
        video_placeholder = st.empty()   # to display frames
        result_placeholder = st.empty()  # to display textual info (response, scores, etc.)

    with col2:
        st.subheader("Dynamic Content")
        st.markdown("**Timestamps**:")
        st.write(st.session_state["timestamps"])
        
        st.markdown("**Current Summary**:")
        st.write(st.session_state["current_summary"])

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Objectives")
        st.write(st.session_state["objectives"])

        st.subheader("Warnings")
        st.write(st.session_state["warnings"])

    with col4:
        st.subheader("Q&A Log")
        for idx, (q, a) in enumerate(st.session_state["qa_pairs"]):
            st.write(f"**Q{idx+1}:** {q}")
            st.write(f"**A{idx+1}:** {a}")
            st.write("---")

    # ------------------------------------
    # Start/Stop the timestamp extraction
    # ------------------------------------
    if "stop_threads" not in st.session_state:
        st.session_state["stop_threads"] = False
    if "video_started" not in st.session_state:
        st.session_state["video_started"] = False

    def start_process():
        st.session_state["stop_threads"] = False
        st.session_state["video_started"] = True

    def stop_process():
        st.session_state["stop_threads"] = True
        st.session_state["video_started"] = False

    start_button = st.button("Start Extraction")
    stop_button = st.button("Stop Extraction")

    if start_button:
        start_process()
    if stop_button:
        stop_process()

    # We'll store incoming data from timestampExtracter in a queue
    global_queue = queue.Queue()

    if st.session_state["video_started"]:
        # Launch the thread that runs start_chat and pushes data to global_queue
        t = threading.Thread(target=run_timestamp_extractor, args=(video_url, global_queue), daemon=True)
        t.start()

        # Keep updating the UI while the thread runs
        while True:
            if st.session_state["stop_threads"]:
                break

            # Try to get new data from the queue
            try:
                timestamp, response, informative_score, relevance_score, frame, additional_info = global_queue.get(timeout=0.2)
            except:
                timestamp = None

            if timestamp is not None:
                # Convert to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", caption=f"Timestamp: {timestamp}s")

                # Display text-based info
                display_text = (
                    f"**Response**: {response}\n"
                    f"**Informative Score**: {informative_score}\n"
                    f"**Relevance Score**: {relevance_score}\n"
                    f"**Additional Info**: {additional_info}"
                )
                result_placeholder.markdown(display_text)

                # Optionally update summary
                # e.g. st.session_state["current_summary"] = response or some ML logic
                st.session_state["current_summary"] = f"Most recent response: {response}"

            st.experimental_yield()

        st.write("Extraction stopped.")

    else:
        st.write("Click 'Start Extraction' to begin.")

if __name__ == "__main__":
    main()
