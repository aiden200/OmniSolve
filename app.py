import os, torchvision, transformers, time
import gradio as gr
from threading import Event

from MMDuet.models import parse_args
from MMDuet.demo.liveinfer import LiveInferForDemo, load_video
from question_generation.prompts import *
logger = transformers.logging.get_logger('liveinfer')

# Predefined / internal parameters (hidden from the UI).
args = parse_args('test')
args.stream_end_prob_threshold = 0.3
args.lora_pretrained = "MMDuet/outputs/mmduet"
liveinfer = LiveInferForDemo(args)

pause_event = Event()  # Event for pausing/resuming
pause_event.set()      # Initially, processing is allowed (not paused)

css = """
    #gr_title {text-align: center;}
    #gr_video {max-height: 480px;}
    #gr_chatbot {max-height: 480px;}
"""

import cv2

# Utility function for frame display, if desired
def display_frame(frame):
    if frame is not None:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)  # Display the frame for 1 millisecond

def update_messages(warnings, objectives_text, objectives_file):
    """
    Update warnings and objectives from both text input and optional file upload.
    """
    warnings_list = warnings.split("\n")
    warnings_markdown = "\n".join(f"- {item}" for item in warnings_list if item.strip())

    if objectives_file is not None:
        # Handle uploaded file (assuming it contains text lines for objectives)
        with open(objectives_file.name, "r") as f:
            file_content = f.read()
        file_obj_lines = [line.strip() for line in file_content.split("\n") if line.strip()]
        # Combine any text-based objectives with file-based objectives
        objectives_list = objectives_text.split("\n") + file_obj_lines
    else:
        objectives_list = objectives_text.split("\n")

    # Remove empty lines
    objectives_list = [obj.strip() for obj in objectives_list if obj.strip()]

    return warnings_markdown, objectives_list

class HistorySynchronizer:
    def __init__(self):
        self.history = []

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history

    def reset(self):
        self.history = []

history_synchronizer = HistorySynchronizer()

class ChatInterfaceWithUserMsgTime(gr.ChatInterface):
    def __init__(self, type: str = "tuples", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type  # Initialize the `type` attribute

    async def _display_input(self, message: str, history):
        message = f"[time={liveinfer.video_time:.1f}s] {message}"
        history = history_synchronizer.get_history()
        if isinstance(message, str) and self.type == "tuples":
            history.append([message, None])  # type: ignore
        elif isinstance(message, str) and self.type == "messages":
            history.append({"role": "user", "content": message})  # type: ignore
        history_synchronizer.set_history(history)
        return history  # type: ignore

with gr.Blocks(title="Demo", css=css) as demo:
    gr.Markdown("# Long Continuous Video Reasoning", elem_id='gr_title')
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ## IMPORTANT
                1. **Objectives**: Brief bullet points that the model uses to focus its summaries.
                2. **Warnings**: Items or objects of interest that the model should alert on if detected.
                3. For demo clarity, there is a short artificial delay after each frame, so playback may seem slow.

                If the app becomes unresponsive, you may need to restart.
                """
            )

    # Hidden chat interface row (so we can still use a ChatInterface)
    with gr.Row(visible=False):
        def handle_user_input(message, history):
            liveinfer.encode_given_query(message)
        gr_chat_interface = ChatInterfaceWithUserMsgTime(
            fn=handle_user_input,
            chatbot=gr.Chatbot(
                elem_id="gr_chatbot",
                label="chatbot",
                avatar_images=("demo/assets/user_avatar.png", "demo/assets/assistant_avatar.png"),
                render=False
            ),
        )

    with gr.Row(), gr.Blocks() as hyperparam_block:
        # Video input
        gr_video = gr.Video(
            label="Input Video", 
            visible=True, 
            autoplay=False
        )

        with gr.Column():
            # These parameters are internally set (below), so no UI is shown
            hidden_thres_mode = gr.Textbox(value="sum score", visible=False)
            hidden_threshold = gr.Slider(value=1.5, minimum=0, maximum=3, visible=False)
            hidden_frame_interval = gr.Slider(value=2.0, minimum=0.1, maximum=10, visible=False)

            # Default query (optional usage if needed)
            gr_query = DEFAULT_MM_PROMPT

            gr.Markdown("## Warnings and Objectives")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Warnings")
                    warnings_list_display = gr.Markdown(label="Warnings List")
                with gr.Column():
                    gr.Markdown("### Objectives")
                    objective_chatbox = gr.Chatbot(label="Objectives", height=200)

            with gr.Row():
                warnings_input = gr.Textbox(
                    label="Add to Warnings", 
                    placeholder="Type warnings here (one per line)..."
                )
                # Let user either type objectives or upload a file
                objectives_input = gr.Textbox(
                    label="Add to Objectives", 
                    placeholder="Type objectives here (one per line)..."
                )
                objectives_file = gr.File(label="Upload Objectives file (optional)")

            update_button = gr.Button("Update Lists")
            update_button.click(
                fn=update_messages, 
                inputs=[warnings_input, objectives_input, objectives_file], 
                outputs=[warnings_list_display, objective_chatbox],
            )

            # Button to start the analysis/chat
            gr_start_button = gr.Button("Start Chat", variant="primary")

        # Simplified examples: Just showing video and a possible prompt
        gr_examples = gr.Examples(
            examples=[
                ["demo/assets/office.mp4", "Please summarize what is happening in the office?"],
                ["demo/assets/drive.mp4", "Who is driving the car?"],
                ["/home/aiden/Documents/cs/OmniSolve/question_generation/trimmed_video.mp4", "Please describe what you see."]
            ],
            inputs=[gr_video, gr_chat_interface.textbox],
            label="Examples"
        )

    with gr.Row() as chat:
        with gr.Column():
            gr_frame_display = gr.Image(label="Current Model Input Frame", interactive=False)
            with gr.Row():
                gr_time_display = gr.Number(label="Current Video Time", value=0)
            with gr.Row():
                gr_inf_score_display = gr.Number(label="Informative Score", value=0)
                gr_rel_score_display = gr.Number(label="Relevance Score", value=0)

            with gr.Row():
                gr_pause_button = gr.Button("Pause Video")
                gr_stop_button = gr.Button("Stop Video", variant='stop')

        with gr.Column():
            gr_chat_interface.render()

    def start_chat(src_video_path, thres_mode, threshold, frame_interval, history):
        """
        Runs the main loop, pulling frames, generating responses, and yielding to UI.
        """
        # Clear the chatbot and frame display
        yield 0, 0, 0, None, []

        # Set the hyperparameters (internally)
        liveinfer.reset()
        history_synchronizer.reset()
        # We define which score heads to use (internally).
        liveinfer.score_heads = ["informative_score", "relevance_score"]

        # Apply thresholds from hidden inputs
        if thres_mode == "single-frame score":
            liveinfer.stream_end_prob_threshold = threshold
            liveinfer.stream_end_score_sum_threshold = None
        elif thres_mode == "sum score":
            liveinfer.stream_end_prob_threshold = None
            liveinfer.stream_end_score_sum_threshold = threshold

        # Setup frames-per-second
        frame_fps = 1 / frame_interval
        liveinfer.set_fps(frame_fps)

        # Load the video
        video_input, original_frame_list = load_video(src_video_path, frame_fps)
        liveinfer.input_video_stream(video_input)

        # Disable parameter editing while video is running
        for component in hyperparam_block.children:
            component.interactive = False

        # Main video loop
        while liveinfer.frame_embeds_queue:
            start_time = time.time()
            # Wait if pause was pressed
            pause_event.wait()

            ret = liveinfer.input_one_frame()
            history = history_synchronizer.get_history()

            if ret['response'] is not None:
                frame_idx = ret['frame_idx']
                if 0 <= frame_idx < len(original_frame_list):
                    frame = original_frame_list[frame_idx]

                history.append((None, f"[time={ret['time']}s] {ret['response']}"))
                history_synchronizer.set_history(history)

            elapsed_time = time.time() - start_time
            target_delay_time = min(frame_interval, 0.2)
            if elapsed_time < target_delay_time:
                time.sleep(frame_interval - elapsed_time)

            yield (
                ret['time'],
                ret['informative_score'],
                ret['relevance_score'],
                original_frame_list[ret['frame_idx'] - 1],
                history
            )

    gr_start_button.click(
        fn=start_chat,
        inputs=[
            gr_video, 
            hidden_thres_mode, 
            hidden_threshold, 
            hidden_frame_interval, 
            gr_chat_interface.chatbot
        ],
        outputs=[
            gr_time_display,
            gr_inf_score_display,
            gr_rel_score_display,
            gr_frame_display,
            gr_chat_interface.chatbot
        ]
    )

    def toggle_pause():
        if pause_event.is_set():
            pause_event.clear()  # Pause processing
            return "Resume Video"
        else:
            pause_event.set()    # Resume processing
            return "Pause Video"

    gr_pause_button.click(
        toggle_pause,
        inputs=[],
        outputs=gr_pause_button
    )

    def stop_chat():
        liveinfer.reset()
        history_synchronizer.reset()

        # Re-enable the hidden param block
        for component in hyperparam_block.children:
            component.interactive = True

        return 0, 0, 0, None, []

    gr_stop_button.click(
        stop_chat,
        inputs=[],
        outputs=[
            gr_time_display,
            gr_inf_score_display,
            gr_rel_score_display,
            gr_frame_display,
            gr_chat_interface.chatbot
        ]
    )

    demo.queue()
    demo.launch(share=False)
