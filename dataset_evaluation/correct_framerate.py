import cv2
import numpy as np
import os

########################################################
# 1. DEFINE YOUR SUBCLIPS INFO AND FPS
########################################################

# Example subclip metadata:
# Each dict has:
#   "path": path to the subclip file (already generated)
#   "start": start time in seconds (relative to original)
#   "end": end time in seconds (relative to original)
subclips = [
    {"path": "0_aaa.mp4", "start": 0,   "end": 10},   # covers [0..10s)
    {"path": "1_aaa.mp4", "start": 10, "end": 20},   # covers [10..20s)
    {"path": "2_aaa.mp4", "start": 20, "end": 30},   # covers [20..30s), etc.
]


def get_video_info_opencv(video_path):
    """
    Returns (fps, duration_in_seconds) for the given video_path using OpenCV.
    WARNING: If the video is variable frame rate (VFR) or metadata is inaccurate,
    this may be off. Use ffprobe for more reliable data if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0
    
    return fps, duration

def get_video_fps(video_path):
    video_capture = cv2.VideoCapture(video_path)
    # duration = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    return fps

def get_video_duration(video_path):
    video_capture = cv2.VideoCapture(video_path)
    duration = video_capture.get(cv2.CAP_PROP_POS_MSEC)
    # fps = video_capture.get(cv2.CAP_PROP_FPS)

    return duration

# Original video path
original_video_path = "aaa.mp4"

# Assume original video is 30 fps
fps_original = 30

# Typically subclips will keep the same fps, but if they differ:
# you could store that in each subclip dict as "fps_subclip" 
# or measure it with cv2/ffprobe. Here we assume it's the same:
fps_subclip = fps_original

########################################################
# 2. HELPER FUNCTION: EXTRACT FRAME FROM A VIDEO
########################################################

def extract_frame_from_video(video_path, frame_index):
    """
    Opens a video file at `video_path` with OpenCV,
    seeks to `frame_index`, grabs that frame, and returns as a numpy array (BGR).
    Returns None if reading fails.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video path does not exist: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return None
    
    # Ask OpenCV to seek directly to frame_index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_index} from {video_path}")
        return None
    
    return frame, fps

########################################################
# 3. FIND SUBCLIP AND FRAME FOR A GIVEN ORIGINAL FRAME
########################################################



def get_video_info_opencv(video_path):
    """
    Returns (fps, duration_in_seconds) for the given video_path using OpenCV.
    WARNING: If the video is variable frame rate (VFR) or metadata is inaccurate,
    this may be off. Use ffprobe for more reliable data if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0.0
    
    return fps, duration


def find_subclip_and_local_frame(original_frame_idx, subclip_path, fps_orig):
    """
    Given an original frame index (0-based) in the original video,
    returns (subclip_path, local_frame_index) for the correct subclip,
    or (None, None) if not found.
    """
    time_in_seconds = original_frame_idx / fps_orig

    results = []
    i = 0

    start_time = 0

    while os.path.exists(os.path.join(subclip_path, f"{i}.mp4")):
        subclip_path = os.path.join(subclip_path, f"{i}.mp4")
        subclip_fps, clip_duration = get_video_info_opencv(subclip_path)
        
        results.append({
            "path": subclip_path,
            "start_time": start_time,
            "end_time": start_time + clip_duration,
            "fps": subclip_fps
        })

        start_time = start_time + clip_duration
    

    
    # Find which subclip interval covers this time
    for clip_info in results:
        start_sec = clip_info["start"]
        end_sec = clip_info["end"]
        if start_sec <= time_in_seconds < end_sec:
            # Found subclip that contains this frame
            subclip_path = clip_info["path"]
            fps_subclip = clip_info["fps"]
            # local subclip frame = offset in seconds * subclip fps
            local_frame_idx = int((time_in_seconds - start_sec) * fps_subclip)
            return subclip_path, local_frame_idx
    
    # If no subclip found, we return None
    print(f"Warning: No subclip covers time={time_in_seconds:.2f}s.")
    return None, None

########################################################
# 4. MAIN FUNCTION: EXTRACT CORRECT FRAME FROM SUBCLIPS
########################################################

def extract_frame_from_subclips(original_video_path, subclip_path, fps_orig, original_frame_idx):
    """
    Finds which subclip covers the frame, then extracts the correct
    frame from that subclip.
    """
    subclip_path, local_frame_idx = find_subclip_and_local_frame(original_frame_idx, subclip_path, fps_orig)
    if subclip_path is None:
        return None
    
    # Now actually extract from subclip
    frame, fps = extract_frame_from_video(subclip_path, local_frame_idx)
    return frame



########################################################
# 5. TEST / SANITY CHECK
########################################################

def test_frame_extraction(original_video_path, subclip_path, original_frame_idx):
    """
    1. Extract the frame (original_frame_idx) directly from the original video.
    2. Extract the same frame via the subclip approach.
    3. Compare if they match.
    """
    print(f"\n=== Testing frame extraction for original frame index {original_frame_idx} ===")

    
    
    # (A) Extract directly from original
    original_frame, fps_orig = extract_frame_from_video(original_video_path, original_frame_idx)
    if original_frame is None:
        print("Failed to extract from original video.")
        return


    # (B) Extract via subclip approach
    subclip_frame = extract_frame_from_subclips(original_video_path, subclip_path, fps_orig, original_frame_idx)
    if subclip_frame is None:
        print("Failed to extract via subclip. Possibly out of range.")
        return
    
    # Compare shape
    if original_frame.shape != subclip_frame.shape:
        print(f"Frame shape mismatch: original={original_frame.shape}, subclip={subclip_frame.shape}")
        return
    
    # For a simple pixel-wise check (BGR difference):
    diff = cv2.absdiff(original_frame, subclip_frame)
    num_diff = np.sum(diff)
    if num_diff == 0:
        print("Frames match perfectly!")
    else:
        # Some differences might exist due to re-encoding, but hopefully minimal
        print(f"Frames differ by sum of pixel abs differences = {num_diff}")
        # You could implement a threshold check if needed.

########################################################
# 6. EXAMPLE USAGE
########################################################

if __name__ == "__main__":
    # Example: we want to check frames #100, #223 from the original
    frames_to_test = [100, 223]
    original_video_path = "/data/multivent_yt_videos/0vKs4-EZ_D0.mp4"
    subclip_path = "/data/multivent_processed/0vKs4-EZ_D0"
    for f_idx in frames_to_test:
        test_frame_extraction(original_video_path, subclip_path, f_idx)
