import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



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


def find_subclip_and_local_frame(original_frame_idx, subclip_folder, fps_orig):
    """
    Given an original frame index (0-based) in the original video,
    returns (subclip_folder, local_frame_index) for the correct subclip,
    or (None, None) if not found.
    """
    time_in_seconds = original_frame_idx / fps_orig

    results = []
    i = 0

    start_time = 0

    while os.path.exists(os.path.join(subclip_folder, f"{i}.mp4")):
        subclip_path = os.path.join(subclip_folder, f"{i}.mp4")
        subclip_fps, clip_duration = get_video_info_opencv(subclip_path)
        
        results.append({
            "path": subclip_path,
            "start": start_time,
            "end": start_time + clip_duration,
            "fps": subclip_fps
            # "fps": fps_orig
        })

        start_time = start_time + clip_duration + 1
        i += 1
    
    # print(results)
    
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
    return frame, subclip_path



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
    subclip_frame, subclip_path = extract_frame_from_subclips(original_video_path, subclip_path, fps_orig, original_frame_idx)
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

    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    subclip_frame_rgb = cv2.cvtColor(subclip_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_frame_rgb)
    axes[0].set_title("Original Video Frame")
    axes[0].axis("off")

    axes[1].imshow(subclip_frame_rgb)
    axes[1].set_title("Extracted Frame from Subclip")
    axes[1].axis("off")

    print(f"Subclip in question: {subclip_path}")

    if num_diff == 0:
        print("✅ Frames match perfectly!")
    else:
        print(f"⚠️ Frames differ by pixel difference sum = {num_diff}")

        # Display heatmap of differences
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        axes[2].imshow(diff_gray, cmap="hot")
        axes[2].set_title("Difference Heatmap")
        axes[2].axis("off")
    if num_diff == 0:
        print("Frames match perfectly!")
    else:
        # Some differences might exist due to re-encoding, but hopefully minimal
        print(f"Frames differ by sum of pixel abs differences = {num_diff}")
        # You could implement a threshold check if needed.
    plt.show()

########################################################
# 6. EXAMPLE USAGE
########################################################

if __name__ == "__main__":
    # Example: we want to check frames #100, #223 from the original
    frames_to_test = [110, 122, 200, 300, 400, 490, 800, 1200]
    frames_to_test = [1000]

    vid_name = "Dk1y6G7hhUo"
    original_video_path = f"/data/multivent_yt_videos/{vid_name}.mp4"
    subclip_path = f"/data/multivent_processed/{vid_name}"
    for f_idx in frames_to_test:
        test_frame_extraction(original_video_path, subclip_path, f_idx)
