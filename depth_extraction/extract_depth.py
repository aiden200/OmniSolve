import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DPTForDepthEstimation
from transformers import pipeline
from depth_extraction.DAV2.depth_anything_v2.dpt import DepthAnythingV2
import os
from PIL import Image
import matplotlib


class DepthCalculator:
    def __init__(self, model_size = "base", batch_size=8):
        self.model = None
        self.load_model(model_size)
        self.batch_size = batch_size


    def load_model(self, model_size):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if model_size == "large":
            encoder = "vitl"
        elif model_size == "small":
            encoder = "vits"
        elif model_size == "base":
            encoder = "vitb"

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'depth_extraction/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu', weights_only=True))
        self.model = self.model.to(DEVICE).eval()
    

    def extract_image_depth(self, frame):
        depth = self.model.infer_image(frame, 518)
        avg_depth = np.mean(depth)

        return avg_depth


    def extract_video_depth(self, video_path, output_path):
        """
        Process each frame of the video to extract and save depth maps.
        """

        raw_video = cv2.VideoCapture(video_path)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        i = 0
        while raw_video.isOpened():
            i += 1
            # print(i)
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = self.model.infer_image(raw_frame, 518)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            out.write(depth)
         
        
        raw_video.release()
        out.release()

