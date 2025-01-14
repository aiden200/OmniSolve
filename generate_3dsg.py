from depth_extraction.extract_depth import DepthCalculator


class SceneGraph:
    def __init__(self):
        self.d_calc = DepthCalculator()
    
    def generate_objects(self, video_input, temporary_path_output):

        output_path = f"{video_input[:-4]}_d.mp4"
        self.d_calc.extract_video_depth(video_input, output_path)



if __name__ == "__main__":
    sg = SceneGraph()
    sg.generate_objects("/home/aiden/Documents/cs/OmniSolve/depth_extraction/train_derailment_scene1/trimmed_output.mp4", "")

    