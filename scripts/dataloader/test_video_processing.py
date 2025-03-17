import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.utils import make_grid
from typing import List
from .loader_video import VideoPreprocessor, generate_video_data

def display_video_frames(frames: List[Image.Image], title: str = None):
    """Save a grid of frames from a video to a temporary directory."""
    import os
    import numpy as np
    from torchvision.utils import save_image
    
    # Create tmp directory if it doesn't exist
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Convert PIL images to tensors
    frame_tensors = [torch.from_numpy(np.array(frame)).float() / 255.0 for frame in frames]
    frame_tensors = [frame.permute(2, 0, 1) for frame in frame_tensors]
    
    # Create a grid
    grid = make_grid(frame_tensors, nrow=4, padding=2, normalize=False)
    
    # Save the grid
    output_path = os.path.join(tmp_dir, f"{title if title else 'frame_grid'}.png")
    save_image(grid, output_path)
    print(f"Saved frame grid to: {output_path}")

def test_video_preprocessor(
    video_path: str,
    target_fps: int = 8,
    target_resolution: tuple = (224, 224),
    num_frames: int = 20,
    skip_frames: bool = True
):
    """Test VideoPreprocessor on a single video."""
    print(f"\nTesting VideoPreprocessor on: {video_path}")
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        target_fps=target_fps,
        target_resolution=target_resolution,
        num_frames=num_frames,
        skip_frames=skip_frames
    )
    
    try:
        # Get video info
        video_info = preprocessor._get_video_info(video_path)
        print("\nVideo Information:")
        for key, value in video_info.items():
            print(f"{key}: {value}")
        
        # Extract frames
        frames = preprocessor.extract_frames(video_path)
        print(f"\nExtracted {len(frames)} frames")
        print(f"Frame size: {frames[0].size}")
        
        # Display frames
        display_video_frames(frames, f"Frames from {os.path.basename(video_path)}")
        
        return True
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return False

def test_data_generation(
    real_video_paths: List[str],
    fake_video_paths: List[str],
    num_samples: int = 2
):
    """Test data generation pipeline."""
    print("\nTesting data generation pipeline")
    
    # Create preprocessor
    preprocessor = VideoPreprocessor()
    
    # Generate some samples
    generator = generate_video_data(
        real_videos=real_video_paths[:num_samples],
        fake_videos=fake_video_paths[:num_samples],
        preprocessor=preprocessor,
        shuffle=True
    )
    
    # Process samples
    for i, sample in enumerate(generator, 1):
        print(f"\nSample {i}:")
        print(f"ID: {sample['id']}")
        print(f"Problem: {sample['problem']}")
        print(f"Answer: {sample['answer']}")
        print(f"Choices: {sample['choices']}")
        print(f"Ground Truth: {sample['ground_truth']}")
        print(f"Number of frames: {len(sample['images'])}")
        
        # Display frames
        display_video_frames(
            sample['images'], 
            f"Frames from {sample['id']} (Answer: {sample['answer']})"
        )
        
        if i >= num_samples:
            break

def main():
    # Example usage
    base_video_path = "/blob/kyoungjun/"
    
    # Test paths - use a small subset
    real_video_paths = [
        os.path.join(base_video_path, "real_activitynet_5sec_reformat"),
        os.path.join(base_video_path, "internvid_flt_1_reformatted"),
    ]
    
    fake_video_paths = [
        os.path.join(base_video_path, "HunyuanVideo/results"),
        os.path.join(base_video_path, "gen_activitynet_5sec_reformat"),
    ]
    
    # Find some video files for testing
    from scripts.dataloader.loader_video import find_video_files
    real_videos = find_video_files(real_video_paths)[:2]  # Get first 2 real videos
    fake_videos = find_video_files(fake_video_paths)[:2]  # Get first 2 fake videos
    
    # Test VideoPreprocessor
    print("\nTesting VideoPreprocessor...")
    for video_path in real_videos + fake_videos:
        test_video_preprocessor(video_path)
    
    # Test data generation
    print("\nTesting data generation...")
    test_data_generation(real_videos, fake_videos)

if __name__ == "__main__":
    main() 