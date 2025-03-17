import os
import cv2
import random
import numpy as np
from typing import List, Dict, Any, Generator, Optional, Tuple
from datasets import Dataset, DatasetDict, Sequence, Image
from torch.utils.data import DataLoader
from PIL import Image as PILImage
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import ImageFilter
import io
import sys

# # Add the project root directory to the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

# print(sys.path)

# Import augmentation classes
from .data_augmentation import BasicAugmenter, AdvancedAugmenter, DatasetBalancer

# Supported video extensions
VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']

class VideoPreprocessor:
    def __init__(
        self,
        target_fps: int = 8,
        target_resolution: tuple = (224, 224),
        num_frames: int = 8,
        skip_frames: bool = True
    ):
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.num_frames = num_frames
        self.skip_frames = skip_frames
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        }
    
    def extract_frames(self, video_path: str) -> List[PILImage.Image]:
        video_info = self._get_video_info(video_path)
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame sampling strategy
        total_frames = video_info["frame_count"]
        
        if self.num_frames > total_frames:
            # Sample 1/3 of the frame count
            self.num_frames = total_frames // 3
        
        
        if self.skip_frames:
            # Evenly sample frames across the video
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Take consecutive frames from a random starting point
            max_start = max(0, total_frames - self.num_frames)
            start_idx = random.randint(0, max_start)
            frame_indices = range(start_idx, start_idx + self.num_frames)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
            
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.target_resolution)
            frame_pil = PILImage.fromarray(frame_resized)
            frames.append(frame_pil)
        
        cap.release()
        return frames

def find_video_files(directories: List[str], return_as_dict: bool = False) -> List[str]:
    print(f"Searching for videos in {len(directories)} directories...")
    
    return_dict = {}
        
    for directory in tqdm(directories, desc="Processing directories"):
        dir_video_files = []
        # start_len = len(global_video_files)
        for root, _, files in os.walk(directory):
            for file in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
                if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                    dir_video_files.append(os.path.join(root, file))
                    # global_video_files.append(os.path.join(root, file))
                
        # dir_count = len(global_video_files) - start_len
        dir_count = len(dir_video_files)
        return_dict[directory] = dir_video_files
    
    if not return_as_dict:
        global_video_files = []
        for directory, video_files in return_dict.items():
            global_video_files.extend(video_files)
    
    ## ----------------------         Video Statistics         ----------------------
    # Total video count
    total_video_count = sum(len(video_files) for video_files in return_dict.values())        
    
    print("\nVideo distribution across directories:")
    for directory, video_paths in return_dict.items():
        print(f"{directory}: {len(video_paths)} videos ({len(video_paths)/total_video_count*100:.1f}%)")
    print(f"\nTotal videos found: {total_video_count}")
    
    if return_as_dict:
        return return_dict
    else:
        return global_video_files

def generate_video_data(
    real_videos: List[str],
    fake_videos: List[str],
    preprocessor: VideoPreprocessor,
    shuffle: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """Generate dataset from real and fake videos."""
    
    # Combine videos with labels
    video_pairs = [(path, 1) for path in real_videos] + [(path, 0) for path in fake_videos]
    
    if shuffle:
        random.shuffle(video_pairs)
    
    for video_path, label in video_pairs:
        try:
            # Extract frames
            frames = preprocessor.extract_frames(video_path)
            
            prompt = """
            Analyze this video sequence and determine if it contains AI-generated or manipulated content. Then, answer the question below.
            Question: Is this video real or fake?
            Answer: real or fake
            """
            
            yield {
                "images": frames,
                "problem": "<images>" + prompt,
                "answer": "real" if label == 1 else "fake",
                "id": os.path.basename(video_path),
                "choices": ["real", "fake"],
                "ground_truth": "A" if label == 1 else "B"
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            continue

def generate_augmented_video_data(
    real_videos: List[str],
    fake_videos: List[str],
    preprocessor: VideoPreprocessor,
    basic_augmenter: BasicAugmenter,
    advanced_augmenter: AdvancedAugmenter,
    augmentation_config: Dict[str, Any],
    shuffle: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """Generate dataset with augmentation for fake videos."""
    
    # Get augmentation parameters
    use_basic_aug = augmentation_config.get('use_basic_augmentation', True)
    use_advanced_aug = augmentation_config.get('use_advanced_augmentation', False)
    aug_probability = augmentation_config.get('augmentation_probability', 0.5)
    target_ratio = augmentation_config.get('target_ratio', 1.0)
    
    # Create dataset balancer
    balancer = DatasetBalancer(seed=augmentation_config.get('seed', 42))
    
    # Create augmentation plan
    real_for_mixing, fake_to_augment, mixing_pairs = balancer.create_augmentation_plan(
        real_videos, fake_videos, target_ratio
    )
    
    # Combine original videos
    video_pairs = [(path, 1, False) for path in real_videos]  # (path, label, needs_augmentation)
    video_pairs.extend([(path, 0, False) for path in fake_videos])  # Original fake videos
    
    # Add augmentation candidates
    for fake_path in fake_to_augment:
        video_pairs.append((fake_path, 0, True))  # Augmented fake videos
    
    if shuffle:
        random.shuffle(video_pairs)
    
    # Create a mapping from fake path to real path for mixing
    mixing_map = {fake: real for real, fake in mixing_pairs}
    
    for video_path, label, needs_augmentation in video_pairs:
        try:
            # Extract frames
            frames = preprocessor.extract_frames(video_path)
            
            # Apply augmentation if needed
            if needs_augmentation:
                # Choose augmentation strategy
                if use_advanced_aug and video_path in mixing_map and random.random() < aug_probability:
                    # Advanced augmentation: mix with real video
                    real_video_path = mixing_map[video_path]
                    real_frames = preprocessor.extract_frames(real_video_path)
                    
                    # Mix videos
                    frames = advanced_augmenter.mix_videos(real_frames, frames)
                    
                    # Add noise to mixed video
                    if random.random() < 0.7:
                        frames = advanced_augmenter.add_noise(frames, noise_type='random')
                else:
                    # Basic augmentation
                    if use_basic_aug:
                        frames = basic_augmenter.augment_frames(frames)
                        
                        # Apply temporal augmentations with some probability
                        if random.random() < aug_probability:
                            frames = basic_augmenter.temporal_augment(frames)
                    
                    # Add noise with some probability
                    if use_advanced_aug and random.random() < 0.5:
                        frames = advanced_augmenter.add_noise(frames, noise_type='random')
            
            prompt = """
            Analyze this video sequence and determine if it contains AI-generated or manipulated content.
            Then, considering temporal consistency, visual quality,
            obeying of physical laws, range and smoothness of motion, etc and tell me if this video is real or fake.
            """
            
            yield {
                "images": frames,
                "problem": "<image>" + prompt,
                "answer": "This video is real." if label == 1 else "This video is fake.",
                "id": os.path.basename(video_path) + ("_augmented" if needs_augmentation else ""),
                "choices": ["This video is real.", "This video is fake."],
                "ground_truth": "A" if label == 1 else "B"
            }
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            continue

def create_dataset(
    real_video_paths: List[str],
    fake_video_paths: List[str],
    preprocessor: VideoPreprocessor,
    train_split: float = 0.8,
    val_split: float = 0.1,
    balance_strategy: str = 'subsample',  # 'subsample', 'augment', 'none'
    balance_train_only: bool = True,
    augmentation_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> DatasetDict:
    """
    Create a dataset for video real/fake discrimination with flexible balancing strategies.
    
    Args:
        real_video_paths: List of directories containing real videos
        fake_video_paths: List of directories containing fake videos
        preprocessor: VideoPreprocessor instance
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        balance_strategy: How to balance the dataset ('subsample', 'augment', 'none')
        balance_train_only: Whether to balance only the training set
        augmentation_config: Configuration for augmentation
        seed: Random seed for reproducibility
    Returns:
        DatasetDict with train, validation, and test splits
    """
    random.seed(seed)
    
    # Find all video files
    real_videos_by_dir = find_video_files(real_video_paths, return_as_dict=True)
    
    fake_videos = find_video_files(fake_video_paths)
    
    # Calculate total counts
    total_real = sum(len(videos) for videos in real_videos_by_dir.values())
    total_fake = len(fake_videos)
    
    print(f"Found {total_real} real videos across {len(real_video_paths)} directories")
    print(f"Found {total_fake} fake videos")
    print(f"Original imbalance ratio (fake:real): {total_fake/total_real:.3f}")
    
    # Create dataset balancer
    balancer = DatasetBalancer(seed=seed)
    
    # Create balanced splits based on strategy
    if balance_strategy == 'subsample':
        # Use stratified subsampling
        real_train, real_val, real_test, fake_train, fake_val, fake_test = balancer.create_balanced_splits(
            real_videos_by_dir, fake_videos, train_split, val_split, balance_train_only
        )
    elif balance_strategy == 'none':
        # No balancing, just split the data
        all_real_videos = [video for videos in real_videos_by_dir.values() for video in videos]
        random.shuffle(all_real_videos)
        random.shuffle(fake_videos)
        
        real_train_idx = int(len(all_real_videos) * train_split)
        real_val_idx = int(len(all_real_videos) * (train_split + val_split))
        fake_train_idx = int(len(fake_videos) * train_split)
        fake_val_idx = int(len(fake_videos) * (train_split + val_split))
        
        real_train = all_real_videos[:real_train_idx]
        real_val = all_real_videos[real_train_idx:real_val_idx]
        real_test = all_real_videos[real_val_idx:]
        fake_train = fake_videos[:fake_train_idx]
        fake_val = fake_videos[fake_train_idx:fake_val_idx]
        fake_test = fake_videos[fake_val_idx:]
    else:  # 'augment'
        # Split data first
        all_real_videos = [video for videos in real_videos_by_dir.values() for video in videos]
        random.shuffle(all_real_videos)
        random.shuffle(fake_videos)
        
        real_train_idx = int(len(all_real_videos) * train_split)
        real_val_idx = int(len(all_real_videos) * (train_split + val_split))
        fake_train_idx = int(len(fake_videos) * train_split)
        fake_val_idx = int(len(fake_videos) * (train_split + val_split))
        
        real_train = all_real_videos[:real_train_idx]
        real_val = all_real_videos[real_train_idx:real_val_idx]
        real_test = all_real_videos[real_val_idx:]
        fake_train = fake_videos[:fake_train_idx]
        fake_val = fake_videos[fake_train_idx:fake_val_idx]
        fake_test = fake_videos[fake_val_idx:]
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Training: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Validation: {len(real_val)} real, {len(fake_val)} fake")
    print(f"Test: {len(real_test)} real, {len(fake_test)} fake")
    
    # Create datasets
    datasets = {}
    
    # Configure augmentation if needed
    if balance_strategy == 'augment' and augmentation_config is None:
        augmentation_config = {
            'use_basic_augmentation': True,
            'use_advanced_augmentation': False,
            'augmentation_probability': 0.5,
            'target_ratio': 1.0,
            'seed': seed
        }
    
    # Initialize augmenters if using augmentation
    if balance_strategy == 'augment':
        basic_augmenter = BasicAugmenter(
            augmentation_strength=augmentation_config.get('augmentation_strength', 0.5)
        )
        advanced_augmenter = AdvancedAugmenter(
            noise_intensity=augmentation_config.get('noise_intensity', 0.1)
        )
        
        # Training set with augmentation
        datasets["train"] = Dataset.from_generator(
            generate_augmented_video_data,
            gen_kwargs={
                "real_videos": real_train,
                "fake_videos": fake_train,
                "preprocessor": preprocessor,
                "basic_augmenter": basic_augmenter,
                "advanced_augmenter": advanced_augmenter,
                "augmentation_config": augmentation_config,
                "shuffle": True
            }
        )
    else:
        # Training set without augmentation
        datasets["train"] = Dataset.from_generator(
            generate_video_data,
            gen_kwargs={
                "real_videos": real_train,
                "fake_videos": fake_train,
                "preprocessor": preprocessor,
                "shuffle": True,
            }
        )
    
    # Validation set (no augmentation)
    datasets["validation"] = Dataset.from_generator(
        generate_video_data,
        gen_kwargs={
            "real_videos": real_val,
            "fake_videos": fake_val,
            "preprocessor": preprocessor,
            "shuffle": False,
        }
    )
    
    # Test set (no augmentation)
    datasets["test"] = Dataset.from_generator(
        generate_video_data,
        gen_kwargs={
            "real_videos": real_test,
            "fake_videos": fake_test,
            "preprocessor": preprocessor,
            "shuffle": False,
        }
    )
    
    return DatasetDict(datasets).cast_column("images", Sequence(Image()))

def main():
    # Example usage
    base_video_path = "/blob/kyoungjun/"
    real_video_paths = [
        "real_activitynet_5sec_reformat",
        "internvid_flt_1_reformatted",
        "internvid_flt_2_reformatted",
        "internvid_flt_3_reformatted",
        # "internvid_flt_4_reformatted",
        # "internvid_flt_5_reformatted",
        # "internvid_flt_6_reformatted",
        # "internvid_flt_7_reformatted",
        # "internvid_flt_8_reformatted",
        # "internvid_flt_9_reformatted",
        # "internvid_flt_10_reformatted",
    ]
    fake_video_paths = [
        "HunyuanVideo/results",
        "gen_activitynet_5sec_reformat",
        "gen_internvid_flt",
        "gen_internvid_flt_10steps",
        "gen_internvid_flt_15steps",
        "gen_internvid_flt_20steps",
        "gen_internvid_flt_25steps",
        "gen_internvid_flt_30steps",
        "gen_internvid_flt_35steps",
        "gen_internvid_flt_40steps",
        "gen_internvid_flt_45steps",
    ]
    
    # Merge the base_video_path with the video_paths
    real_video_paths = [os.path.join(base_video_path, path) for path in real_video_paths]
    fake_video_paths = [os.path.join(base_video_path, path) for path in fake_video_paths]
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        target_fps=8,
        target_resolution=(224, 224),
        num_frames=8,
        skip_frames=True
    )
    
    # Create dataset with subsampling strategy (default)
    dataset = create_dataset(
        real_video_paths=real_video_paths,
        fake_video_paths=fake_video_paths,
        preprocessor=preprocessor,
        train_split=0.8,
        val_split=0.1,
        balance_strategy='subsample',  # Options: 'subsample', 'augment', 'none'
        balance_train_only=False,
        seed=42,  
    )
    
    # Optionally push to hub
    # dataset.push_to_hub("your-username/video-discrimination-dataset")
    
    # Example of using augmentation strategy
    """
    augmentation_config = {
        'use_basic_augmentation': True,
        'use_advanced_augmentation': True,
        'augmentation_probability': 0.5,
        'augmentation_strength': 0.5,
        'noise_intensity': 0.1,
        'target_ratio': 1.0,
        'seed': 42
    }
    
    dataset_augmented = create_dataset(
        real_video_paths=real_video_paths,
        fake_video_paths=fake_video_paths,
        preprocessor=preprocessor,
        balance_strategy='augment',
        augmentation_config=augmentation_config
    )
    """

if __name__ == "__main__":
    main() 