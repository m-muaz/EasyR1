import os
import cv2
import random
import numpy as np
from typing import List, Dict, Any, Generator
from datasets import Dataset, DatasetDict, Sequence, Image
from PIL import Image as PILImage
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import ImageFilter
import io
import sys
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
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
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

def find_video_files(directories: List[str]) -> List[str]:
    video_files = []
    print(f"Searching for videos in {len(directories)} directories...")
    
    dir_stats = {}  # Track videos per directory
    for directory in tqdm(directories, desc="Processing directories"):
        start_len = len(video_files)
        for root, _, files in os.walk(directory):
            for file in tqdm(files, desc=f"Scanning {os.path.basename(root)}", leave=False):
                if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                    video_files.append(os.path.join(root, file))
        
        dir_count = len(video_files) - start_len
        dir_stats[directory] = dir_count
        # print(f"\nFound {dir_count} videos in {directory}")
    
    print("\nVideo distribution across directories:")
    for directory, count in dir_stats.items():
        print(f"{directory}: {count} videos ({count/len(video_files)*100:.1f}%)")
    print(f"\nTotal videos found: {len(video_files)}")
    
    return video_files

def generate_balanced_video_data(
    real_videos: List[str],
    fake_videos: List[str],
    preprocessor: VideoPreprocessor,
    target_ratio: float = 1.0,
    augmentation_probability: float = 0.8,
    advanced_augmentation: bool = True,  # Enable advanced augmentation
    shuffle: bool = True
) -> Generator[Dict[str, Any], None, None]:
    """Generate balanced dataset with augmentation for the minority class (fake videos)."""
    
    num_real = len(real_videos)
    num_fake = len(fake_videos)
    
    # Calculate how many augmented fake videos we need
    target_fake = int(num_real * target_ratio)
    num_augmented_needed = max(0, target_fake - num_fake)
    
    # Create augmentation candidates (videos to be augmented)
    if num_augmented_needed > 0:
        # Select videos to augment (with replacement if needed)
        if num_augmented_needed <= num_fake:
            augmentation_candidates = random.sample(fake_videos, num_augmented_needed)
        else:
            # We need more augmented videos than we have originals
            augmentation_candidates = []
            while len(augmentation_candidates) < num_augmented_needed:
                augmentation_candidates.extend(random.sample(fake_videos, 
                                                            min(num_fake, num_augmented_needed - len(augmentation_candidates))))
    else:
        augmentation_candidates = []
    
    # For advanced augmentation, we need to load some real videos for mixing
    real_videos_for_mixing = []
    if advanced_augmentation and num_augmented_needed > 0:
        # Sample a subset of real videos for mixing
        mixing_candidates = min(len(real_videos), num_augmented_needed)
        real_videos_for_mixing = random.sample(real_videos, mixing_candidates)
    
    # Combine original videos with augmentation candidates
    video_pairs = [(path, 1, False) for path in real_videos]  # (path, label, needs_augmentation)
    video_pairs.extend([(path, 0, False) for path in fake_videos])  # Original fake videos
    video_pairs.extend([(path, 0, True) for path in augmentation_candidates])  # Augmented fake videos
    
    if shuffle:
        random.shuffle(video_pairs)
    
    for video_path, label, needs_augmentation in video_pairs:
        try:
            # Extract frames
            frames = preprocessor.extract_frames(video_path)
            
            # Apply augmentation if needed
            if needs_augmentation:
                # Choose augmentation strategy
                if advanced_augmentation and real_videos_for_mixing and random.random() < 0.4:
                    # Advanced augmentation: mix with real video
                    real_video_path = random.choice(real_videos_for_mixing)
                    real_frames = preprocessor.extract_frames(real_video_path)
                    
                    # Mix videos
                    frames = mix_videos(real_frames, frames)
                    
                    # Add noise to mixed video
                    if random.random() < 0.7:
                        frames = add_noise_to_frames(frames, noise_type='random')
                else:
                    # Basic augmentation
                    frames = augment_frames(frames, augmentation_strength=0.5)
                    
                    # Apply temporal augmentations with some probability
                    if random.random() < augmentation_probability:
                        frames = temporal_augment(frames)
                    
                    # Add noise with some probability
                    if random.random() < 0.5:
                        frames = add_noise_to_frames(frames, noise_type='random')
            
            prompt = "Analyze this video sequence and determine if it contains AI-generated or manipulated content. Is this video real or fake?"
            
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

def create_video_dataset(
    real_video_paths: List[str],
    fake_video_paths: List[str],
    train_split: float = 0.8,
    val_split: float = 0.1,
    target_fps: int = 8,
    target_resolution: tuple = (224, 224),
    num_frames: int = 8,
    skip_frames: bool = True,
    balance_dataset: bool = True,
    target_ratio: float = 1.0,
    seed: int = 42
) -> DatasetDict:
    """
    Create a dataset for video real/fake discrimination task with balanced classes.
    
    Args:
        real_video_paths: List of directories containing real videos
        fake_video_paths: List of directories containing fake videos
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        target_fps: Target frames per second
        target_resolution: Target frame resolution (height, width)
        num_frames: Number of frames to extract per video
        skip_frames: If True, sample frames evenly across video, else take consecutive frames
        balance_dataset: Whether to balance the dataset through augmentation
        target_ratio: Target ratio of fake:real videos (1.0 = equal numbers)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Find all video files
    real_videos = find_video_files(real_video_paths)
    fake_videos = find_video_files(fake_video_paths)
    
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    print(f"Class imbalance ratio (fake:real): {len(fake_videos)/len(real_videos):.2f}")
    
    print("debug; stop here")
    sys.exit()
    
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        target_fps=target_fps,
        target_resolution=target_resolution,
        num_frames=num_frames,
        skip_frames=skip_frames
    )
    
    # Split videos into train/val/test
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    real_train_idx = int(len(real_videos) * train_split)
    real_val_idx = int(len(real_videos) * (train_split + val_split))
    fake_train_idx = int(len(fake_videos) * train_split)
    fake_val_idx = int(len(fake_videos) * (train_split + val_split))
    
    splits = {
        "train": (real_videos[:real_train_idx], fake_videos[:fake_train_idx]),
        "validation": (real_videos[real_train_idx:real_val_idx], fake_videos[fake_train_idx:fake_val_idx]),
        "test": (real_videos[real_val_idx:], fake_videos[fake_val_idx:])
    }
    
    datasets = {}
    for split_name, (split_real, split_fake) in splits.items():
        # Only balance the training set
        use_balancing = balance_dataset and split_name == "train"
        
        if use_balancing:
            dataset = Dataset.from_generator(
                generate_balanced_video_data,
                gen_kwargs={
                    "real_videos": split_real,
                    "fake_videos": split_fake,
                    "preprocessor": preprocessor,
                    "target_ratio": target_ratio,
                    "shuffle": True
                }
            )
        else:
            dataset = Dataset.from_generator(
                generate_video_data,
                gen_kwargs={
                    "real_videos": split_real,
                    "fake_videos": split_fake,
                    "preprocessor": preprocessor,
                    "shuffle": True if split_name == "train" else False
                }
            )
        datasets[split_name] = dataset
    
    return DatasetDict(datasets).cast_column("images", Sequence(Image()))

def augment_frames(frames: List[PILImage.Image], augmentation_strength: float = 0.5) -> List[PILImage.Image]:
    """Apply basic augmentations to video frames."""
    # Define augmentation transforms
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2 * augmentation_strength,
            contrast=0.2 * augmentation_strength,
            saturation=0.2 * augmentation_strength,
            hue=0.1 * augmentation_strength
        ),
        transforms.RandomAffine(
            degrees=5 * augmentation_strength,
            translate=(0.1 * augmentation_strength, 0.1 * augmentation_strength),
            scale=(1 - 0.1 * augmentation_strength, 1 + 0.1 * augmentation_strength)
        )
    ])
    
    # Apply the same transform to all frames to maintain temporal consistency
    seed = random.randint(0, 2**32 - 1)
    augmented_frames = []
    
    for frame in frames:
        random.seed(seed)  # Use the same seed for all frames
        torch.manual_seed(seed)
        augmented_frames.append(augmentation_transforms(frame))
    
    return augmented_frames

def temporal_augment(frames: List[PILImage.Image]) -> List[PILImage.Image]:
    """Apply temporal augmentations to video frames."""
    num_frames = len(frames)
    
    # Randomly choose an augmentation type
    aug_type = random.choice(['reverse', 'subsample', 'repeat_frames'])
    
    if aug_type == 'reverse':
        # Reverse the order of frames
        return frames[::-1]
    
    elif aug_type == 'subsample':
        # Subsample frames and duplicate to maintain length
        if num_frames >= 4:
            subsample_size = random.randint(max(2, num_frames//2), num_frames-1)
            indices = sorted(random.sample(range(num_frames), subsample_size))
            subsampled = [frames[i] for i in indices]
            
            # Duplicate frames to reach original length
            result = []
            for i in range(num_frames):
                idx = min(i, subsample_size-1)
                result.append(subsampled[idx])
            return result
    
    elif aug_type == 'repeat_frames':
        # Randomly repeat some frames
        result = frames.copy()
        num_repeats = random.randint(1, max(1, num_frames//4))
        
        for _ in range(num_repeats):
            idx = random.randint(0, num_frames-1)
            # Insert a copy of the frame at a random position
            insert_pos = random.randint(0, len(result))
            result.insert(insert_pos, frames[idx].copy())
        
        # Trim to original length
        if len(result) > num_frames:
            result = result[:num_frames]
            
        return result
    
    # Default: return original frames
    return frames

def add_noise_to_frames(frames: List[PILImage.Image], noise_type: str = 'random', intensity: float = 0.1) -> List[PILImage.Image]:
    """
    Add various types of noise to video frames.
    
    Args:
        frames: List of PIL Image frames
        noise_type: Type of noise to add ('gaussian', 'salt_pepper', 'compression', 'blur', 'random')
        intensity: Strength of the noise effect (0.0 to 1.0)
    
    Returns:
        List of noisy frames
    """
    noisy_frames = []
    
    # Apply consistent noise pattern across all frames
    noise_seed = random.randint(0, 2**32 - 1)
    
    # Choose random noise type if specified
    if noise_type == 'random':
        noise_type = random.choice(['gaussian', 'salt_pepper', 'compression', 'blur', 'mixed'])
    
    # For mixed noise, apply multiple noise types
    if noise_type == 'mixed':
        # Apply 2-3 different noise types in sequence
        noise_types = random.sample(['gaussian', 'salt_pepper', 'compression', 'blur'], 
                                   k=random.randint(2, 3))
        
        # Reduce intensity for each noise type when combining
        reduced_intensity = intensity * 0.7
        
        temp_frames = frames
        for noise in noise_types:
            temp_frames = add_noise_to_frames(temp_frames, noise_type=noise, 
                                             intensity=reduced_intensity)
        return temp_frames
    
    for frame in frames:
        # Convert PIL to numpy array
        np_frame = np.array(frame).astype(np.float32) / 255.0
        
        if noise_type == 'gaussian':
            # Gaussian noise (simulates sensor noise)
            random.seed(noise_seed)
            np.random.seed(noise_seed)
            
            # Scale intensity to reasonable standard deviation
            std = intensity * 0.2
            noise = np.random.normal(0, std, np_frame.shape)
            noisy_np = np.clip(np_frame + noise, 0, 1)
            
            # Convert back to uint8 and PIL
            noisy_frame = PILImage.fromarray((noisy_np * 255).astype(np.uint8))
            
        elif noise_type == 'salt_pepper':
            # Salt and pepper noise (simulates transmission errors)
            random.seed(noise_seed)
            np.random.seed(noise_seed)
            
            # Scale intensity to reasonable probability
            prob = intensity * 0.05
            
            # Create salt and pepper mask
            mask = np.random.random(np_frame.shape[:2])
            salt = (mask < prob/2)
            pepper = (mask > 1 - prob/2)
            
            # Apply salt (white) and pepper (black) noise
            noisy_np = np_frame.copy()
            noisy_np[salt] = 1.0
            noisy_np[pepper] = 0.0
            
            # Convert back to uint8 and PIL
            noisy_frame = PILImage.fromarray((noisy_np * 255).astype(np.uint8))
            
        elif noise_type == 'compression':
            # JPEG compression artifacts (common in shared videos)
            # Scale intensity to quality factor (100 = best, 1 = worst)
            quality = int(100 - (intensity * 70))  # Map 0.0-1.0 to 100-30 quality
            
            # Save with JPEG compression and reload
            buffer = io.BytesIO()
            frame.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            noisy_frame = PILImage.open(buffer)
            
        elif noise_type == 'blur':
            # Motion or gaussian blur (simulates camera shake or focus issues)
            # Scale intensity to blur radius
            blur_radius = intensity * 3.0
            
            if random.random() < 0.5:
                # Gaussian blur
                noisy_frame = frame.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            else:
                # Motion blur
                kernel_size = int(blur_radius * 10) | 1  # Ensure odd number
                kernel_size = max(3, kernel_size)
                
                # Random angle for motion blur
                angle = random.uniform(0, 360)
                
                # Create motion blur kernel
                kernel = np.zeros((kernel_size, kernel_size))
                center = kernel_size // 2
                
                # Calculate points for line
                x1 = center
                y1 = center
                x2 = int(center + (kernel_size/2) * np.cos(np.radians(angle)))
                y2 = int(center + (kernel_size/2) * np.sin(np.radians(angle)))
                
                # Draw line on kernel
                cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
                
                # Normalize kernel
                kernel = kernel / np.sum(kernel)
                
                # Apply motion blur
                np_blurred = cv2.filter2D(np.array(frame), -1, kernel)
                noisy_frame = PILImage.fromarray(np_blurred)
        
        else:
            # Default: no noise
            noisy_frame = frame
            
        noisy_frames.append(noisy_frame)
    
    return noisy_frames

def mix_videos(real_frames: List[PILImage.Image], fake_frames: List[PILImage.Image], 
               mix_method: str = 'random') -> List[PILImage.Image]:
    """
    Mix real and fake video frames to create new fake samples.
    
    Args:
        real_frames: List of frames from a real video
        fake_frames: List of frames from a fake video
        mix_method: Mixing method ('blend', 'temporal_splice', 'spatial_splice', 'random')
    
    Returns:
        List of mixed frames (considered fake)
    """
    if len(real_frames) != len(fake_frames):
        # Ensure same number of frames by duplicating or truncating
        min_frames = min(len(real_frames), len(fake_frames))
        real_frames = real_frames[:min_frames]
        fake_frames = fake_frames[:min_frames]
    
    # Choose random mix method if specified
    if mix_method == 'random':
        mix_method = random.choice(['blend', 'temporal_splice', 'spatial_splice'])
    
    mixed_frames = []
    
    if mix_method == 'blend':
        # Alpha blending between real and fake frames
        alpha = random.uniform(0.2, 0.8)
        
        for real_frame, fake_frame in zip(real_frames, fake_frames):
            # Convert to numpy arrays
            real_np = np.array(real_frame).astype(float)
            fake_np = np.array(fake_frame).astype(float)
            
            # Blend frames
            mixed_np = (real_np * (1-alpha) + fake_np * alpha).astype(np.uint8)
            mixed_frames.append(PILImage.fromarray(mixed_np))
    
    elif mix_method == 'temporal_splice':
        # Splice frames temporally (some frames from real, some from fake)
        num_frames = len(real_frames)
        
        # Create random splice points
        splice_points = sorted(random.sample(range(1, num_frames), k=random.randint(1, 3)))
        
        # Start with either real or fake
        use_real = random.choice([True, False])
        
        for i in range(num_frames):
            # Check if we've hit a splice point
            if i in splice_points:
                use_real = not use_real
            
            # Add appropriate frame
            mixed_frames.append(real_frames[i] if use_real else fake_frames[i])
    
    elif mix_method == 'spatial_splice':
        # Splice frames spatially (part of frame from real, part from fake)
        for real_frame, fake_frame in zip(real_frames, fake_frames):
            # Get dimensions
            width, height = real_frame.size
            
            # Create a new blank image
            mixed_frame = PILImage.new('RGB', (width, height))
            
            # Choose splice type
            splice_type = random.choice(['vertical', 'horizontal', 'quadrant'])
            
            if splice_type == 'vertical':
                # Vertical split
                split_point = random.randint(width//4, 3*width//4)
                mixed_frame.paste(real_frame.crop((0, 0, split_point, height)), (0, 0))
                mixed_frame.paste(fake_frame.crop((split_point, 0, width, height)), (split_point, 0))
            
            elif splice_type == 'horizontal':
                # Horizontal split
                split_point = random.randint(height//4, 3*height//4)
                mixed_frame.paste(real_frame.crop((0, 0, width, split_point)), (0, 0))
                mixed_frame.paste(fake_frame.crop((0, split_point, width, height)), (0, split_point))
            
            elif splice_type == 'quadrant':
                # Random quadrant is fake, rest is real
                quad_x = random.randint(0, 1)
                quad_y = random.randint(0, 1)
                
                # Copy real frame first
                mixed_frame.paste(real_frame, (0, 0))
                
                # Replace one quadrant with fake
                quad_width = width // 2
                quad_height = height // 2
                x_start = quad_x * quad_width
                y_start = quad_y * quad_height
                
                quad_fake = fake_frame.crop((x_start, y_start, 
                                            x_start + quad_width, 
                                            y_start + quad_height))
                mixed_frame.paste(quad_fake, (x_start, y_start))
            
            mixed_frames.append(mixed_frame)
    
    return mixed_frames

def main():
    # Example usage
    base_video_path = "/blob/kyoungjun/"
    real_video_paths = [
        "real_activitynet_5sec_reformat",
        # "real_epic_5sec",
        # "real_kinetics_5sec",
        # "real_physics_5sec",
        # "real_ucf_5sec",
        "internvid_flt_1_reformatted",
        "internvid_flt_2_reformatted",
        "internvid_flt_3_reformatted",
        "internvid_flt_4_reformatted",
        "internvid_flt_5_reformatted",
        "internvid_flt_6_reformatted",
        "internvid_flt_7_reformatted",
        "internvid_flt_8_reformatted",
        "internvid_flt_9_reformatted",
        "internvid_flt_10_reformatted",
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
    
    dataset = create_video_dataset(
        real_video_paths=real_video_paths,
        fake_video_paths=fake_video_paths,
        train_split=0.8,
        val_split=0.1,
        target_fps=8,
        target_resolution=(224, 224),
        num_frames=8,
        skip_frames=True,
        balance_dataset=True,
        target_ratio=1.0,
        seed=42
    )
    
    # Optionally push to hub
    # dataset.push_to_hub("your-username/video-discrimination-dataset")

if __name__ == "__main__":
    main() 