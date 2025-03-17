import random
import numpy as np
import torch
import cv2
import io
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image as PILImage, ImageFilter
from torchvision import transforms
from tqdm import tqdm
class BasicAugmenter:
    """Class for basic frame-level and temporal augmentations."""
    
    def __init__(self, augmentation_strength: float = 0.5):
        """
        Initialize basic augmenter.
        
        Args:
            augmentation_strength: Strength of augmentations (0.0 to 1.0)
        """
        self.augmentation_strength = augmentation_strength
        self.frame_transforms = transforms.Compose([
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
    
    def augment_frames(self, frames: List[PILImage.Image]) -> List[PILImage.Image]:
        """
        Apply basic augmentations to video frames.
        
        Args:
            frames: List of PIL Image frames
            
        Returns:
            List of augmented frames
        """
        # Apply the same transform to all frames to maintain temporal consistency
        seed = random.randint(0, 2**32 - 1)
        augmented_frames = []
        
        for frame in frames:
            random.seed(seed)  # Use the same seed for all frames
            torch.manual_seed(seed)
            augmented_frames.append(self.frame_transforms(frame))
        
        return augmented_frames
    
    def temporal_augment(self, frames: List[PILImage.Image]) -> List[PILImage.Image]:
        """
        Apply temporal augmentations to video frames.
        
        Args:
            frames: List of PIL Image frames
            
        Returns:
            List of temporally augmented frames
        """
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


class AdvancedAugmenter:
    """Class for advanced noise and video mixing augmentations."""
    
    def __init__(self, noise_intensity: float = 0.1):
        """
        Initialize advanced augmenter.
        
        Args:
            noise_intensity: Strength of noise effects (0.0 to 1.0)
        """
        self.noise_intensity = noise_intensity
    
    def add_noise(self, frames: List[PILImage.Image], noise_type: str = 'random') -> List[PILImage.Image]:
        """
        Add various types of noise to video frames.
        
        Args:
            frames: List of PIL Image frames
            noise_type: Type of noise to add ('gaussian', 'salt_pepper', 'compression', 'blur', 'mixed', 'random')
        
        Returns:
            List of noisy frames
        """
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
            reduced_intensity = self.noise_intensity * 0.7
            temp_frames = frames
            
            for noise in noise_types:
                temp_augmenter = AdvancedAugmenter(noise_intensity=reduced_intensity)
                temp_frames = temp_augmenter.add_noise(temp_frames, noise_type=noise)
            
            return temp_frames
        
        noisy_frames = []
        for frame in frames:
            # Convert PIL to numpy array
            np_frame = np.array(frame).astype(np.float32) / 255.0
            
            if noise_type == 'gaussian':
                # Gaussian noise (simulates sensor noise)
                random.seed(noise_seed)
                np.random.seed(noise_seed)
                
                # Scale intensity to reasonable standard deviation
                std = self.noise_intensity * 0.2
                noise = np.random.normal(0, std, np_frame.shape)
                noisy_np = np.clip(np_frame + noise, 0, 1)
                
                # Convert back to uint8 and PIL
                noisy_frame = PILImage.fromarray((noisy_np * 255).astype(np.uint8))
                
            elif noise_type == 'salt_pepper':
                # Salt and pepper noise (simulates transmission errors)
                random.seed(noise_seed)
                np.random.seed(noise_seed)
                
                # Scale intensity to reasonable probability
                prob = self.noise_intensity * 0.05
                
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
                quality = int(100 - (self.noise_intensity * 70))  # Map 0.0-1.0 to 100-30 quality
                
                # Save with JPEG compression and reload
                buffer = io.BytesIO()
                frame.save(buffer, format="JPEG", quality=quality)
                buffer.seek(0)
                noisy_frame = PILImage.open(buffer)
                
            elif noise_type == 'blur':
                # Motion or gaussian blur (simulates camera shake or focus issues)
                # Scale intensity to blur radius
                blur_radius = self.noise_intensity * 3.0
                
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
    
    def mix_videos(self, real_frames: List[PILImage.Image], fake_frames: List[PILImage.Image], 
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


class DatasetBalancer:
    """Class to handle dataset balancing strategies."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize dataset balancer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def stratified_subsample(
        self,
        real_videos_by_dir: Dict[str, List[str]],
        target_count: int
    ) -> List[str]:
        """
        Perform stratified subsampling from multiple directories.
        
        Args:
            real_videos_by_dir: Dictionary mapping directories to lists of video paths
            target_count: Target number of videos to sample
            
        Returns:
            List of sampled video paths
        """
        print(f"Starting stratified subsampling, target count: {target_count}")
        total_real = sum(len(videos) for videos in real_videos_by_dir.values())
        sampling_ratio = target_count / total_real
        print(f"Total real videos: {total_real}, sampling ratio: {sampling_ratio:.3f}")
        
        sampled_videos = []
        for directory, videos in tqdm(real_videos_by_dir.items(), desc="Processing directories"):
            # Stratified sampling - maintain original distribution
            dir_target_count = int(len(videos) * sampling_ratio)
            print(f"Sampling from directory {directory}: {len(videos)} videos -> {dir_target_count} videos")
            dir_sampled = random.sample(videos, min(dir_target_count, len(videos)))
            sampled_videos.extend(dir_sampled)
        
        # If we didn't get enough videos due to rounding, sample more
        if len(sampled_videos) < target_count:
            remaining = target_count - len(sampled_videos)
            print(f"Need {remaining} more videos to reach target count")
            # Get all videos not already sampled
            remaining_videos = []
            for videos in tqdm(real_videos_by_dir.values(), desc="Gathering remaining videos"):
                remaining_videos.extend([v for v in videos if v not in sampled_videos])
            
            if remaining_videos:
                additional = random.sample(remaining_videos, min(remaining, len(remaining_videos)))
                sampled_videos.extend(additional)
                print(f"Sampled {len(additional)} additional videos")
        
        print(f"Stratified subsampling complete. Total sampled: {len(sampled_videos)}")
        return sampled_videos
    
    def create_balanced_splits(
        self,
        real_videos_by_dir: Dict[str, List[str]],
        fake_videos: List[str],
        train_split: float = 0.8,
        val_split: float = 0.1,
        balance_train_only: bool = True
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
        """
        Create balanced train/val/test splits using subsampling.
        
        Args:
            real_videos_by_dir: Dictionary mapping directories to lists of real video paths
            fake_videos: List of fake video paths
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            balance_train_only: Whether to balance only the training set
            
        Returns:
            Tuple of (real_train, real_val, real_test, fake_train, fake_val, fake_test)
        """
        print("\n=== Creating Balanced Splits ===")
        print(f"Balance train only: {balance_train_only}")
        
        # Split fake videos into train/val/test
        print("\nSplitting fake videos...")
        random.shuffle(fake_videos)
        fake_train_idx = int(len(fake_videos) * train_split)
        fake_val_idx = int(len(fake_videos) * (train_split + val_split))
        
        fake_train = fake_videos[:fake_train_idx]
        fake_val = fake_videos[fake_train_idx:fake_val_idx]
        fake_test = fake_videos[fake_val_idx:]
        print(f"Fake videos split: Train={len(fake_train)}, Val={len(fake_val)}, Test={len(fake_test)}")
        
        # Calculate total real videos
        total_real = sum(len(videos) for videos in real_videos_by_dir.values())
        print(f"\nTotal real videos available: {total_real}")
        
        if balance_train_only:
            print("\nBalancing training set only...")
            # Only balance training set, keep all real videos for val/test
            real_train_target = len(fake_train)
            print(f"Target count for real training videos: {real_train_target}")
            
            # Sample for training
            real_train = self.stratified_subsample(real_videos_by_dir, real_train_target)
            
            # For val and test, use all remaining videos
            print("\nAllocating remaining videos to validation and test sets...")
            remaining_by_dir = {}
            for directory, videos in tqdm(real_videos_by_dir.items(), desc="Processing remaining videos"):
                remaining_by_dir[directory] = [v for v in videos if v not in real_train]
            
            # Split remaining videos
            remaining_videos = [v for videos in remaining_by_dir.values() for v in videos]
            random.shuffle(remaining_videos)
            
            val_size = int(len(remaining_videos) * (val_split / (1 - train_split)))
            real_val = remaining_videos[:val_size]
            real_test = remaining_videos[val_size:]
        else:
            print("\nBalancing all splits...")
            # Balance all splits
            real_train_target = len(fake_train)
            real_val_target = len(fake_val)
            real_test_target = len(fake_test)
            print(f"Target counts - Train: {real_train_target}, Val: {real_val_target}, Test: {real_test_target}")
            
            # Create copies of the videos for each split
            remaining_by_dir = {dir: videos.copy() for dir, videos in real_videos_by_dir.items()}
            
            print("\nSampling training set...")
            real_train = self.stratified_subsample(remaining_by_dir, real_train_target)
            
            # Remove training videos from the pool
            for directory in tqdm(remaining_by_dir.keys(), desc="Removing training videos"):
                remaining_by_dir[directory] = [v for v in remaining_by_dir[directory] if v not in real_train]
            
            print("\nSampling validation set...")
            real_val = self.stratified_subsample(remaining_by_dir, real_val_target)
            
            # Remove validation videos from the pool
            for directory in tqdm(remaining_by_dir.keys(), desc="Removing validation videos"):
                remaining_by_dir[directory] = [v for v in remaining_by_dir[directory] if v not in real_val]
            
            print("\nSampling test set...")
            real_test = self.stratified_subsample(remaining_by_dir, real_test_target)
        
        print("\n=== Split Creation Complete ===")
        print(f"Final split sizes:")
        print(f"Train - Real: {len(real_train)}, Fake: {len(fake_train)}")
        print(f"Val   - Real: {len(real_val)}, Fake: {len(fake_val)}")
        print(f"Test  - Real: {len(real_test)}, Fake: {len(fake_test)}\n")
        
        return real_train, real_val, real_test, fake_train, fake_val, fake_test
    
    def create_augmentation_plan(
        self,
        real_videos: List[str],
        fake_videos: List[str],
        target_ratio: float = 1.0
    ) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
        """
        Create a plan for augmenting fake videos to balance the dataset.
        
        Args:
            real_videos: List of real video paths
            fake_videos: List of fake video paths
            target_ratio: Target ratio of fake:real videos (1.0 = equal numbers)
            
        Returns:
            Tuple of (real_videos_for_mixing, fake_videos_to_augment, mixing_pairs)
        """
        # Calculate how many augmented fake videos we need
        target_fake_count = int(len(real_videos) * target_ratio)
        num_augmented_needed = max(0, target_fake_count - len(fake_videos))
        
        if num_augmented_needed <= 0:
            return [], [], []
        
        # Select fake videos to augment (with replacement if needed)
        fake_videos_to_augment = []
        while len(fake_videos_to_augment) < num_augmented_needed:
            fake_videos_to_augment.extend(
                random.sample(fake_videos, min(len(fake_videos), 
                                              num_augmented_needed - len(fake_videos_to_augment)))
            )
        
        # Select real videos for mixing
        real_videos_for_mixing = random.sample(
            real_videos, 
            min(len(real_videos), num_augmented_needed)
        )
        
        # Create mixing pairs
        mixing_pairs = []
        for i in range(min(len(real_videos_for_mixing), len(fake_videos_to_augment))):
            mixing_pairs.append((real_videos_for_mixing[i], fake_videos_to_augment[i]))
        
        return real_videos_for_mixing, fake_videos_to_augment, mixing_pairs 