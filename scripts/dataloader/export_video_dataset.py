import os
import argparse
from datasets import Dataset, DatasetDict
from scripts.dataloader.loader_video import create_dataset, VideoPreprocessor, find_video_files
import pandas as pd

def export_dataset_for_rlhf(
    real_video_paths,
    fake_video_paths,
    output_dir,
    balance_strategy='subsample',
    balance_train_only=True,
    train_split=0.8,
    val_split=0.1,
    target_fps=8,
    target_resolution=(224, 224),
    num_frames=8,
    skip_frames=True,
    augmentation_config=None,
    seed=42
):
    """
    Create and export a video discrimination dataset in a format compatible with RLHFDataset.
    
    Args:
        real_video_paths: List of directories containing real videos
        fake_video_paths: List of directories containing fake videos
        output_dir: Directory to save the exported dataset
        balance_strategy: Strategy for balancing the dataset ('subsample', 'augment', 'none')
        balance_train_only: Whether to balance only the training set
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        target_fps: Target frames per second
        target_resolution: Target frame resolution (height, width)
        num_frames: Number of frames to extract per video
        skip_frames: If True, sample frames evenly across video, else take consecutive frames
        augmentation_config: Configuration for augmentation
        seed: Random seed for reproducibility
    """
    # Create preprocessor
    preprocessor = VideoPreprocessor(
        target_fps=target_fps,
        target_resolution=target_resolution,
        num_frames=num_frames,
        skip_frames=skip_frames
    )
    
    # Create dataset
    dataset = create_dataset(
        real_video_paths=real_video_paths,
        fake_video_paths=fake_video_paths,
        preprocessor=preprocessor,
        train_split=train_split,
        val_split=val_split,
        balance_strategy=balance_strategy,
        balance_train_only=balance_train_only,
        augmentation_config=augmentation_config,
        seed=seed
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export each split
    for split_name, split_dataset in dataset.items():
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame({
            "prompt": split_dataset["problem"],
            "answer": split_dataset["answer"],
            "images": split_dataset["images"],
            "id": split_dataset["id"],
            "choices": split_dataset["choices"],
            "ground_truth": split_dataset["ground_truth"]
        })
        
        # Save to parquet
        split_path = os.path.join(output_dir, f"{split_name}.parquet")
        df.to_parquet(split_path, index=False)
        print(f"Exported {split_name} dataset to {split_path} ({len(df)} examples)")

def main():
    parser = argparse.ArgumentParser(description="Export video dataset for RLHF training")
    parser.add_argument("--real-video-dirs", nargs="+", required=True, help="Directories containing real videos")
    parser.add_argument("--fake-video-dirs", nargs="+", required=True, help="Directories containing fake videos")
    parser.add_argument("--output-dir", required=True, help="Directory to save the exported dataset")
    parser.add_argument("--balance-strategy", choices=["subsample", "augment", "none"], default="subsample", 
                        help="Strategy for balancing the dataset")
    parser.add_argument("--balance-train-only", action="store_true", help="Balance only the training set")
    parser.add_argument("--train-split", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--val-split", type=float, default=0.1, help="Proportion of data for validation")
    parser.add_argument("--target-fps", type=int, default=8, help="Target frames per second")
    parser.add_argument("--target-height", type=int, default=224, help="Target frame height")
    parser.add_argument("--target-width", type=int, default=224, help="Target frame width")
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames to extract per video")
    parser.add_argument("--consecutive-frames", action="store_true", help="Use consecutive frames instead of evenly spaced")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Augmentation arguments
    parser.add_argument("--use-basic-augmentation", action="store_true", help="Use basic augmentation")
    parser.add_argument("--use-advanced-augmentation", action="store_true", help="Use advanced augmentation")
    parser.add_argument("--augmentation-probability", type=float, default=0.5, help="Probability of applying augmentation")
    parser.add_argument("--augmentation-strength", type=float, default=0.5, help="Strength of augmentation")
    parser.add_argument("--noise-intensity", type=float, default=0.1, help="Intensity of noise for advanced augmentation")
    parser.add_argument("--target-ratio", type=float, default=1.0, help="Target ratio of fake:real videos")
    
    args = parser.parse_args()
    
    # Create augmentation config if needed
    augmentation_config = None
    if args.balance_strategy == "augment":
        augmentation_config = {
            'use_basic_augmentation': args.use_basic_augmentation,
            'use_advanced_augmentation': args.use_advanced_augmentation,
            'augmentation_probability': args.augmentation_probability,
            'augmentation_strength': args.augmentation_strength,
            'noise_intensity': args.noise_intensity,
            'target_ratio': args.target_ratio,
            'seed': args.seed
        }
    
    export_dataset_for_rlhf(
        real_video_paths=args.real_video_dirs,
        fake_video_paths=args.fake_video_dirs,
        output_dir=args.output_dir,
        balance_strategy=args.balance_strategy,
        balance_train_only=args.balance_train_only,
        train_split=args.train_split,
        val_split=args.val_split,
        target_fps=args.target_fps,
        target_resolution=(args.target_height, args.target_width),
        num_frames=args.num_frames,
        skip_frames=not args.consecutive_frames,
        augmentation_config=augmentation_config,
        seed=args.seed
    )

def get_default_config():
    base_video_path = "/blob/kyoungjun/"
    real_video_paths = [
        "real_activitynet_5sec_reformat",
        "internvid_flt_1_reformatted",
        "internvid_flt_2_reformatted", 
        "internvid_flt_3_reformatted",
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

    # Merge paths
    real_video_paths = [os.path.join(base_video_path, path) for path in real_video_paths]
    fake_video_paths = [os.path.join(base_video_path, path) for path in fake_video_paths]

    # Default config
    config = {
        "real_video_dirs": real_video_paths,
        "fake_video_dirs": fake_video_paths,
        "output_dir": "/blob/zyhe/muaz/Real_fake_vid_data",
        "balance_strategy": "subsample",  # Options: 'subsample', 'augment', 'none'
        "balance_train_only": False,
        "train_split": 0.8,
        "val_split": 0.1,
        "target_fps": 8,
        "target_height": 224,
        "target_width": 224,
        "num_frames": 15,
        "consecutive_frames": False,
        "seed": 42,
        "use_basic_augmentation": False,
        "use_advanced_augmentation": False,
        "augmentation_probability": 0.5,
        "augmentation_strength": 0.5,
        "noise_intensity": 0.1,
        "target_ratio": 1.0
    }
    return config

def export_with_config(config):
    # Create augmentation config if needed
    augmentation_config = None
    if config["balance_strategy"] == "augment":
        augmentation_config = {
            'use_basic_augmentation': config["use_basic_augmentation"],
            'use_advanced_augmentation': config["use_advanced_augmentation"], 
            'augmentation_probability': config["augmentation_probability"],
            'augmentation_strength': config["augmentation_strength"],
            'noise_intensity': config["noise_intensity"],
            'target_ratio': config["target_ratio"],
            'seed': config["seed"]
        }

    export_dataset_for_rlhf(
        real_video_paths=config["real_video_dirs"],
        fake_video_paths=config["fake_video_dirs"], 
        output_dir=config["output_dir"],
        balance_strategy=config["balance_strategy"],
        balance_train_only=config["balance_train_only"],
        train_split=config["train_split"],
        val_split=config["val_split"],
        target_fps=config["target_fps"],
        target_resolution=(config["target_height"], config["target_width"]),
        num_frames=config["num_frames"],
        skip_frames=not config["consecutive_frames"],
        augmentation_config=augmentation_config,
        seed=config["seed"]
    )

if __name__ == "__main__":
    config = get_default_config()
    export_with_config(config)
    
    # main()