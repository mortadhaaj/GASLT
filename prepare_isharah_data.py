#!/usr/bin/env python3
"""
Script to convert Isharah dataset from CSV format to the pickle format
expected by SignTranslationDataset.
"""

import pandas as pd
import torch
import pickle
import gzip
from pathlib import Path
from tqdm import tqdm
import os

def load_features(video_name, features_dir):
    """Load and reshape features for a video."""
    # Remove 'train/' prefix if present
    video_file = video_name.split('/')[-1]
    feature_path = features_dir / f"{video_file}.pt"

    if not feature_path.exists():
        print(f"Warning: Feature file not found: {feature_path}")
        return None

    # Load features (list of tensors with shape [1, 1024, 2, 2])
    features = torch.load(feature_path)

    # Reshape to [num_frames, 1024]
    # Each tensor is [1, 1024, 2, 2], we need to flatten spatial dims
    reshaped_features = []
    for feat in features:
        # feat shape: [1, 1024, 2, 2]
        # Take mean over spatial dimensions (2, 2)
        feat_pooled = feat.mean(dim=[2, 3])  # Shape: [1, 1024]
        reshaped_features.append(feat_pooled.squeeze(0))  # Shape: [1024]

    # Stack to get [num_frames, 1024]
    feature_tensor = torch.stack(reshaped_features, dim=0)

    return feature_tensor


def create_dataset_files(
    csv_path,
    features_dir,
    output_dir,
    use_english=True
):
    """
    Convert CSV dataset to pickle format.

    Args:
        csv_path: Path to dataset_random.csv
        features_dir: Path to features directory (e.g., /tmp/isharah_data/dataset_256/features/i3d-features/span=8_stride=2)
        output_dir: Output directory for pickle files
        use_english: If True, use English translation; otherwise use Arabic sentence
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(features_dir)

    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        split_df = df[df['split'] == split]

        if len(split_df) == 0:
            print(f"Warning: No samples found for {split} split")
            continue

        samples = []
        skipped = 0

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df)):
            video_name = row['video_pth']

            # Load features
            features = load_features(video_name, features_dir)

            if features is None:
                skipped += 1
                continue

            # Extract video ID as name (e.g., "00_0001" from "train/00_0001.mp4")
            name = Path(video_name).stem

            # Create sample in expected format
            sample = {
                'name': name,
                'signer': 'unknown',  # No signer info in CSV
                'gloss': '',  # No gloss annotations in CSV
                'text': row['sentence_en'] if use_english else row['sentence'],
                'sign': features
            }

            samples.append(sample)

        print(f"Processed {len(samples)} samples ({skipped} skipped due to missing features)")

        # Save to pickle file
        output_file = output_dir / f"isharah.{split}"
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(samples, f)

        print(f"Saved to {output_file}")

    print("\nDataset conversion complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Isharah dataset to pickle format")
    parser.add_argument(
        "--csv",
        default="/tmp/isharah_data/dataset_random.csv",
        help="Path to dataset_random.csv"
    )
    parser.add_argument(
        "--features",
        default="/tmp/isharah_data/dataset_256/features/i3d-features/span=8_stride=2",
        help="Path to features directory"
    )
    parser.add_argument(
        "--output",
        default="./data/isharah/data",
        help="Output directory for pickle files"
    )
    parser.add_argument(
        "--use-english",
        action="store_true",
        help="Use English translation instead of Arabic text"
    )
    parser.add_argument(
        "--use-arabic",
        action="store_true",
        default=True,
        help="Use Arabic text (default)"
    )

    args = parser.parse_args()

    use_english = args.use_english

    create_dataset_files(
        csv_path=args.csv,
        features_dir=args.features,
        output_dir=args.output,
        use_english=use_english
    )
