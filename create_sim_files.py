#!/usr/bin/env python3
"""
Create similarity files for Isharah dataset variants using sentence embeddings.
Based on: https://github.com/GASLT-SLTC/GASLT/issues
"""

import json
import pickle
import gzip
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def create_sim_files(data_dir, output_dir, dataset_name):
    """
    Create name_to_video_id.json and cos_sim.pkl files using sentence embeddings.

    Args:
        data_dir: Directory containing the pickle dataset files
        output_dir: Output directory for sim files
        dataset_name: Name prefix for dataset files (e.g., 'isharah')
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique video names and their texts across all splits
    video_data = {}  # {name: text}

    for split in ['train', 'val', 'test']:
        data_file = Path(data_dir) / f"{dataset_name}.{split}"
        if not data_file.exists():
            print(f"Warning: {data_file} not found, skipping")
            continue

        with gzip.open(data_file, 'rb') as f:
            samples = pickle.load(f)
            for sample in samples:
                name = sample['name']
                text = sample['text']
                if name not in video_data:
                    video_data[name] = text
                else:
                    # Verify same video has same text across splits
                    assert video_data[name] == text, f"Inconsistent text for {name}"

    print(f"Found {len(video_data)} unique videos")

    # Create name_to_video_id mapping (sorted for consistency)
    sorted_names = sorted(video_data.keys())
    name_to_video_id = {name: idx for idx, name in enumerate(sorted_names)}

    # Save name_to_video_id.json
    json_path = output_dir / 'name_to_video_id.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(name_to_video_id, f, indent=2, ensure_ascii=False)
    print(f"Saved {json_path}")

    # Get sentences in the same order as name_to_video_id
    sentences = [video_data[name] for name in sorted_names]

    # Initialize multilingual sentence transformer
    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

    # Generate embeddings
    print("Generating sentence embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)

    # Compute cosine similarity matrix
    print("Computing cosine similarity matrix...")
    cos_sim_matrix = cosine_similarity(embeddings)

    # Convert to torch tensor
    cos_sim = torch.from_numpy(cos_sim_matrix).float()

    # Save cos_sim.pkl
    pkl_path = output_dir / 'cos_sim.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(cos_sim, f)
    print(f"Saved {pkl_path} (shape: {cos_sim.shape})")

    # Print some statistics
    print(f"\nCosine similarity statistics:")
    print(f"  Min: {cos_sim.min():.4f}")
    print(f"  Max: {cos_sim.max():.4f}")
    print(f"  Mean: {cos_sim.mean():.4f}")
    print(f"  Diagonal (self-similarity): {cos_sim.diag().mean():.4f}")

    return len(sorted_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create similarity files for dataset")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing dataset pickle files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for sim files"
    )
    parser.add_argument(
        "--dataset-name",
        default="isharah",
        help="Dataset name prefix (default: isharah)"
    )

    args = parser.parse_args()

    n_videos = create_sim_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )

    print(f"\nCreated sim files for {n_videos} videos")
