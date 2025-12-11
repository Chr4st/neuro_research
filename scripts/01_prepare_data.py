#!/usr/bin/env python3
"""
Script 01: Prepare and preprocess ADNI data.

Loads raw cortical thickness data, applies preprocessing, and saves
processed matrices for downstream analysis.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.data_loading import load_cortical_thickness, create_mock_data
from src.preprocessing import zscore_cortical_thickness, handle_missing_values

logger = setup_logging()


def main(config_path: str = None, use_mock: bool = False):
    """Main function."""
    # Load config
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    # Setup paths
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    
    logger.info("=" * 60)
    logger.info("Script 01: Prepare Data")
    logger.info("=" * 60)
    
    if use_mock:
        logger.info("Using mock data for demonstration")
        thickness, metadata = create_mock_data(
            n_subjects=100,
            n_rois=68,
            seed=config.get('random_seed', 42)
        )
    else:
        # Load real data
        thickness_path = config.get('data.cortical_thickness_path')
        if thickness_path is None or not Path(thickness_path).exists():
            logger.error(f"Cortical thickness file not found: {thickness_path}")
            logger.info("Use --use-mock to generate mock data for testing")
            return
        
        logger.info(f"Loading cortical thickness from {thickness_path}")
        thickness, metadata = load_cortical_thickness(thickness_path)
    
    logger.info(f"Loaded data: {thickness.shape[0]} subjects, {thickness.shape[1]} ROIs")
    
    # Handle missing values
    thickness_clean, missing_mask = handle_missing_values(
        thickness,
        method="mean",
        axis=0
    )
    
    if np.any(missing_mask):
        logger.info(f"Imputed {np.sum(missing_mask)} missing values")
    
    # Z-score
    logger.info("Z-scoring cortical thickness...")
    thickness_z = zscore_cortical_thickness(thickness_clean, axis=0)
    
    # Save processed data
    thickness_path = processed_dir / "cortical_thickness_processed.npy"
    np.save(thickness_path, thickness_z)
    logger.info(f"Saved processed thickness to {thickness_path}")
    
    metadata_path = processed_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Save ROI count
    roi_count_path = processed_dir / "n_rois.txt"
    with open(roi_count_path, 'w') as f:
        f.write(str(thickness_z.shape[1]))
    
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ADNI data")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real ADNI data"
    )
    
    args = parser.parse_args()
    main(config_path=args.config, use_mock=args.use_mock)

