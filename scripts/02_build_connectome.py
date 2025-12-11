#!/usr/bin/env python3
"""
Script 02: Build or load structural connectome.

Loads or generates a structural connectome and computes the Laplacian.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.connectome import load_connectome, compute_laplacian, validate_connectome

logger = setup_logging()


def create_mock_connectome(n_rois: int = 68, seed: int = 42) -> np.ndarray:
    """Create a mock structural connectome for testing."""
    np.random.seed(seed)
    
    W = np.random.rand(n_rois, n_rois)
    W = (W + W.T) / 2  # Symmetrize
    W = W * (W > 0.3)  # Threshold for sparsity
    np.fill_diagonal(W, 0)  # No self-loops
    
    # Normalize
    W = W / np.max(W)
    
    return W


def main(config_path: str = None, use_mock: bool = False):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    
    logger.info("=" * 60)
    logger.info("Script 02: Build Connectome")
    logger.info("=" * 60)
    
    connectome_path = config.get('data.connectome_path')
    
    if use_mock or (connectome_path is None or not Path(connectome_path).exists()):
        logger.info("Generating mock connectome...")
        
        roi_count_path = processed_dir / "n_rois.txt"
        if roi_count_path.exists():
            with open(roi_count_path, 'r') as f:
                n_rois = int(f.read().strip())
        else:
            n_rois = 68
            logger.warning(f"ROI count file not found, using default {n_rois}")
        
        W = create_mock_connectome(n_rois=n_rois, seed=config.get('random_seed', 42))
        
        connectome_path = processed_dir / "connectome.npy"
        np.save(connectome_path, W)
        logger.info(f"Saved mock connectome to {connectome_path}")
    else:
        logger.info(f"Loading connectome from {connectome_path}")
        W = load_connectome(connectome_path)
    
    validate_connectome(W)
    logger.info(f"Connectome: {W.shape[0]} ROIs, {np.count_nonzero(W)} edges")
    
    normalized = config.get('ndm.normalize_laplacian', True)
    logger.info(f"Computing {'normalized' if normalized else 'unnormalized'} Laplacian...")
    L = compute_laplacian(W, normalized=normalized)
    
    laplacian_path = processed_dir / "laplacian.npy"
    np.save(laplacian_path, L)
    logger.info(f"Saved Laplacian to {laplacian_path}")
    
    # Also save connectome if it wasn't already saved
    if not use_mock and connectome_path != processed_dir / "connectome.npy":
        connectome_save_path = processed_dir / "connectome.npy"
        np.save(connectome_save_path, W)
        logger.info(f"Saved connectome copy to {connectome_save_path}")
    
    logger.info("=" * 60)
    logger.info("Connectome preparation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or load connectome")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Generate mock connectome"
    )
    
    args = parser.parse_args()
    main(config_path=args.config, use_mock=args.use_mock)

