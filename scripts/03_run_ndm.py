#!/usr/bin/env python3
"""
Script 03: Run Network Diffusion Model.

Applies NDM to cortical thickness patterns to predict diffusion-based pathology.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.connectome import load_connectome
from src.ndm import run_ndm_batch

logger = setup_logging()


def main(config_path: str = None):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    results_dir = ensure_dir(config.get('output.results_dir', 'results'))
    
    logger.info("=" * 60)
    logger.info("Script 03: Run Network Diffusion Model")
    logger.info("=" * 60)
    
    # Load data
    thickness_path = processed_dir / "cortical_thickness_processed.npy"
    if not thickness_path.exists():
        logger.error(f"Processed thickness not found: {thickness_path}")
        logger.error("Run script 01_prepare_data.py first")
        return
    
    thickness = np.load(thickness_path)
    logger.info(f"Loaded thickness: {thickness.shape}")
    
    # Load connectome
    connectome_path = processed_dir / "connectome.npy"
    if not connectome_path.exists():
        logger.error(f"Connectome not found: {connectome_path}")
        logger.error("Run script 02_build_connectome.py first")
        return
    
    W = np.load(connectome_path)
    logger.info(f"Loaded connectome: {W.shape}")
    
    # NDM parameters
    beta = config.get('ndm.beta', 0.1)
    time_steps = config.get('ndm.time_steps', [1.0])
    normalized = config.get('ndm.normalize_laplacian', True)
    
    logger.info(f"NDM parameters: beta={beta}, time_steps={time_steps}, normalized={normalized}")
    
    # Run NDM for each time step
    if len(time_steps) == 1:
        t = time_steps[0]
        logger.info(f"Running NDM at t={t}...")
        thickness_diff = run_ndm_batch(
            W,
            thickness,
            beta=beta,
            t=t,
            normalized_laplacian=normalized
        )
        
        # Save
        diff_path = results_dir / f"thickness_diffused_t{t}.npy"
        np.save(diff_path, thickness_diff)
        logger.info(f"Saved diffused thickness to {diff_path}")
    else:
        logger.info(f"Running NDM for {len(time_steps)} time steps...")
        thickness_diff = run_ndm_batch(
            W,
            thickness,
            beta=beta,
            t=np.array(time_steps),
            normalized_laplacian=normalized
        )
        
        # Save each time point
        for i, t in enumerate(time_steps):
            diff_path = results_dir / f"thickness_diffused_t{t}.npy"
            np.save(diff_path, thickness_diff[:, i, :])
            logger.info(f"Saved diffused thickness at t={t} to {diff_path}")
        
        # Also save full array
        diff_path_full = results_dir / "thickness_diffused_all_times.npy"
        np.save(diff_path_full, thickness_diff)
        logger.info(f"Saved all time points to {diff_path_full}")
    
    logger.info("=" * 60)
    logger.info("NDM computation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Network Diffusion Model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    
    args = parser.parse_args()
    main(config_path=args.config)

