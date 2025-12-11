#!/usr/bin/env python3
"""
Script 04: Compute residuals.

Computes residuals: r = x_obs - x_diff and saves diagnostic plots.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.residuals import compute_residual, compute_residual_stats
from src.residuals import plot_residual_distribution, plot_residual_vs_predicted

logger = setup_logging()


def main(config_path: str = None, time_point: float = 1.0):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    results_dir = ensure_dir(config.get('output.results_dir', 'results'))
    figures_dir = ensure_dir(config.get('output.figures_dir', 'results/figures'))
    
    logger.info("=" * 60)
    logger.info("Script 04: Compute Residuals")
    logger.info("=" * 60)
    
    # Load observed thickness
    thickness_path = processed_dir / "cortical_thickness_processed.npy"
    if not thickness_path.exists():
        logger.error(f"Processed thickness not found: {thickness_path}")
        logger.error("Run script 01_prepare_data.py first")
        return
    
    thickness_obs = np.load(thickness_path)
    logger.info(f"Loaded observed thickness: {thickness_obs.shape}")
    
    # Load diffused thickness
    diff_path = results_dir / f"thickness_diffused_t{time_point}.npy"
    if not diff_path.exists():
        logger.error(f"Diffused thickness not found: {diff_path}")
        logger.error("Run script 03_run_ndm.py first")
        return
    
    thickness_diff = np.load(diff_path)
    logger.info(f"Loaded diffused thickness: {thickness_diff.shape}")
    
    # Compute residuals
    logger.info("Computing residuals...")
    residuals = compute_residual(thickness_obs, thickness_diff)
    
    # Save residuals
    residual_path = results_dir / f"residuals_t{time_point}.npy"
    np.save(residual_path, residuals)
    logger.info(f"Saved residuals to {residual_path}")
    
    # Compute statistics
    stats = compute_residual_stats(residuals)
    logger.info(f"Residual statistics:")
    logger.info(f"  Mean: {np.mean(residuals):.4f}")
    logger.info(f"  Std: {np.std(residuals):.4f}")
    logger.info(f"  Min: {np.min(residuals):.4f}")
    logger.info(f"  Max: {np.max(residuals):.4f}")
    
    # Create diagnostic plots
    logger.info("Creating diagnostic plots...")
    
    plot_path = figures_dir / f"residual_distribution_t{time_point}.png"
    plot_residual_distribution(residuals, save_path=plot_path)
    
    plot_path = figures_dir / f"residual_vs_predicted_t{time_point}.png"
    plot_residual_vs_predicted(thickness_obs, thickness_diff, residuals, save_path=plot_path)
    
    logger.info("=" * 60)
    logger.info("Residual computation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute residuals")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--time-point",
        type=float,
        default=1.0,
        help="Time point to use for residuals"
    )
    
    args = parser.parse_args()
    main(config_path=args.config, time_point=args.time_point)

