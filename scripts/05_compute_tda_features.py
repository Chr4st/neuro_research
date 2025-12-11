#!/usr/bin/env python3
"""
Script 05: Compute TDA features.

Computes persistent homology, persistence images, and TMI from residuals.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.tda_features import (
    compute_persistence_diagrams,
    compute_persistence_images,
    compute_tmi_batch,
    get_reference_diagram
)
from src.connectome import load_connectome
from src.utils import apply_baire_category_to_persistence_diagrams

logger = setup_logging()


def main(config_path: str = None, time_point: float = 1.0):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    results_dir = ensure_dir(config.get('output.results_dir', 'results'))
    
    logger.info("=" * 60)
    logger.info("Script 05: Compute TDA Features")
    logger.info("=" * 60)
    
    residual_path = results_dir / f"residuals_t{time_point}.npy"
    if not residual_path.exists():
        logger.error(f"Residuals not found: {residual_path}")
        logger.error("Run script 04_compute_residuals.py first")
        return
    
    residuals = np.load(residual_path)
    logger.info(f"Loaded residuals: {residuals.shape}")
    
    connectome_path = processed_dir / "connectome.npy"
    W = None
    if connectome_path.exists():
        W = np.load(connectome_path)
        logger.info(f"Loaded connectome: {W.shape}")
    
    # TDA parameters
    mode = config.get('tda.filtration_mode', 'node')
    max_dim = config.get('tda.max_dimension', 1)
    
    logger.info(f"TDA parameters: mode={mode}, max_dimension={max_dim}")
    
    logger.info("Computing persistence diagrams...")
    diagrams = compute_persistence_diagrams(
        residuals,
        W=W,
        mode=mode,
        max_dimension=max_dim
    )
    
    diagrams_path = results_dir / f"persistence_diagrams_t{time_point}.npy"
    np.save(diagrams_path, diagrams, allow_pickle=True)
    logger.info(f"Saved {len(diagrams)} persistence diagrams to {diagrams_path}")
    
    logger.info("Computing persistence images...")
    resolution = tuple(config.get('tda.persistence_image_resolution', [20, 20]))
    bandwidth = config.get('tda.persistence_image_bandwidth', 0.1)
    
    persistence_images = compute_persistence_images(
        diagrams,
        resolution=resolution,
        bandwidth=bandwidth
    )
    
    pi_path = results_dir / f"persistence_images_t{time_point}.npy"
    np.save(pi_path, persistence_images)
    logger.info(f"Saved persistence images: {persistence_images.shape}")
    
    logger.info("Computing reference diagram...")
    reference_diagram = get_reference_diagram(diagrams, method='mean')
    
    ref_path = results_dir / f"reference_diagram_t{time_point}.npy"
    np.save(ref_path, reference_diagram)
    logger.info(f"Saved reference diagram to {ref_path}")
    
    logger.info("Computing Topological Misalignment Index (TMI)...")
    metric = config.get('tda.tmi_metric', 'wasserstein')
    p = config.get('tda.tmi_p', 2)
    
    tmi_values = compute_tmi_batch(
        diagrams,
        reference_diagram,
        metric=metric,
        p=p
    )
    
    tmi_path = results_dir / f"tmi_values_t{time_point}.npy"
    np.save(tmi_path, tmi_values)
    logger.info(f"Saved TMI values: mean={np.mean(tmi_values):.4f}, std={np.std(tmi_values):.4f}")
    
    # Apply Baire Category Theorem analysis
    logger.info("Applying Baire Category Theorem analysis to persistence diagrams...")
    try:
        baire_results = apply_baire_category_to_persistence_diagrams(
            diagrams,
            reference_diagram,
            metric=metric
        )
        logger.info(f"Baire Category Analysis:")
        logger.info(f"  Space completeness: {baire_results['is_complete']}")
        logger.info(f"  Baire property holds: {baire_results['baire_property']}")
        logger.info(f"  Number of diagrams analyzed: {baire_results['n_diagrams']}")
        
        import json
        baire_path = results_dir / f"baire_analysis_t{time_point}.json"
        with open(baire_path, 'w') as f:
            json.dump(baire_results, f, indent=2)
        logger.info(f"Saved Baire analysis to {baire_path}")
    except Exception as e:
        logger.warning(f"Baire Category Theorem analysis failed: {e}")
    
    logger.info("=" * 60)
    logger.info("TDA feature computation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TDA features")
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
        help="Time point to use"
    )
    
    args = parser.parse_args()
    main(config_path=args.config, time_point=args.time_point)

