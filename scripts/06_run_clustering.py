#!/usr/bin/env python3
"""
Script 06: Run clustering for subtype discovery.

Performs dimensionality reduction and clustering on TDA features + TMI.
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
from src.clustering import discover_subtypes
from src.visualization import plot_embedding

logger = setup_logging()


def main(config_path: str = None, time_point: float = 1.0):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    results_dir = ensure_dir(config.get('output.results_dir', 'results'))
    figures_dir = ensure_dir(config.get('output.figures_dir', 'results/figures'))
    
    logger.info("=" * 60)
    logger.info("Script 06: Run Clustering")
    logger.info("=" * 60)
    
    # Load TDA features
    pi_path = results_dir / f"persistence_images_t{time_point}.npy"
    tmi_path = results_dir / f"tmi_values_t{time_point}.npy"
    
    if not pi_path.exists() or not tmi_path.exists():
        logger.error("TDA features not found. Run script 05_compute_tda_features.py first")
        return
    
    persistence_images = np.load(pi_path)
    tmi_values = np.load(tmi_path)
    
    logger.info(f"Loaded persistence images: {persistence_images.shape}")
    logger.info(f"Loaded TMI values: {tmi_values.shape}")
    
    # Combine features: persistence images + TMI
    tmi_reshaped = tmi_values.reshape(-1, 1)
    features = np.hstack([persistence_images, tmi_reshaped])
    
    logger.info(f"Combined feature matrix: {features.shape}")
    
    # Clustering parameters
    n_components = config.get('clustering.n_components', 10)
    reduction_method = config.get('clustering.reduction_method', 'umap')
    clustering_method = config.get('clustering.clustering_method', 'hdbscan')
    
    reduction_kwargs = {
        'n_neighbors': config.get('clustering.umap_n_neighbors', 15),
        'min_dist': config.get('clustering.umap_min_dist', 0.1),
    }
    
    clustering_kwargs = {
        'min_cluster_size': config.get('clustering.hdbscan_min_cluster_size', 5),
        'min_samples': config.get('clustering.hdbscan_min_samples', 3),
    }
    
    logger.info(f"Clustering parameters:")
    logger.info(f"  Reduction: {reduction_method}, n_components={n_components}")
    logger.info(f"  Clustering: {clustering_method}")
    
    # Run clustering
    logger.info("Running clustering...")
    results = discover_subtypes(
        features,
        n_components=n_components,
        reduction_method=reduction_method,
        clustering_method=clustering_method,
        reduction_kwargs=reduction_kwargs,
        clustering_kwargs=clustering_kwargs,
        random_state=config.get('random_seed', 42)
    )
    
    labels = results['labels']
    embedding = results['embedding']
    
    # Save results
    labels_path = results_dir / f"cluster_labels_t{time_point}.npy"
    np.save(labels_path, labels)
    logger.info(f"Saved cluster labels to {labels_path}")
    
    embedding_path = results_dir / f"embedding_t{time_point}.npy"
    np.save(embedding_path, embedding)
    logger.info(f"Saved embedding to {embedding_path}")
    
    # Create summary
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1)
    
    logger.info(f"Clustering results:")
    logger.info(f"  Number of clusters: {n_clusters}")
    logger.info(f"  Noise points: {n_noise}")
    for label in unique_labels:
        if label != -1:
            n_in_cluster = np.sum(labels == label)
            logger.info(f"  Cluster {label}: {n_in_cluster} subjects")
    
    # Save cluster assignments as CSV
    metadata_path = processed_dir / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        metadata['Cluster'] = labels
        cluster_csv_path = results_dir / f"cluster_assignments_t{time_point}.csv"
        metadata.to_csv(cluster_csv_path, index=False)
        logger.info(f"Saved cluster assignments to {cluster_csv_path}")
    
    # Plot embedding
    logger.info("Creating embedding plot...")
    plot_path = figures_dir / f"embedding_clusters_t{time_point}.png"
    plot_embedding(embedding, labels=labels, save_path=plot_path, title="Subtype Clusters")
    
    logger.info("=" * 60)
    logger.info("Clustering complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering")
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

