#!/usr/bin/env python3
"""
Script 07: Run statistical analysis and create final plots.

Computes cluster statistics, clinical associations, and generates publication figures.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.utils import setup_logging, set_random_seed, ensure_dir
from src.analysis import (
    compute_cluster_statistics,
    test_cluster_differences,
    correlate_with_clinical,
    compute_tmi_by_diagnosis
)
from src.visualization import (
    plot_tmi_distribution,
    plot_cluster_comparison,
    plot_embedding,
    plot_persistence_diagram
)

logger = setup_logging()


def main(config_path: str = None, time_point: float = 1.0):
    """Main function."""
    config = load_config(config_path)
    set_random_seed(config.get('random_seed', 42))
    
    processed_dir = ensure_dir(config.get('data.processed_data_dir', 'data/processed'))
    results_dir = ensure_dir(config.get('output.results_dir', 'results'))
    figures_dir = ensure_dir(config.get('output.figures_dir', 'results/figures'))
    tables_dir = ensure_dir(config.get('output.tables_dir', 'results/tables'))
    
    logger.info("=" * 60)
    logger.info("Script 07: Run Statistics and Plots")
    logger.info("=" * 60)
    
    labels_path = results_dir / f"cluster_labels_t{time_point}.npy"
    tmi_path = results_dir / f"tmi_values_t{time_point}.npy"
    pi_path = results_dir / f"persistence_images_t{time_point}.npy"
    embedding_path = results_dir / f"embedding_t{time_point}.npy"
    
    if not labels_path.exists():
        logger.error("Cluster labels not found. Run script 06_run_clustering.py first")
        return
    
    labels = np.load(labels_path)
    tmi_values = np.load(tmi_path) if tmi_path.exists() else None
    persistence_images = np.load(pi_path) if pi_path.exists() else None
    embedding = np.load(embedding_path) if embedding_path.exists() else None
    
    logger.info(f"Loaded cluster labels: {len(labels)} subjects")
    
    metadata_path = processed_dir / "metadata.csv"
    metadata = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata: {metadata.shape}")
    
    if persistence_images is not None:
        logger.info("Computing cluster statistics...")
        cluster_stats = compute_cluster_statistics(
            persistence_images,
            labels,
            feature_names=[f"PI_{i}" for i in range(persistence_images.shape[1])]
        )
        
        stats_path = tables_dir / f"cluster_statistics_t{time_point}.csv"
        cluster_stats.to_csv(stats_path, index=False)
        logger.info(f"Saved cluster statistics to {stats_path}")
        
        # Test for cluster differences
        test_result = test_cluster_differences(persistence_images, labels, test='anova')
        logger.info(f"ANOVA test: F={test_result['statistic']:.4f}, p={test_result['pvalue']:.4e}")
    
    # TMI analysis
    if tmi_values is not None:
        logger.info("Analyzing TMI...")
        
        diagnoses = None
        if metadata is not None and 'Diagnosis' in metadata.columns:
            diagnoses = metadata['Diagnosis'].values
            plot_path = figures_dir / f"tmi_by_diagnosis_t{time_point}.png"
            plot_tmi_distribution(tmi_values, diagnoses=diagnoses, save_path=plot_path)
            
            tmi_by_diag = compute_tmi_by_diagnosis(tmi_values, diagnoses)
            tmi_stats_path = tables_dir / f"tmi_by_diagnosis_t{time_point}.csv"
            tmi_by_diag.to_csv(tmi_stats_path, index=False)
            logger.info(f"Saved TMI by diagnosis to {tmi_stats_path}")
        else:
            plot_path = figures_dir / f"tmi_distribution_t{time_point}.png"
            plot_tmi_distribution(tmi_values, save_path=plot_path)
        
        if embedding is not None:
            plot_path = figures_dir / f"embedding_by_tmi_t{time_point}.png"
            plot_embedding(embedding, colors=tmi_values, save_path=plot_path, title="Embedding Colored by TMI")
    
    if embedding is not None and metadata is not None and 'Diagnosis' in metadata.columns:
        diagnoses = metadata['Diagnosis'].values
        plot_path = figures_dir / f"embedding_by_diagnosis_t{time_point}.png"
        plot_embedding(embedding, labels=diagnoses, save_path=plot_path, title="Embedding Colored by Diagnosis")
    
    # Clinical correlations
    if metadata is not None and tmi_values is not None:
        logger.info("Computing clinical correlations...")
        
        clinical_vars = ['Age'] if 'Age' in metadata.columns else []
        
        for var in clinical_vars:
            if var in metadata.columns:
                clinical_values = metadata[var].values
                # Remove NaN
                valid_mask = ~np.isnan(clinical_values)
                if np.sum(valid_mask) > 10:  # Need sufficient data
                    corr = correlate_with_clinical(
                        tmi_values[valid_mask].reshape(-1, 1),
                        clinical_values[valid_mask],
                        feature_names=['TMI'],
                        clinical_name=var
                    )
                    corr_path = tables_dir / f"tmi_correlation_{var}_t{time_point}.csv"
                    corr.to_csv(corr_path, index=False)
                    logger.info(f"Saved TMI-{var} correlation to {corr_path}")
    
    diagrams_path = results_dir / f"persistence_diagrams_t{time_point}.npy"
    if diagrams_path.exists():
        diagrams = np.load(diagrams_path, allow_pickle=True)
        if len(diagrams) > 0:
            plot_path = figures_dir / f"persistence_diagram_example_t{time_point}.png"
            plot_persistence_diagram(diagrams[0], save_path=plot_path, title="Example Persistence Diagram")
    
    logger.info("=" * 60)
    logger.info("Statistical analysis complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run statistics and plots")
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

