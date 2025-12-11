"""
Residual computation and diagnostics.

Computes residuals: r = x_obs - x_diff
Provides summary statistics and quality control plots.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from .utils import validate_array

logger = logging.getLogger(__name__)


def compute_residual(
    x_obs: np.ndarray,
    x_diff: np.ndarray
) -> np.ndarray:
    """
    Compute residual: r = x_obs - x_diff.
    
    Args:
        x_obs: Observed pattern (n_rois,) or (n_subjects, n_rois)
        x_diff: Diffusion-predicted pattern (same shape as x_obs)
        
    Returns:
        Residual pattern (same shape as input)
    """
    validate_array(x_obs, expected_shape=None, name="observed_pattern")
    validate_array(x_diff, expected_shape=x_obs.shape, name="diffused_pattern")
    
    r = x_obs - x_diff
    
    logger.debug(f"Computed residuals: shape {r.shape}, mean {np.mean(r):.4f}, std {np.std(r):.4f}")
    
    return r


def compute_residual_stats(
    residuals: np.ndarray,
    roi_labels: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute summary statistics for residuals.
    
    Args:
        residuals: Residual patterns (n_subjects, n_rois) or (n_rois,)
        roi_labels: Optional ROI labels for reporting
        
    Returns:
        Dictionary with statistics:
        - 'mean': Mean residual per ROI
        - 'std': Std residual per ROI
        - 'abs_mean': Mean absolute residual per ROI
        - 'subject_mean': Mean residual per subject
        - 'subject_std': Std residual per subject
    """
    residuals = np.atleast_2d(residuals)
    n_subjects, n_rois = residuals.shape
    
    stats = {
        'mean': np.mean(residuals, axis=0),  # Per ROI
        'std': np.std(residuals, axis=0, ddof=1),
        'abs_mean': np.mean(np.abs(residuals), axis=0),
        'subject_mean': np.mean(residuals, axis=1),  # Per subject
        'subject_std': np.std(residuals, axis=1, ddof=1),
    }
    
    if roi_labels is not None:
        stats['roi_labels'] = roi_labels
    
    logger.info(
        f"Residual stats: ROI mean range [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}], "
        f"subject mean range [{stats['subject_mean'].min():.4f}, {stats['subject_mean'].max():.4f}]"
    )
    
    return stats


def plot_residual_distribution(
    residuals: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Residual Distribution"
) -> None:
    """
    Plot histogram of residual values.
    
    Args:
        residuals: Residual patterns (n_subjects, n_rois) or flattened
        save_path: Optional path to save figure
        title: Plot title
    """
    residuals_flat = residuals.flatten()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(residuals_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(np.mean(residuals_flat), color='green', linestyle='--', linewidth=2, label='Mean')
    ax.set_xlabel('Residual Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residual distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residual_vs_predicted(
    x_obs: np.ndarray,
    x_diff: np.ndarray,
    residuals: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Residuals vs Predicted"
) -> None:
    """
    Plot residuals vs predicted values (scatter plot).
    
    Args:
        x_obs: Observed patterns (n_subjects, n_rois) or (n_rois,)
        x_diff: Predicted patterns (same shape)
        residuals: Optional precomputed residuals
        save_path: Optional path to save figure
        title: Plot title
    """
    if residuals is None:
        residuals = compute_residual(x_obs, x_diff)
    
    x_obs_flat = x_obs.flatten()
    x_diff_flat = x_diff.flatten()
    residuals_flat = residuals.flatten()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(x_diff_flat, residuals_flat, alpha=0.5, s=10, c=x_obs_flat, cmap='viridis')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Value (x_diff)')
    ax.set_ylabel('Residual (x_obs - x_diff)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Observed Value')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved residual vs predicted plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roi_residuals(
    residuals: np.ndarray,
    roi_labels: Optional[np.ndarray] = None,
    top_n: int = 20,
    save_path: Optional[Path] = None,
    title: str = "Top ROI Residuals"
) -> None:
    """
    Plot mean absolute residuals per ROI (top N).
    
    Args:
        residuals: Residual patterns (n_subjects, n_rois)
        roi_labels: Optional ROI labels
        top_n: Number of top ROIs to show
        save_path: Optional path to save figure
        title: Plot title
    """
    residuals = np.atleast_2d(residuals)
    mean_abs_residuals = np.mean(np.abs(residuals), axis=0)
    
    top_indices = np.argsort(mean_abs_residuals)[-top_n:][::-1]
    top_values = mean_abs_residuals[top_indices]
    
    if roi_labels is not None:
        top_labels = [roi_labels[i] for i in top_indices]
    else:
        top_labels = [f"ROI {i}" for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_labels)), top_values)
    ax.set_yticks(range(len(top_labels)))
    ax.set_yticklabels(top_labels)
    ax.set_xlabel('Mean Absolute Residual')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROI residuals plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

