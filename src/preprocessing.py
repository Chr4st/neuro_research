"""
Data preprocessing utilities.

Handles z-scoring, ROI alignment, missing value handling, etc.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
import logging

from .utils import validate_array, zscore

logger = logging.getLogger(__name__)


def zscore_cortical_thickness(
    thickness: np.ndarray,
    axis: int = 0,
    reference_group: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Z-score cortical thickness values.
    
    Args:
        thickness: Cortical thickness (n_subjects, n_rois) or (n_rois,)
        axis: Axis along which to compute z-scores (0=across subjects, 1=across ROIs)
        reference_group: Optional reference group for z-scoring (uses its mean/std)
        
    Returns:
        Z-scored thickness values
    """
    validate_array(thickness, expected_shape=None, name="thickness")
    
    if reference_group is not None:
        mean_ref = np.mean(reference_group, axis=axis, keepdims=True)
        std_ref = np.std(reference_group, axis=axis, keepdims=True, ddof=1)
        
        # Avoid division by zero
        std_ref = np.where(std_ref == 0, 1.0, std_ref)
        
        thickness_z = (thickness - mean_ref) / std_ref
    else:
        thickness_z = zscore(thickness, axis=axis)
    
    logger.info(f"Z-scored cortical thickness: mean={np.mean(thickness_z):.4f}, std={np.std(thickness_z):.4f}")
    
    return thickness_z


def handle_missing_values(
    data: np.ndarray,
    method: str = "mean",
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values (NaN) in data.
    
    Args:
        data: Input data array
        method: Imputation method ('mean', 'median', 'zero', 'drop')
        axis: Axis along which to compute statistics for imputation
        drop: If True, drop rows/columns with missing values
        
    Returns:
        Tuple of (imputed_data, mask) where mask indicates which values were imputed
    """
    validate_array(data, expected_shape=None, name="data")
    
    missing_mask = np.isnan(data)
    n_missing = np.sum(missing_mask)
    
    if n_missing == 0:
        logger.debug("No missing values found")
        return data.copy(), np.zeros_like(data, dtype=bool)
    
    logger.info(f"Found {n_missing} missing values ({100*n_missing/data.size:.2f}%)")
    
    data_imputed = data.copy()
    
    if method == "mean":
        fill_value = np.nanmean(data, axis=axis, keepdims=True)
        data_imputed[missing_mask] = np.take(fill_value, np.where(missing_mask)[axis])
    elif method == "median":
        fill_value = np.nanmedian(data, axis=axis, keepdims=True)
        data_imputed[missing_mask] = np.take(fill_value, np.where(missing_mask)[axis])
    elif method == "zero":
        data_imputed[missing_mask] = 0.0
    elif method == "drop":
        # Drop rows/columns with any missing values
        if axis == 0:
            valid_rows = ~np.any(missing_mask, axis=1)
            data_imputed = data_imputed[valid_rows]
        else:
            valid_cols = ~np.any(missing_mask, axis=0)
            data_imputed = data_imputed[:, valid_cols]
        missing_mask = np.zeros_like(data_imputed, dtype=bool)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    
    return data_imputed, missing_mask


def align_rois(
    data1: np.ndarray,
    data2: np.ndarray,
    labels1: Optional[np.ndarray] = None,
    labels2: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align ROI order between two datasets.
    
    Args:
        data1: First dataset (n_subjects, n_rois)
        data2: Second dataset (n_subjects, n_rois) or (n_rois,)
        labels1: ROI labels for first dataset
        labels2: ROI labels for second dataset
        
    Returns:
        Tuple of (aligned_data1, aligned_data2)
    """
    if labels1 is None or labels2 is None:
        # Assume same order if no labels provided
        if data1.shape[-1] != data2.shape[-1]:
            raise ValueError(
                f"Cannot align: data1 has {data1.shape[-1]} ROIs, "
                f"data2 has {data2.shape[-1]} ROIs"
            )
        return data1, data2
    
    common_labels = np.intersect1d(labels1, labels2)
    
    if len(common_labels) == 0:
        raise ValueError("No common ROIs found between datasets")
    
    indices1 = np.array([np.where(labels1 == label)[0][0] for label in common_labels])
    indices2 = np.array([np.where(labels2 == label)[0][0] for label in common_labels])
    
    # Reorder data
    if data1.ndim == 2:
        aligned_data1 = data1[:, indices1]
    else:
        aligned_data1 = data1[indices1]
    
    if data2.ndim == 2:
        aligned_data2 = data2[:, indices2]
    else:
        aligned_data2 = data2[indices2]
    
    logger.info(f"Aligned {len(common_labels)} common ROIs")
    
    return aligned_data1, aligned_data2


def normalize_to_reference(
    data: np.ndarray,
    reference: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize data to a reference group.
    
    Args:
        data: Data to normalize (n_subjects, n_rois)
        reference: Reference data (n_ref_subjects, n_rois)
        method: Normalization method ('zscore', 'percentile')
        
    Returns:
        Normalized data
    """
    validate_array(data, expected_shape=None, name="data")
    validate_array(reference, expected_shape=None, name="reference")
    
    if data.shape[-1] != reference.shape[-1]:
        raise ValueError("ROI dimension mismatch")
    
    if method == "zscore":
        return zscore_cortical_thickness(data, reference_group=reference)
    elif method == "percentile":
        # Map to percentiles of reference distribution
        normalized = np.zeros_like(data)
        for i in range(data.shape[-1]):
            ref_values = reference[:, i] if reference.ndim == 2 else reference
            data_values = data[:, i] if data.ndim == 2 else data[i]
            normalized[:, i] = np.array([
                np.percentile(ref_values, 100 * (ref_values < val).mean())
                for val in data_values
            ])
        return normalized
    else:
        raise ValueError(f"Unknown normalization method: {method}")

