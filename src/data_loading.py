"""
Data loading utilities for ADNI-like data.

Handles loading cortical thickness, demographics, and other subject-level data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import logging

from .utils import validate_array

logger = logging.getLogger(__name__)


def load_cortical_thickness(
    path: Union[str, Path],
    subject_col: str = "SubjectID",
    roi_prefix: str = "ROI_"
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load cortical thickness data from CSV.
    
    Expected format: CSV with columns [SubjectID, ROI_1, ROI_2, ..., ROI_N, ...]
    
    Args:
        path: Path to CSV file
        subject_col: Name of subject ID column
        roi_prefix: Prefix for ROI columns (e.g., 'ROI_' for 'ROI_1', 'ROI_2', ...)
        
    Returns:
        Tuple of (thickness_matrix, metadata_df)
        - thickness_matrix: (n_subjects, n_rois) array
        - metadata_df: DataFrame with subject IDs and any other non-ROI columns
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Cortical thickness file not found: {path}")
    
    logger.info(f"Loading cortical thickness from {path}")
    
    df = pd.read_csv(path)
    
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found in data")
    
    # Extract ROI columns
    roi_cols = [col for col in df.columns if col.startswith(roi_prefix)]
    
    if len(roi_cols) == 0:
        raise ValueError(f"No ROI columns found with prefix '{roi_prefix}'")
    
    # Extract thickness matrix
    thickness_matrix = df[roi_cols].values.astype(float)
    
    # Extract metadata (non-ROI columns)
    metadata_cols = [col for col in df.columns if col not in roi_cols]
    metadata_df = df[metadata_cols].copy()
    
    logger.info(
        f"Loaded {thickness_matrix.shape[0]} subjects, {thickness_matrix.shape[1]} ROIs"
    )
    
    return thickness_matrix, metadata_df


def load_demographics(
    path: Union[str, Path],
    subject_col: str = "SubjectID"
) -> pd.DataFrame:
    """
    Load demographic/clinical data.
    
    Args:
        path: Path to CSV file
        subject_col: Name of subject ID column
        
    Returns:
        DataFrame with demographic data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Demographics file not found: {path}")
    
    logger.info(f"Loading demographics from {path}")
    
    df = pd.read_csv(path)
    
    if subject_col not in df.columns:
        raise ValueError(f"Subject column '{subject_col}' not found")
    
    return df


def load_roi_labels(
    path: Union[str, Path]
) -> np.ndarray:
    """
    Load ROI labels from text file (one label per line).
    
    Args:
        path: Path to text file
        
    Returns:
        Array of ROI labels (strings)
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"ROI labels file not found: {path}. Using default labels.")
        return None
    
    with open(path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    return np.array(labels)


def merge_subject_data(
    thickness_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    subject_col: str = "SubjectID"
) -> pd.DataFrame:
    """
    Merge cortical thickness and demographic data.
    
    Args:
        thickness_df: DataFrame with thickness data
        demographics_df: DataFrame with demographic data
        subject_col: Column name for subject ID
        
    Returns:
        Merged DataFrame
    """
    merged = pd.merge(
        thickness_df,
        demographics_df,
        on=subject_col,
        how='inner'
    )
    
    logger.info(f"Merged data: {len(merged)} subjects")
    
    return merged


def create_mock_data(
    n_subjects: int = 100,
    n_rois: int = 68,
    seed: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create mock cortical thickness data for testing.
    
    Args:
        n_subjects: Number of subjects
        n_rois: Number of ROIs
        seed: Random seed
        
    Returns:
        Tuple of (thickness_matrix, metadata_df)
    """
    np.random.seed(seed)
    
    # Generate mock thickness (typical range: 1-5 mm)
    thickness_matrix = np.random.normal(2.5, 0.5, size=(n_subjects, n_rois))
    thickness_matrix = np.clip(thickness_matrix, 1.0, 5.0)
    
    # Create metadata
    metadata_df = pd.DataFrame({
        'SubjectID': [f'SUBJ_{i:03d}' for i in range(n_subjects)],
        'Age': np.random.normal(75, 8, n_subjects),
        'Sex': np.random.choice(['M', 'F'], n_subjects),
        'Diagnosis': np.random.choice(['CN', 'MCI', 'AD'], n_subjects, p=[0.3, 0.4, 0.3]),
    })
    
    logger.info(f"Created mock data: {n_subjects} subjects, {n_rois} ROIs")
    
    return thickness_matrix, metadata_df

