"""
Statistical analysis and group comparisons.

Provides functions for cluster-wise statistics, clinical associations, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

from .utils import validate_array

logger = logging.getLogger(__name__)


def compute_cluster_statistics(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute summary statistics for each cluster.
    
    Args:
        data: Feature/data matrix (n_subjects, n_features)
        labels: Cluster labels (n_subjects,)
        feature_names: Optional names for features
        
    Returns:
        DataFrame with cluster statistics (mean, std, n) per cluster
    """
    validate_array(data, expected_shape=None, name="data")
    validate_array(labels, expected_shape=(data.shape[0],), name="labels")
    
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    unique_labels = np.unique(labels)
    n_features = data.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    results = []
    
    for label in unique_labels:
        if label == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster_{label}"
        
        mask = labels == label
        cluster_data = data[mask]
        n_subjects = len(cluster_data)
        
        for i, feat_name in enumerate(feature_names):
            values = cluster_data[:, i]
            results.append({
                'Cluster': cluster_name,
                'Feature': feat_name,
                'Mean': np.mean(values),
                'Std': np.std(values, ddof=1),
                'Median': np.median(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'N': n_subjects
            })
    
    return pd.DataFrame(results)


def test_cluster_differences(
    data: np.ndarray,
    labels: np.ndarray,
    feature_idx: Optional[int] = None,
    test: str = "anova"
) -> Dict[str, float]:
    """
    Test for differences between clusters.
    
    Args:
        data: Feature matrix (n_subjects, n_features)
        labels: Cluster labels (n_subjects,)
        feature_idx: Index of feature to test (None for all features)
        test: Statistical test ('anova', 'kruskal', 'ttest')
        
    Returns:
        Dictionary with test statistics and p-values
    """
    validate_array(data, expected_shape=None, name="data")
    validate_array(labels, expected_shape=(data.shape[0],), name="labels")
    
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise
    
    if len(unique_labels) < 2:
        logger.warning("Need at least 2 clusters for comparison")
        return {'statistic': np.nan, 'pvalue': np.nan}
    
    if feature_idx is not None:
        values = data[:, feature_idx]
    else:
        # Use first feature or mean across features
        values = np.mean(data, axis=1)
    
    # Group by cluster
    groups = [values[labels == label] for label in unique_labels]
    
    if test == "anova":
        statistic, pvalue = stats.f_oneway(*groups)
    elif test == "kruskal":
        statistic, pvalue = stats.kruskal(*groups)
    elif test == "ttest":
        if len(groups) != 2:
            raise ValueError("t-test requires exactly 2 groups")
        statistic, pvalue = stats.ttest_ind(groups[0], groups[1])
    else:
        raise ValueError(f"Unknown test: {test}")
    
    return {'statistic': statistic, 'pvalue': pvalue}


def correlate_with_clinical(
    features: np.ndarray,
    clinical: np.ndarray,
    feature_names: Optional[np.ndarray] = None,
    clinical_name: str = "Clinical"
) -> pd.DataFrame:
    """
    Compute correlations between features and clinical variables.
    
    Args:
        features: Feature matrix (n_subjects, n_features)
        clinical: Clinical variable (n_subjects,)
        feature_names: Optional feature names
        clinical_name: Name of clinical variable
        
    Returns:
        DataFrame with correlations and p-values
    """
    validate_array(features, expected_shape=None, name="features")
    validate_array(clinical, expected_shape=(features.shape[0],), name="clinical")
    
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D, got shape {features.shape}")
    
    n_features = features.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    results = []
    
    for i, feat_name in enumerate(feature_names):
        r, p = stats.pearsonr(features[:, i], clinical)
        results.append({
            'Feature': feat_name,
            'Clinical': clinical_name,
            'Correlation': r,
            'P_value': p
        })
    
    return pd.DataFrame(results)


def compute_tmi_by_diagnosis(
    tmi_values: np.ndarray,
    diagnoses: np.ndarray
) -> pd.DataFrame:
    """
    Compute TMI statistics grouped by diagnosis.
    
    Args:
        tmi_values: TMI values (n_subjects,)
        diagnoses: Diagnosis labels (n_subjects,)
        
    Returns:
        DataFrame with TMI statistics per diagnosis
    """
    validate_array(tmi_values, expected_shape=None, name="tmi_values")
    validate_array(diagnoses, expected_shape=tmi_values.shape, name="diagnoses")
    
    unique_diagnoses = np.unique(diagnoses)
    
    results = []
    for diag in unique_diagnoses:
        mask = diagnoses == diag
        tmi_diag = tmi_values[mask]
        
        results.append({
            'Diagnosis': diag,
            'N': len(tmi_diag),
            'Mean': np.mean(tmi_diag),
            'Std': np.std(tmi_diag, ddof=1),
            'Median': np.median(tmi_diag),
            'Min': np.min(tmi_diag),
            'Max': np.max(tmi_diag)
        })
    
    return pd.DataFrame(results)

