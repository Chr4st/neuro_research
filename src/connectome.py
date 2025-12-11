"""
Structural connectome loading and processing.

Handles loading, validation, and Laplacian computation for structural connectomes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional, Tuple
import logging
from scipy import sparse

from .utils import validate_array

logger = logging.getLogger(__name__)


def load_connectome(
    path: Union[str, Path],
    format: str = "auto"
) -> np.ndarray:
    """
    Load structural connectome from file.
    
    Supports .npy (numpy), .csv (CSV), and .txt (text) formats.
    
    Args:
        path: Path to connectome file
        format: File format ('auto', 'npy', 'csv', 'txt')
        
    Returns:
        Connectome matrix (n_rois x n_rois)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Connectome file not found: {path}")
    
    if format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".npy":
            format = "npy"
        elif suffix == ".csv":
            format = "csv"
        elif suffix in [".txt", ".dat"]:
            format = "txt"
        else:
            raise ValueError(f"Unknown file format: {suffix}")
    
    logger.info(f"Loading connectome from {path} (format: {format})")
    
    if format == "npy":
        W = np.load(path)
    elif format == "csv":
        W = pd.read_csv(path, header=None).values
    elif format == "txt":
        W = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    validate_connectome(W)
    logger.info(f"Loaded connectome: shape {W.shape}, density {np.count_nonzero(W) / W.size:.4f}")
    
    return W


def validate_connectome(W: np.ndarray) -> None:
    """
    Validate connectome matrix properties.
    
    Checks for:
    - Square matrix
    - Symmetry
    - Non-negative values
    - No self-loops (diagonal should be zero)
    
    Args:
        W: Connectome matrix
        
    Raises:
        ValueError: If validation fails
    """
    validate_array(W, expected_shape=None, name="connectome")
    
    if W.ndim != 2:
        raise ValueError("Connectome must be 2D")
    
    if W.shape[0] != W.shape[1]:
        raise ValueError(f"Connectome must be square, got shape {W.shape}")
    
    # Check symmetry (allow small numerical errors)
    if not np.allclose(W, W.T, atol=1e-10):
        raise ValueError("Connectome must be symmetric")
    
    # Check non-negativity
    if np.any(W < -1e-10):  # Allow small numerical errors
        raise ValueError("Connectome must have non-negative edge weights")
    
    # Warn if diagonal is not zero
    if np.any(np.abs(np.diag(W)) > 1e-10):
        logger.warning("Connectome has non-zero diagonal (self-loops)")


def compute_laplacian(
    W: np.ndarray,
    normalized: bool = True,
    sparse_format: bool = False
) -> Union[np.ndarray, sparse.csr_matrix]:
    """
    Compute graph Laplacian from adjacency matrix.
    
    For unnormalized: L = D - W, where D is degree matrix
    For normalized: L = I - D^(-1/2) W D^(-1/2)
    
    Args:
        W: Adjacency/connectome matrix (n x n)
        normalized: If True, compute normalized Laplacian
        sparse_format: If True, return sparse matrix
        
    Returns:
        Laplacian matrix (n x n)
    """
    validate_connectome(W)
    
    n = W.shape[0]
    
    if sparse_format:
        W_sparse = sparse.csr_matrix(W)
        degrees = np.array(W_sparse.sum(axis=1)).flatten()
    else:
        degrees = np.sum(W, axis=1)
    
    if normalized:
        # Avoid division by zero for isolated nodes
        degrees_sqrt_inv = np.zeros_like(degrees)
        mask = degrees > 0
        degrees_sqrt_inv[mask] = 1.0 / np.sqrt(degrees[mask])
        
        if sparse_format:
            D_inv_sqrt = sparse.diags(degrees_sqrt_inv)
            L = sparse.eye(n) - D_inv_sqrt @ W_sparse @ D_inv_sqrt
        else:
            D_inv_sqrt = np.diag(degrees_sqrt_inv)
            L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        if sparse_format:
            D = sparse.diags(degrees)
            L = D - W_sparse
        else:
            D = np.diag(degrees)
            L = D - W
    
    logger.debug(f"Computed {'normalized' if normalized else 'unnormalized'} Laplacian")
    
    return L


def get_degree_matrix(W: np.ndarray) -> np.ndarray:
    """
    Compute degree matrix from adjacency matrix.
    
    Args:
        W: Adjacency/connectome matrix (n x n)
        
    Returns:
        Degree matrix (n x n, diagonal)
    """
    validate_connectome(W)
    degrees = np.sum(W, axis=1)
    return np.diag(degrees)


def threshold_connectome(
    W: np.ndarray,
    threshold: float,
    method: str = "absolute"
) -> np.ndarray:
    """
    Threshold connectome to keep only strong connections.
    
    Args:
        W: Connectome matrix (n x n)
        threshold: Threshold value
        method: 'absolute' (keep |w| > threshold) or 'percentile' (keep top percentile)
        
    Returns:
        Thresholded connectome matrix
    """
    validate_connectome(W)
    
    if method == "absolute":
        W_thresh = W.copy()
        W_thresh[W_thresh < threshold] = 0
    elif method == "percentile":
        threshold_value = np.percentile(W[W > 0], 100 - threshold)
        W_thresh = W.copy()
        W_thresh[W_thresh < threshold_value] = 0
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    # Ensure symmetry
    W_thresh = (W_thresh + W_thresh.T) / 2
    
    logger.info(f"Thresholded connectome: {np.count_nonzero(W)} -> {np.count_nonzero(W_thresh)} edges")
    
    return W_thresh

