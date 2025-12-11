"""
Network Diffusion Model (NDM) implementation.

Implements the heat equation on graphs: dx/dt = -β L x
Uses matrix exponential for solution: x(t) = exp(-β L t) x(0)
"""

import numpy as np
from typing import Union, Optional, List
from scipy.linalg import expm
from scipy import sparse
import logging

from .connectome import compute_laplacian, validate_connectome
from .utils import validate_array

logger = logging.getLogger(__name__)


def run_ndm_from_initial_seed(
    W: np.ndarray,
    x0: np.ndarray,
    beta: float,
    t: Union[float, np.ndarray],
    normalized_laplacian: bool = True
) -> np.ndarray:
    """
    Run Network Diffusion Model from initial seed pattern.
    
    Solves: x(t) = exp(-β L t) x(0)
    
    Args:
        W: Structural connectome (n_rois x n_rois)
        x0: Initial seed pattern (n_rois,)
        beta: Diffusion rate parameter
        t: Time point(s) - scalar or array
        normalized_laplacian: Whether to use normalized Laplacian
        
    Returns:
        Predicted pattern at time t:
        - If t is scalar: (n_rois,)
        - If t is array: (n_times, n_rois)
    """
    validate_connectome(W)
    validate_array(x0, expected_shape=(W.shape[0],), name="initial_seed")
    
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    
    L = compute_laplacian(W, normalized=normalized_laplacian, sparse_format=False)
    
    t_array = np.atleast_1d(t)
    n_rois = W.shape[0]
    n_times = len(t_array)
    
    # Compute matrix exponential for each time point
    # For efficiency, we can compute exp(-β L t) for all t at once if needed
    # But for large matrices, it's better to compute separately
    
    if n_times == 1:
        # Single time point
        exp_Lt = expm(-beta * L * t_array[0])
        x_pred = exp_Lt @ x0
        return x_pred
    else:
        # Multiple time points
        x_pred = np.zeros((n_times, n_rois))
        for i, t_val in enumerate(t_array):
            exp_Lt = expm(-beta * L * t_val)
            x_pred[i] = exp_Lt @ x0
        return x_pred


def smooth_observed_pattern(
    W: np.ndarray,
    x_obs: np.ndarray,
    beta: float,
    t: float,
    normalized_laplacian: bool = True
) -> np.ndarray:
    """
    Smooth observed pattern using NDM.
    
    This applies diffusion to the observed pattern, effectively
    predicting what the pattern would look like after diffusion.
    
    Args:
        W: Structural connectome (n_rois x n_rois)
        x_obs: Observed pattern (n_rois,)
        beta: Diffusion rate parameter
        t: Diffusion time
        normalized_laplacian: Whether to use normalized Laplacian
        
    Returns:
        Smoothed/diffused pattern (n_rois,)
    """
    return run_ndm_from_initial_seed(W, x_obs, beta, t, normalized_laplacian)


def run_ndm_batch(
    W: np.ndarray,
    X0: np.ndarray,
    beta: float,
    t: Union[float, np.ndarray],
    normalized_laplacian: bool = True
) -> np.ndarray:
    """
    Run NDM for multiple subjects in batch.
    
    Args:
        W: Structural connectome (n_rois x n_rois)
        X0: Initial patterns (n_subjects, n_rois)
        beta: Diffusion rate parameter
        t: Time point(s) - scalar or array
        normalized_laplacian: Whether to use normalized Laplacian
        
    Returns:
        Predicted patterns:
        - If t is scalar: (n_subjects, n_rois)
        - If t is array: (n_subjects, n_times, n_rois)
    """
    validate_connectome(W)
    validate_array(X0, expected_shape=None, name="initial_patterns")
    
    if X0.shape[-1] != W.shape[0]:
        raise ValueError(
            f"ROI dimension mismatch: X0 has {X0.shape[-1]} ROIs, "
            f"connectome has {W.shape[0]} ROIs"
        )
    
    n_subjects = X0.shape[0]
    t_array = np.atleast_1d(t)
    n_times = len(t_array)
    
    L = compute_laplacian(W, normalized=normalized_laplacian, sparse_format=False)
    
    # Pre-compute matrix exponentials for all time points
    exp_Lt_list = [expm(-beta * L * t_val) for t_val in t_array]
    
    if n_times == 1:
        # Single time point
        exp_Lt = exp_Lt_list[0]
        X_pred = X0 @ exp_Lt.T  # (n_subjects, n_rois) @ (n_rois, n_rois)
        return X_pred
    else:
        # Multiple time points
        X_pred = np.zeros((n_subjects, n_times, W.shape[0]))
        for i, exp_Lt in enumerate(exp_Lt_list):
            X_pred[:, i, :] = X0 @ exp_Lt.T
        return X_pred


def find_optimal_beta(
    W: np.ndarray,
    x_obs: np.ndarray,
    x_seed: Optional[np.ndarray] = None,
    beta_range: Optional[np.ndarray] = None,
    t: float = 1.0,
    metric: str = "mse"
) -> float:
    """
    Find optimal beta parameter by minimizing prediction error.
    
    Args:
        W: Structural connectome
        x_obs: Observed pattern
        x_seed: Initial seed (if None, uses x_obs)
        beta_range: Range of beta values to test
        t: Fixed time point
        metric: Error metric ('mse', 'mae', 'correlation')
        
    Returns:
        Optimal beta value
    """
    if x_seed is None:
        x_seed = x_obs
    
    if beta_range is None:
        beta_range = np.logspace(-3, 1, 20)  # 0.001 to 10
    
    errors = []
    for beta in beta_range:
        x_pred = run_ndm_from_initial_seed(W, x_seed, beta, t)
        
        if metric == "mse":
            error = np.mean((x_obs - x_pred) ** 2)
        elif metric == "mae":
            error = np.mean(np.abs(x_obs - x_pred))
        elif metric == "correlation":
            error = -np.corrcoef(x_obs, x_pred)[0, 1]  # Negative for minimization
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        errors.append(error)
    
    optimal_idx = np.argmin(errors)
    optimal_beta = beta_range[optimal_idx]
    
    logger.info(f"Optimal beta: {optimal_beta:.4f} (error: {errors[optimal_idx]:.4f})")
    
    return optimal_beta

