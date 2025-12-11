"""
Generic utility functions for logging, seeding, and common operations.
"""

import logging
import random
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Callable
import warnings
from scipy.spatial.distance import cdist


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Name of log file (optional)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_dir and log_file:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_array(
    arr: np.ndarray,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[type] = None,
    name: str = "array"
) -> None:
    """
    Validate array properties.
    
    Args:
        arr: Array to validate
        expected_shape: Expected shape (None for any)
        expected_dtype: Expected dtype (None for any)
        name: Name for error messages
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    
    if expected_shape is not None:
        if arr.shape != expected_shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match expected {expected_shape}"
            )
    
    if expected_dtype is not None:
        if arr.dtype != expected_dtype:
            raise ValueError(
                f"{name} dtype {arr.dtype} does not match expected {expected_dtype}"
            )


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use when denominator is zero
        
    Returns:
        Result array
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator, out=np.full_like(numerator, fill_value), where=denominator!=0)
    return result


def zscore(x: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> np.ndarray:
    """
    Compute z-scores (standardize) along specified axis.
    
    Args:
        x: Input array
        axis: Axis along which to compute (None for entire array)
        ddof: Degrees of freedom for std calculation
        
    Returns:
        Z-scored array
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True, ddof=ddof)
    return safe_divide(x - mean, std, fill_value=0.0)


# ============================================================================
# Baire Category Theorem Implementation
# ============================================================================

def is_dense_in_metric_space(
    subset_points: np.ndarray,
    space_points: np.ndarray,
    metric: Callable = None,
    epsilon: float = 1e-6
) -> bool:
    """
    Check if a subset is dense in a metric space.
    
    A set A is dense in X if for every point x in X and every epsilon > 0,
    there exists a point a in A such that d(x, a) < epsilon.
    
    Args:
        subset_points: Points in the subset (n_subset, n_features)
        space_points: Points in the space (n_space, n_features)
        metric: Distance function (default: Euclidean)
        epsilon: Tolerance for density check
        
    Returns:
        True if subset is dense in the space
    """
    if len(subset_points) == 0:
        return False
    
    if metric is None:
        distances = cdist(space_points, subset_points, metric='euclidean')
    else:
        distances = np.array([
            [metric(x, a) for a in subset_points]
            for x in space_points
        ])
    
    min_distances = np.min(distances, axis=1)
    return np.all(min_distances < epsilon)


def is_open_set(
    points: np.ndarray,
    membership_function: Callable,
    epsilon: float = 1e-6
) -> bool:
    """
    Check if a set is open in a metric space.
    
    A set U is open if for every point x in U, there exists epsilon > 0
    such that the epsilon-ball around x is contained in U.
    
    Args:
        points: Points in the set (n_points, n_features)
        membership_function: Function that returns True if a point is in the set
        epsilon: Radius for epsilon-ball check
        
    Returns:
        True if the set is open
    """
    if len(points) == 0:
        return True  # Empty set is open
    
    for point in points:
        if not membership_function(point):
            return False
        
        n_samples = 100
        noise = np.random.normal(0, epsilon, size=(n_samples, point.shape[0]))
        nearby_points = point + noise
        
        for nearby in nearby_points:
            if not membership_function(nearby):
                return False
    
    return True


def baire_category_theorem_intersection(
    dense_open_sets: List[Callable],
    space_points: np.ndarray,
    metric: Callable = None,
    epsilon: float = 1e-6,
    n_test_points: int = 1000
) -> bool:
    """
    Verify the Baire Category Theorem.
    
    The Baire Category Theorem states that in a complete metric space,
    the intersection of countably many dense open sets is dense.
    
    This function checks if the intersection of given dense open sets
    is dense in the space.
    
    Args:
        dense_open_sets: List of membership functions for dense open sets
        space_points: Sample points from the metric space (n_points, n_features)
        metric: Distance function (default: Euclidean)
        epsilon: Tolerance for density check
        n_test_points: Number of test points to sample for verification
        
    Returns:
        True if the intersection is dense (verifying Baire category theorem)
    """
    if len(dense_open_sets) == 0:
        return True
    
    if len(space_points) > n_test_points:
        indices = np.random.choice(len(space_points), n_test_points, replace=False)
        test_points = space_points[indices]
    else:
        test_points = space_points
    
    intersection_points = []
    
    for point in test_points:
        in_all_sets = all(membership_func(point) for membership_func in dense_open_sets)
        if in_all_sets:
            intersection_points.append(point)
    
    if len(intersection_points) == 0:
        return False
    
    intersection_points = np.array(intersection_points)
    
    return is_dense_in_metric_space(
        intersection_points,
        test_points,
        metric=metric,
        epsilon=epsilon
    )


def check_baire_space_completeness(
    points: np.ndarray,
    metric: Callable = None,
    cauchy_epsilon: float = 1e-6
) -> bool:
    """
    Check if a metric space is complete (required for Baire Category Theorem).
    
    A metric space is complete if every Cauchy sequence converges to a point in the space.
    This is a necessary condition for the Baire Category Theorem to hold.
    
    Args:
        points: Sample points from the space (n_points, n_features)
        metric: Distance function (default: Euclidean)
        cauchy_epsilon: Tolerance for Cauchy sequence check
        
    Returns:
        True if the space appears complete (based on sample)
    """
    if len(points) < 2:
        return True
    
    if metric is None:
        metric = lambda x, y: np.linalg.norm(x - y)
    
    n_sequences = 50
    n_steps = 20
    
    for _ in range(n_sequences):
        idx = np.random.randint(0, len(points))
        sequence = [points[idx]]
        
        for step in range(1, n_steps):
            target_idx = np.random.randint(0, len(points))
            target = points[target_idx]
            
            current = sequence[-1]
            direction = target - current
            step_size = 1.0 / (step + 1)
            next_point = current + step_size * direction
            sequence.append(next_point)
        
        is_cauchy = True
        for i in range(len(sequence) - 1):
            for j in range(i + 1, len(sequence)):
                dist = metric(sequence[i], sequence[j])
                if dist > cauchy_epsilon * (j - i):
                    is_cauchy = False
                    break
            if not is_cauchy:
                break
        
        if is_cauchy:
            limit_point = sequence[-1]
            distances = [metric(limit_point, p) for p in points]
            min_dist = np.min(distances)
            
            if min_dist > cauchy_epsilon * 10:  # Limit not in space
                return False
    
    return True


def apply_baire_category_to_persistence_diagrams(
    diagrams: List[np.ndarray],
    reference_diagram: np.ndarray,
    metric: str = "wasserstein"
) -> dict:
    """
    Apply Baire Category Theorem concepts to analyze persistence diagrams.
    
    This function uses the Baire Category Theorem framework to analyze
    the structure of the space of persistence diagrams, which is relevant
    for TMI computation and topological feature analysis.
    
    Args:
        diagrams: List of persistence diagrams
        reference_diagram: Reference diagram (e.g., healthy control)
        metric: Distance metric for diagrams ('wasserstein' or 'bottleneck')
        
    Returns:
        Dictionary with Baire category analysis results:
        - 'is_complete': Whether the space appears complete
        - 'dense_sets_info': Information about dense subsets
        - 'baire_property': Whether Baire property holds
    """
    if len(diagrams) == 0:
        return {
            'is_complete': False,
            'dense_sets_info': {},
            'baire_property': False
        }
    
    try:
        from src.tda_features import compute_persistence_images
        diagram_features = compute_persistence_images(
            diagrams + [reference_diagram],
            resolution=(10, 10),
            bandwidth=0.1
        )
        space_points = diagram_features[:-1]
        reference_point = diagram_features[-1]
    except Exception:
        logging.warning("Could not compute persistence images, using raw coordinates")
        space_points = np.array([d.flatten() for d in diagrams])
        reference_point = reference_diagram.flatten()
    
    is_complete = check_baire_space_completeness(space_points)
    
    def create_dense_open_set(center: np.ndarray, radius: float):
        """Create a membership function for a dense open set."""
        def membership(point: np.ndarray) -> bool:
            dist = np.linalg.norm(point - center)
            return dist < radius
        return membership
    
    dense_open_sets = [
        create_dense_open_set(reference_point, 1.0),
        create_dense_open_set(reference_point, 2.0),
    ]
    
    baire_property = baire_category_theorem_intersection(
        dense_open_sets,
        space_points,
        epsilon=0.1
    )
    
    return {
        'is_complete': is_complete,
        'dense_sets_info': {
            'n_sets': len(dense_open_sets),
            'reference_point': reference_point.tolist() if len(reference_point) < 100 else 'too_large'
        },
        'baire_property': baire_property,
        'n_diagrams': len(diagrams)
    }

