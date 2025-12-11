"""
Topological Data Analysis (TDA) features from residuals.

Computes persistent homology, persistence images, and Topological Misalignment Index (TMI).
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
from pathlib import Path

try:
    from giotto.homology import VietorisRipsPersistence
    from giotto.diagrams import PersistenceImage, BettiCurve
    from giotto.diagrams import pairwise_persistence_diagram_distances
    GIOTTO_AVAILABLE = True
except ImportError:
    try:
        from giotto_tda.homology import VietorisRipsPersistence
        from giotto_tda.diagrams import PersistenceImage, BettiCurve
        from giotto_tda.diagrams import pairwise_persistence_diagram_distances
        GIOTTO_AVAILABLE = True
    except ImportError:
        GIOTTO_AVAILABLE = False
        logging.warning("giotto-tda not available. TDA features will not work.")

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from .utils import validate_array, apply_baire_category_to_persistence_diagrams

logger = logging.getLogger(__name__)


def compute_persistence_diagrams(
    residuals: np.ndarray,
    W: Optional[np.ndarray] = None,
    mode: str = "node",
    max_dimension: int = 1,
    metric: str = "euclidean"
) -> List[np.ndarray]:
    """
    Compute persistence diagrams from residual patterns.
    
    Two modes:
    - 'node': Treat each ROI as a point with residual value as filtration height
    - 'edge': Use connectome edges with edge-based filtration
    
    Args:
        residuals: Residual patterns (n_subjects, n_rois) or (n_rois,)
        W: Optional connectome for edge-based filtration
        mode: 'node' or 'edge'
        max_dimension: Maximum homology dimension (0=H0, 1=H1, etc.)
        metric: Distance metric for point cloud ('euclidean', 'precomputed')
        
    Returns:
        List of persistence diagrams, one per subject.
        Each diagram is (n_features, 3) with columns [dim, birth, death]
    """
    if not GIOTTO_AVAILABLE and not GUDHI_AVAILABLE:
        raise ImportError(
            "Neither giotto-tda nor gudhi is available. "
            "Please install one: pip install giotto-tda or pip install gudhi"
        )
    
    residuals = np.atleast_2d(residuals)
    n_subjects, n_rois = residuals.shape
    
    diagrams = []
    
    if mode == "node":
        for i in range(n_subjects):
            r = residuals[i, :]
            
            if GIOTTO_AVAILABLE:
                point_cloud = r.reshape(-1, 1)
                vr = VietorisRipsPersistence(metric=metric, homology_dimensions=(0, max_dimension))
                diagram = vr.fit_transform([point_cloud])[0]
                
            elif GUDHI_AVAILABLE:
                point_cloud = r.reshape(-1, 1)
                rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=np.inf)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
                diagram_raw = simplex_tree.persistence()
                
                diagram = []
                for dim, (birth, death) in diagram_raw:
                    if death == np.inf:
                        death = birth + 1.0
                    diagram.append([dim, birth, death])
                diagram = np.array(diagram)
            
            diagrams.append(diagram)
            logger.debug(f"Computed persistence diagram for subject {i}: {len(diagram)} features")
    
    elif mode == "edge":
        if W is None:
            raise ValueError("Connectome W required for edge-based filtration")
        raise NotImplementedError("Edge-based filtration not yet implemented")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'node' or 'edge'")
    
    logger.info(f"Computed {len(diagrams)} persistence diagrams")
    
    return diagrams


def compute_persistence_images(
    diagrams: List[np.ndarray],
    resolution: Tuple[int, int] = (20, 20),
    bandwidth: float = 0.1,
    weight_function: Optional[callable] = None
) -> np.ndarray:
    """
    Convert persistence diagrams to persistence images.
    
    Args:
        diagrams: List of persistence diagrams
        resolution: (width, height) of persistence image
        bandwidth: Bandwidth for Gaussian kernel
        weight_function: Optional function to weight persistence points
        
    Returns:
        Persistence images (n_subjects, width * height)
    """
    if not GIOTTO_AVAILABLE:
        raise ImportError("giotto-tda required for persistence images")
    
    persistence_images = []
    
    for diagram in diagrams:
        # Filter to only H0 and H1 (dimensions 0 and 1)
        diagram_filtered = diagram[diagram[:, 0] <= 1]
        
        pi = PersistenceImage(
            bandwidth=bandwidth,
            weight=weight_function,
            resolution=resolution
        )
        
        diagram_for_giotto = []
        for dim in [0, 1]:
            dim_diagram = diagram_filtered[diagram_filtered[:, 0] == dim]
            if len(dim_diagram) > 0:
                for row in dim_diagram:
                    diagram_for_giotto.append([row[1], row[2]])
        
        if len(diagram_for_giotto) == 0:
            persistence_images.append(np.zeros(resolution[0] * resolution[1]))
        else:
            diagram_array = np.array(diagram_for_giotto)
            pi.fit([diagram_array])
            image = pi.transform([diagram_array])[0]
            persistence_images.append(image.flatten())
    
    return np.array(persistence_images)


def compute_tmi(
    diagram: np.ndarray,
    reference_diagram: np.ndarray,
    metric: str = "wasserstein",
    p: int = 2
) -> float:
    """
    Compute Topological Misalignment Index (TMI).
    
    TMI measures the distance between a subject's persistence diagram
    and a reference diagram (e.g., healthy control average).
    
    Args:
        diagram: Subject persistence diagram (n_features, 3)
        reference_diagram: Reference persistence diagram (n_features, 3)
        metric: Distance metric ('wasserstein' or 'bottleneck')
        p: p-norm for Wasserstein distance (default: 2)
        
    Returns:
        TMI value (non-negative float)
    """
    if not GIOTTO_AVAILABLE and not GUDHI_AVAILABLE:
        raise ImportError("giotto-tda or gudhi required for TMI computation")
    
    diagram_filtered = diagram[diagram[:, 0] <= 1]
    ref_filtered = reference_diagram[reference_diagram[:, 0] <= 1]
    
    if GIOTTO_AVAILABLE:
        diagrams_list = []
        ref_list = []
        
        for dim in [0, 1]:
            dim_diag = diagram_filtered[diagram_filtered[:, 0] == dim]
            dim_ref = ref_filtered[ref_filtered[:, 0] == dim]
            
            if len(dim_diag) > 0:
                diagrams_list.append(dim_diag[:, 1:3])
            else:
                diagrams_list.append(np.array([]).reshape(0, 2))
            
            if len(dim_ref) > 0:
                ref_list.append(dim_ref[:, 1:3])
            else:
                ref_list.append(np.array([]).reshape(0, 2))
        
        distances = []
        for dim_diag, dim_ref in zip(diagrams_list, ref_list):
            if len(dim_diag) == 0 and len(dim_ref) == 0:
                distances.append(0.0)
            elif len(dim_diag) == 0 or len(dim_ref) == 0:
                all_points = np.vstack([dim_diag, dim_ref]) if len(dim_diag) > 0 else dim_ref
                max_dist = np.max(np.linalg.norm(all_points, axis=1))
                distances.append(max_dist)
            else:
                if metric == "wasserstein":
                    from giotto.diagrams import PairwisePersistenceDiagramDistance
                    metric_obj = PairwisePersistenceDiagramDistance(metric="wasserstein", metric_params={"p": p})
                    dist_matrix = metric_obj.fit_transform([dim_diag, dim_ref])
                    distances.append(dist_matrix[0, 1])
                elif metric == "bottleneck":
                    from giotto.diagrams import PairwisePersistenceDiagramDistance
                    metric_obj = PairwisePersistenceDiagramDistance(metric="bottleneck")
                    dist_matrix = metric_obj.fit_transform([dim_diag, dim_ref])
                    distances.append(dist_matrix[0, 1])
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        tmi = np.sqrt(sum(d**2 for d in distances))
        
    elif GUDHI_AVAILABLE:
        tmi = 0.0
        for dim in [0, 1]:
            dim_diag = diagram_filtered[diagram_filtered[:, 0] == dim]
            dim_ref = ref_filtered[ref_filtered[:, 0] == dim]
            
            if len(dim_diag) == 0 and len(dim_ref) == 0:
                continue
            elif len(dim_diag) == 0 or len(dim_ref) == 0:
                all_points = np.vstack([dim_diag, dim_ref]) if len(dim_diag) > 0 else dim_ref
                max_dist = np.max(np.linalg.norm(all_points, axis=1))
                tmi += max_dist**2
            else:
                diag_points = dim_diag[:, 1:3]
                ref_points = dim_ref[:, 1:3]
                
                if metric == "wasserstein":
                    dist = gudhi.bottleneck_distance(diag_points, ref_points)
                    tmi += dist**2
                elif metric == "bottleneck":
                    dist = gudhi.bottleneck_distance(diag_points, ref_points)
                    tmi += dist**2
                else:
                    raise ValueError(f"Unknown metric: {metric}")
        
        tmi = np.sqrt(tmi)
    
    return float(tmi)


def compute_tmi_batch(
    diagrams: List[np.ndarray],
    reference_diagram: np.ndarray,
    metric: str = "wasserstein",
    p: int = 2
) -> np.ndarray:
    """
    Compute TMI for multiple subjects.
    
    Args:
        diagrams: List of persistence diagrams (n_subjects)
        reference_diagram: Reference persistence diagram
        metric: Distance metric
        p: p-norm for Wasserstein
        
    Returns:
        TMI values (n_subjects,)
    """
    tmi_values = []
    for diagram in diagrams:
        tmi = compute_tmi(diagram, reference_diagram, metric=metric, p=p)
        tmi_values.append(tmi)
    
    return np.array(tmi_values)


def get_reference_diagram(
    diagrams: List[np.ndarray],
    method: str = "mean"
) -> np.ndarray:
    """
    Compute reference diagram from a set of diagrams.
    
    Args:
        diagrams: List of persistence diagrams
        method: 'mean' (average birth/death) or 'median'
        
    Returns:
        Reference persistence diagram
    """
    if method == "mean":
        all_diagrams_by_dim = {0: [], 1: []}
        
        for diagram in diagrams:
            for dim in [0, 1]:
                dim_points = diagram[diagram[:, 0] == dim]
                if len(dim_points) > 0:
                    all_diagrams_by_dim[dim].append(dim_points[:, 1:3])
        
        reference_points = []
        for dim in [0, 1]:
            if len(all_diagrams_by_dim[dim]) > 0:
                all_points = np.vstack(all_diagrams_by_dim[dim])
                mean_point = np.mean(all_points, axis=0)
                reference_points.append([dim, mean_point[0], mean_point[1]])
        
        if len(reference_points) == 0:
            return np.array([]).reshape(0, 3)
        
        return np.array(reference_points)
    
    elif method == "median":
        all_diagrams_by_dim = {0: [], 1: []}
        
        for diagram in diagrams:
            for dim in [0, 1]:
                dim_points = diagram[diagram[:, 0] == dim]
                if len(dim_points) > 0:
                    all_diagrams_by_dim[dim].append(dim_points[:, 1:3])
        
        reference_points = []
        for dim in [0, 1]:
            if len(all_diagrams_by_dim[dim]) > 0:
                all_points = np.vstack(all_diagrams_by_dim[dim])
                median_point = np.median(all_points, axis=0)
                reference_points.append([dim, median_point[0], median_point[1]])
        
        if len(reference_points) == 0:
            return np.array([]).reshape(0, 3)
        
        return np.array(reference_points)
    
    else:
        raise ValueError(f"Unknown method: {method}")

