"""
Clustering and dimensionality reduction for subtype discovery.

Uses UMAP for embedding and HDBSCAN for clustering.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("HDBSCAN not available. Install with: pip install hdbscan")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from .utils import validate_array

logger = logging.getLogger(__name__)


def reduce_dimensions(
    features: np.ndarray,
    n_components: int = 10,
    method: str = "umap",
    **kwargs
) -> Tuple[np.ndarray, object]:
    """
    Reduce dimensionality of feature matrix.
    
    Args:
        features: Feature matrix (n_subjects, n_features)
        n_components: Number of components to keep
        method: 'umap', 'pca', or 'none'
        **kwargs: Additional arguments for dimensionality reduction
        
    Returns:
        Tuple of (reduced_features, transformer_object)
    """
    validate_array(features, expected_shape=None, name="features")
    
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D, got shape {features.shape}")
    
    n_subjects, n_features = features.shape
    
    if n_components >= n_features:
        logger.warning(f"n_components ({n_components}) >= n_features ({n_features}), skipping reduction")
        return features, None
    
    if method == "umap":
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to PCA")
            method = "pca"
        else:
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=kwargs.get('random_state', 42)
            )
            reduced = reducer.fit_transform(features)
            logger.info(f"UMAP reduction: {features.shape} -> {reduced.shape}")
            return reduced, reducer
    
    if method == "pca":
        # Standardize first
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        reducer = PCA(n_components=n_components, random_state=kwargs.get('random_state', 42))
        reduced = reducer.fit_transform(features_scaled)
        
        explained_var = np.sum(reducer.explained_variance_ratio_)
        logger.info(
            f"PCA reduction: {features.shape} -> {reduced.shape}, "
            f"explained variance: {explained_var:.2%}"
        )
        return reduced, {'pca': reducer, 'scaler': scaler}
    
    if method == "none":
        return features, None
    
    raise ValueError(f"Unknown reduction method: {method}")


def cluster_subjects(
    features: np.ndarray,
    method: str = "hdbscan",
    **kwargs
) -> Tuple[np.ndarray, object]:
    """
    Cluster subjects based on feature matrix.
    
    Args:
        features: Feature matrix (n_subjects, n_features) or reduced features
        method: Clustering method ('hdbscan', 'gmm', 'kmeans')
        **kwargs: Additional arguments for clustering
        
    Returns:
        Tuple of (cluster_labels, clusterer_object)
        - cluster_labels: (n_subjects,) array, -1 indicates noise/outliers
    """
    validate_array(features, expected_shape=None, name="features")
    
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D, got shape {features.shape}")
    
    if method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, falling back to GMM")
            method = "gmm"
        else:
            min_cluster_size = kwargs.get('min_cluster_size', 5)
            min_samples = kwargs.get('min_samples', 3)
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(features)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            logger.info(
                f"HDBSCAN clustering: {n_clusters} clusters, {n_noise} noise points"
            )
            return labels, clusterer
    
    if method == "gmm":
        n_components = kwargs.get('n_components', 3)
        random_state = kwargs.get('random_state', 42)
        
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type='full'
        )
        labels = gmm.fit_predict(features)
        
        logger.info(f"GMM clustering: {n_components} components")
        return labels, gmm
    
    if method == "kmeans":
        from sklearn.cluster import KMeans
        n_clusters = kwargs.get('n_clusters', 3)
        random_state = kwargs.get('random_state', 42)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(features)
        
        logger.info(f"K-means clustering: {n_clusters} clusters")
        return labels, kmeans
    
    raise ValueError(f"Unknown clustering method: {method}")


def discover_subtypes(
    features: np.ndarray,
    n_components: int = 10,
    reduction_method: str = "umap",
    clustering_method: str = "hdbscan",
    reduction_kwargs: Optional[Dict] = None,
    clustering_kwargs: Optional[Dict] = None,
    random_state: int = 42
) -> Dict[str, any]:
    """
    Complete pipeline: dimensionality reduction + clustering.
    
    Args:
        features: Feature matrix (n_subjects, n_features)
        n_components: Number of components for reduction
        reduction_method: 'umap', 'pca', or 'none'
        clustering_method: 'hdbscan', 'gmm', or 'kmeans'
        reduction_kwargs: Additional kwargs for reduction
        clustering_kwargs: Additional kwargs for clustering
        random_state: Random seed
        
    Returns:
        Dictionary with:
        - 'labels': Cluster labels (n_subjects,)
        - 'embedding': Reduced features (n_subjects, n_components)
        - 'reducer': Reduction transformer
        - 'clusterer': Clustering object
    """
    reduction_kwargs = reduction_kwargs or {}
    clustering_kwargs = clustering_kwargs or {}
    
    reduction_kwargs['random_state'] = random_state
    clustering_kwargs['random_state'] = random_state
    
    # Reduce dimensions
    embedding, reducer = reduce_dimensions(
        features,
        n_components=n_components,
        method=reduction_method,
        **reduction_kwargs
    )
    
    # Cluster
    labels, clusterer = cluster_subjects(
        embedding,
        method=clustering_method,
        **clustering_kwargs
    )
    
    return {
        'labels': labels,
        'embedding': embedding,
        'reducer': reducer,
        'clusterer': clusterer
    }

