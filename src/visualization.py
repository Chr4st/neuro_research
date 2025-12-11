"""
Visualization utilities for the pipeline.

Provides plotting functions for connectomes, PH diagrams, embeddings, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from .utils import validate_array

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_connectome(
    W: np.ndarray,
    node_positions: Optional[np.ndarray] = None,
    node_colors: Optional[np.ndarray] = None,
    threshold: float = 0.0,
    save_path: Optional[Path] = None,
    title: str = "Structural Connectome"
) -> None:
    """
    Plot structural connectome as network graph.
    
    Args:
        W: Connectome matrix (n_rois x n_rois)
        node_positions: Optional 2D positions for nodes (n_rois, 2)
        node_colors: Optional colors for nodes (n_rois,)
        threshold: Threshold for edge display
        save_path: Optional path to save figure
        title: Plot title
    """
    if not NETWORKX_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(W, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('ROI')
        ax.set_ylabel('ROI')
        plt.colorbar(im, ax=ax, label='Connection Strength')
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        return
    
    G = nx.from_numpy_array(W)
    
    if threshold > 0:
        edges_to_remove = [(u, v) for u, v, w in G.edges(data='weight') if w < threshold]
        G.remove_edges_from(edges_to_remove)
    
    if node_positions is None:
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        pos = {i: node_positions[i] for i in range(W.shape[0])}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*2 for w in weights], ax=ax)
    
    if node_colors is not None:
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='viridis', ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=100, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved connectome plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_persistence_diagram(
    diagram: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Persistence Diagram"
) -> None:
    """
    Plot persistence diagram.
    
    Args:
        diagram: Persistence diagram (n_features, 3) with [dim, birth, death]
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for dim in [0, 1]:
        dim_diagram = diagram[diagram[:, 0] == dim]
        if len(dim_diagram) > 0:
            births = dim_diagram[:, 1]
            deaths = dim_diagram[:, 2]
            
            label = f'H{dim}'
            ax.scatter(births, deaths, label=label, alpha=0.6, s=50)
    
    max_val = np.max(diagram[:, 1:3]) if len(diagram) > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Diagonal')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved persistence diagram to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "Embedding"
) -> None:
    """
    Plot 2D or 3D embedding (UMAP/PCA).
    
    Args:
        embedding: Embedding coordinates (n_subjects, 2 or 3)
        labels: Optional cluster labels for coloring
        colors: Optional continuous values for coloring
        save_path: Optional path to save figure
        title: Plot title
    """
    validate_array(embedding, expected_shape=None, name="embedding")
    
    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got shape {embedding.shape}")
    
    n_dims = embedding.shape[1]
    
    if n_dims not in [2, 3]:
        embedding = embedding[:, :2]
        n_dims = 2
        logger.warning("Embedding has >3 dimensions, using first 2")
    
    if n_dims == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             c='gray', alpha=0.3, s=20, label='Noise')
                else:
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                             label=f'Cluster {label}', alpha=0.6, s=50)
        elif colors is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=colors, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Value')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=50)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title(title)
        if labels is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    else:  # 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                             c='gray', alpha=0.3, s=20, label='Noise')
                else:
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                             label=f'Cluster {label}', alpha=0.6, s=50)
        elif colors is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=colors, cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Value')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6, s=50)
        
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(title)
        if labels is not None:
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved embedding plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_tmi_distribution(
    tmi_values: np.ndarray,
    diagnoses: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = "TMI Distribution"
) -> None:
    """
    Plot distribution of TMI values.
    
    Args:
        tmi_values: TMI values (n_subjects,)
        diagnoses: Optional diagnosis labels for grouping
        save_path: Optional path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if diagnoses is not None:
        unique_diag = np.unique(diagnoses)
        for diag in unique_diag:
            mask = diagnoses == diag
            ax.hist(tmi_values[mask], alpha=0.6, label=diag, bins=30)
        ax.legend()
    else:
        ax.hist(tmi_values, bins=30, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Topological Misalignment Index (TMI)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved TMI distribution plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cluster_comparison(
    data: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[np.ndarray] = None,
    top_n: int = 10,
    save_path: Optional[Path] = None,
    title: str = "Cluster Comparison"
) -> None:
    """
    Plot comparison of top features across clusters.
    
    Args:
        data: Feature matrix (n_subjects, n_features)
        labels: Cluster labels (n_subjects,)
        feature_names: Optional feature names
        top_n: Number of top features to show
        save_path: Optional path to save figure
        title: Plot title
    """
    unique_labels = np.unique(labels[labels != -1])
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
    
    cluster_means = np.array([np.mean(data[labels == label], axis=0) for label in unique_labels])
    feature_variance = np.var(cluster_means, axis=0)
    top_indices = np.argsort(feature_variance)[-top_n:][::-1]
    
    plot_data = []
    for label in unique_labels:
        for idx in top_indices:
            values = data[labels == label, idx]
            for val in values:
                plot_data.append({
                    'Cluster': f'Cluster {label}',
                    'Feature': feature_names[idx],
                    'Value': val
                })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=plot_df, x='Feature', y='Value', hue='Cluster', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cluster comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()

