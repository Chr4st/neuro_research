"""
Central configuration management for the ADNI NDM-TDA pipeline.

Supports loading from YAML files, environment variables, and defaults.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Central configuration class for the pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_path: Path to YAML config file. If None, uses defaults.
        """
        self.config = self._load_defaults()
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
                self._update_config(self.config, user_config)
        
        # Override with environment variables
        self._load_from_env()
        
        # Convert string paths to Path objects
        self._normalize_paths()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'data': {
                'raw_data_dir': 'data/raw',
                'processed_data_dir': 'data/processed',
                'connectome_path': 'data/processed/connectome.npy',
                'cortical_thickness_path': 'data/processed/cortical_thickness.csv',
                'roi_labels_path': 'data/processed/roi_labels.txt',
            },
            'ndm': {
                'beta': 0.1,  # diffusion rate
                'time_steps': [0.1, 0.5, 1.0, 2.0, 5.0],
                'normalize_laplacian': True,
            },
            'tda': {
                'filtration_mode': 'node',  # 'node' or 'edge'
                'max_dimension': 1,  # H0 and H1
                'persistence_image_resolution': (20, 20),
                'persistence_image_bandwidth': 0.1,
                'tmi_metric': 'wasserstein',  # 'wasserstein' or 'bottleneck'
                'tmi_p': 2,  # Wasserstein p-norm
            },
            'clustering': {
                'n_components': 10,  # PCA/UMAP components
                'umap_n_neighbors': 15,
                'umap_min_dist': 0.1,
                'hdbscan_min_cluster_size': 5,
                'hdbscan_min_samples': 3,
            },
            'output': {
                'results_dir': 'results',
                'figures_dir': 'results/figures',
                'tables_dir': 'results/tables',
                'logs_dir': 'results/logs',
            },
            'random_seed': 42,
            'n_jobs': -1,  # -1 for all cores
        }
    
    def _update_config(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update base config with update dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'ADNI_DATA_DIR': ('data', 'raw_data_dir'),
            'RESULTS_DIR': ('output', 'results_dir'),
            'RANDOM_SEED': ('random_seed',),
            'N_JOBS': ('n_jobs',),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if len(config_path) == 1:
                    self.config[config_path[0]] = self._convert_type(value)
                else:
                    if config_path[0] not in self.config:
                        self.config[config_path[0]] = {}
                    self.config[config_path[0]][config_path[1]] = self._convert_type(value)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string to appropriate type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _normalize_paths(self) -> None:
        """Convert string paths to Path objects."""
        for key, value in self.config.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if 'path' in subkey.lower() or 'dir' in subkey.lower():
                        if isinstance(subvalue, str):
                            self.config[key][subkey] = Path(subvalue)
            elif 'path' in key.lower() or 'dir' in key.lower():
                if isinstance(value, str):
                    self.config[key] = Path(value)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation (e.g., 'ndm.beta').
        
        Args:
            key_path: Dot-separated path to config value
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.config


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    return Config(config_path)

