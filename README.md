# Topological Misalignment of Network-Diffusion Residuals Reveals Alzheimer's Disease Subtypes and Progression Pathways

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains a complete computational pipeline for discovering Alzheimer's disease (AD) subtypes using **Network Diffusion Models (NDM)** and **Topological Data Analysis (TDA)**. The method identifies distinct AD progression pathways by analyzing the topological structure of residuals between observed cortical atrophy patterns and those predicted by network diffusion on structural connectomes.

### What This Project Does

1. **Network Diffusion Modeling**: Uses structural brain connectomes to predict how pathology spreads through the brain network
2. **Residual Analysis**: Computes differences between observed and predicted atrophy patterns
3. **Topological Features**: Extracts persistent homology features from residual fields to capture their topological structure
4. **Topological Misalignment Index (TMI)**: Quantifies how much a subject's residual topology deviates from a reference (e.g., healthy controls)
5. **Subtype Discovery**: Clusters subjects based on TDA features + TMI to identify distinct AD subtypes

### Key Concepts (Simple Terms)

- **Network Diffusion Model (NDM)**: A mathematical model that predicts how disease spreads through brain connections, like heat diffusing through a network
- **Persistent Homology (PH)**: A topological method that captures the "shape" of data, identifying holes, clusters, and other topological features that persist across different scales
- **Topological Misalignment Index (TMI)**: A measure of how "topologically different" a subject's brain pattern is compared to a reference, based on the shape/structure of their residual field

### Why Use This Repository?

- **Reproducible Research**: Complete, production-quality code that can be run end-to-end
- **Modular Design**: Well-organized modules that can be adapted for other datasets or research questions
- **Comprehensive Pipeline**: From raw data to final figures and statistics
- **Well-Documented**: Extensive docstrings, type hints, and examples

## Pipeline Overview

```
ADNI Data
    ↓
[01] Data Preprocessing (z-scoring, missing values)
    ↓
[02] Structural Connectome (load/build SC, compute Laplacian)
    ↓
[03] Network Diffusion Model (predict pathology spread)
    ↓
[04] Residual Computation (r = x_obs - x_diff)
    ↓
[05] TDA Features (persistent homology, persistence images, TMI)
    ↓
[06] Clustering (UMAP + HDBSCAN for subtype discovery)
    ↓
[07] Statistics & Visualization (cluster analysis, clinical associations)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Option 1: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd researhcprojec

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional, for imports)
pip install -e .
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n ndm-tda python=3.10
conda activate ndm-tda

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import numpy, scipy, pandas, sklearn; print('Core packages OK')"
python -c "import giotto_tda; print('TDA packages OK')"
```

## Usage

### Quick Start (Mock Data)

To test the pipeline with synthetic data:

```bash
# Step 1: Prepare data
python scripts/01_prepare_data.py --use-mock

# Step 2: Build connectome
python scripts/02_build_connectome.py --use-mock

# Step 3: Run NDM
python scripts/03_run_ndm.py

# Step 4: Compute residuals
python scripts/04_compute_residuals.py --time-point 1.0

# Step 5: Compute TDA features
python scripts/05_compute_tda_features.py --time-point 1.0

# Step 6: Run clustering
python scripts/06_run_clustering.py --time-point 1.0

# Step 7: Statistics and plots
python scripts/07_run_stats_and_plots.py --time-point 1.0
```

### Using Real ADNI Data

1. **Obtain ADNI Data**: Get access from [adni.loni.usc.edu](https://adni.loni.usc.edu/)

2. **Prepare Data Files**:
   - Cortical thickness CSV: `SubjectID,ROI_1,ROI_2,...,ROI_N,Age,Sex,Diagnosis,...`
   - Structural connectome: NumPy array (`.npy`) or CSV, shape `(n_rois, n_rois)`

3. **Configure Paths**: Edit `configs/default.yaml` or create your own config:

```yaml
data:
  cortical_thickness_path: "path/to/your/thickness.csv"
  connectome_path: "path/to/your/connectome.npy"
```

4. **Run Pipeline**:

```bash
# Use your config file
python scripts/01_prepare_data.py --config configs/default.yaml
python scripts/02_build_connectome.py --config configs/default.yaml
# ... continue with remaining scripts
```

### Configuration

The pipeline uses YAML configuration files (see `configs/default.yaml`). Key parameters:

- **NDM**: `beta` (diffusion rate), `time_steps`, `normalize_laplacian`
- **TDA**: `filtration_mode`, `max_dimension`, `persistence_image_resolution`
- **Clustering**: `n_components`, `reduction_method`, `clustering_method`

You can also set environment variables:
```bash
export ADNI_DATA_DIR="/path/to/data"
export RANDOM_SEED=42
```

### Jupyter Notebooks

Interactive analysis notebooks are available in `notebooks/`:

- `EDA_ADNI.ipynb`: Exploratory data analysis
- `Visualize_PH_and_TMI.ipynb`: Visualization of persistence diagrams and TMI

## Output Structure

```
results/
├── figures/          # All plots and visualizations
├── tables/           # Statistical tables (CSV)
├── logs/             # Log files
├── *.npy             # Intermediate results (residuals, TMI, etc.)
└── *.csv             # Cluster assignments, statistics
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Topological Misalignment of Network-Diffusion Residuals Reveals Alzheimer's Disease Subtypes and Progression Pathways},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  note={arXiv preprint arXiv:XXXX.XXXXX}
}
```

## Reproducibility

- **Random Seeds**: All scripts use fixed random seeds (default: 42) for reproducibility
- **Version Control**: Dependencies are pinned in `requirements.txt`
- **Mock Data**: Scripts can generate synthetic data for testing without real ADNI access
- **Configuration**: All parameters are configurable via YAML files

**Note**: Results on mock data will differ from real ADNI data. The mock data is for code testing only.

## Project Structure

```
.
├── src/                    # Source code modules
│   ├── config.py          # Configuration management
│   ├── data_loading.py     # Data I/O
│   ├── preprocessing.py    # Data preprocessing
│   ├── connectome.py      # Connectome handling
│   ├── ndm.py             # Network Diffusion Model
│   ├── residuals.py       # Residual computation
│   ├── tda_features.py    # Persistent homology, TMI
│   ├── clustering.py      # Subtype discovery
│   ├── analysis.py        # Statistical analysis
│   ├── visualization.py   # Plotting utilities
│   └── utils.py           # Helper functions
├── scripts/                # Pipeline scripts (01-07)
├── notebooks/              # Jupyter notebooks
├── configs/                # Configuration files
├── data/                   # Data directory (not in repo)
├── results/                # Output directory
├── paper/                  # LaTeX paper skeleton
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ADNI data contributors
- Developers of giotto-tda, scikit-learn, and other open-source tools
- The topological data analysis and network neuroscience communities

## References

1. Raj, A., Kuceyeski, A., & Weiner, M. (2012). A network diffusion model of disease progression in dementia. *Neuron*, 73(6), 1204-1215.

2. Edelsbrunner, H., & Harer, J. (2010). *Computational topology: an introduction*. American Mathematical Society.

3. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.

