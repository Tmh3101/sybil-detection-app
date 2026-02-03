# Lens Protocol Sybil Detection App

A comprehensive Graph Neural Network (GNN)-based web application for detecting Sybil accounts on the Lens Protocol decentralized social network.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This application detects **Sybil accounts** (fake or automated accounts) on the **Lens Protocol** network using a Graph Attention Network (GAT) model. It analyzes account relationships across four distinct layers:

1. **Follow Layer**: Direct follow relationships
2. **Interaction Layer**: Comments, quotes, upvotes, tips, collects
3. **Co-ownership Layer**: Accounts sharing the same wallet address
4. **Similarity Layer**: Accounts with similar handles, bios, avatars, or creation times

The system provides both real-time prediction for individual profiles and batch data exploration capabilities with interactive visualizations.

---

## Features

### 🔍 Sybil Detector (Model Inference)
- Real-time profile analysis using a trained GAT model
- Confidence scoring with risk level classification (High/Medium/Low)
- Interactive network visualization with PyVis
- Support for both mock data and live BigQuery data

### 📊 Data Exploration Dashboard
- Query raw Lens Protocol data from BigQuery
- Construct 4-layer graphs on-the-fly
- Visualize edge distributions and creation patterns
- Export processed data as CSV files

### 🧪 Model Laboratory (NEW)
- **Data Ingestion**: Load from exploration or upload custom CSV files
- **Unsupervised Clustering**: K-Means with automatic optimal K selection
- **Semi-Supervised Labeling**: Rule-based pseudo-labeling with configurable thresholds
- **Supervised Training**: GAT model training with live metrics visualization
- **Model Export**: Save trained models for deployment

### 🎨 Dual Visualization Modes
- **Interactive (PyVis)**: Zoomable, pannable graph with hover tooltips
- **Static (Matplotlib)**: Clean visualizations for reports and fallback

### 🛡️ Robust Design
- Graceful BigQuery credential handling
- Automatic fallback from interactive to static visualization
- Performance guards for large datasets

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                             STREAMLIT WEB APP                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐             │
│  │ Data Exploration │  │ Model Laboratory │  │  Sybil Detector  │             │
│  │   (pages/1_...)  │  │   (pages/2_...)  │  │   (pages/3_...)  │             │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘             │
│           │                     │                     │                       │
│           ▼                     ▼                     ▼                       │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                          UTILITY LAYER                                 │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │   │
│  │  │DataLoader │  │ Clustering│  │ Labeling  │  │    Predictor      │    │   │
│  │  │(Batch ETL)│  │  Engine   │  │  Engine   │  │   (Inference)     │    │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘    │   │
│  │        │              │              │                  │              │   │
│  │        │              ▼              ▼                  │              │   │
│  │        │        ┌───────────┐  ┌───────────┐            │              │   │
│  │        │        │  Trainer  │  │ Visualizer│◄───────────┘              │   │
│  │        │        │  (GAT)    │  │(PyVis/Mpl)│                           │   │
│  │        │        └─────┬─────┘  └───────────┘                           │   │
│  │        │              │                                                │   │
│  │        │              ▼                                                │   │
│  │        │        ┌───────────┐                                          │   │
│  │        │        │ SybilGAT  │                                          │   │
│  │        │        │  (Model)  │                                          │   │
│  │        │        └───────────┘                                          │   │
│  └────────┼──────────────┼────────────────────────────────────────────────┘   │
│           │              │                                                    │
│           ▼              ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                           DATA LAYER                                   │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │   │
│  │  │    BigQuery     │    │  Local Assets   │    │  User Uploads   │     │   │
│  │  │ (Lens Protocol) │    │ (Model, Scaler) │    │  (CSV files)    │     │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘     │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
sybil-detection-app/
│
├── app.py                          # Main entry point - Navigation home page
├── config.py                       # Environment configuration
├── requirements.txt                # Python dependencies
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Data_Exploration.py       # Data exploration dashboard
│   ├── 2_Model_Laboratory.py       # ML training workbench
│   └── 3_Sybil_Detector.py         # Model inference page
│
├── models/                         # Neural network architectures
│   ├── __init__.py
│   └── gat_model.py                # SybilGAT + GATEncoder implementations
│
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── data_fetcher.py             # BigQuery fetching for single profiles
│   ├── data_loader.py              # Batch data loading (ETL)
│   ├── predictor.py                # SybilPredictor class
│   ├── visualizer.py               # Graph visualization (PyVis/Matplotlib)
│   ├── clustering_engine.py        # K-Means clustering pipeline (NEW)
│   ├── labeling_engine.py          # Rule-based labeling engine (NEW)
│   └── trainer.py                  # GAT training loop (NEW)
│
├── assets/                         # Pre-trained model & data
│   ├── best_gat_model.pt           # Trained GAT model weights
│   ├── std_scaler.bin              # Feature scaler (sklearn)
│   ├── processed_sybil_data.pt     # Reference graph (PyG Data object)
│   └── nodes_with_clusters_k21.csv # Reference node metadata
│
└── creds/                          # Credentials (gitignored)
    └── service-account-key.json    # GCP service account for BigQuery
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- (Optional) Google Cloud account with BigQuery access

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sybil-detection-app.git
   cd sybil-detection-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure BigQuery credentials**
   
   Place your GCP service account JSON key in:
   ```
   creds/service-account-key.json
   ```
   
   Or set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## Configuration

Configuration is managed through environment variables in `config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_ID` | `lens-protocol-mainnet` | BigQuery dataset ID |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-BERT model for text embeddings |
| `MAX_DAYS_RANGE` | `7` | Maximum date range for data exploration |

Set environment variables:
```bash
export DATASET_ID="lens-protocol-mainnet"
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export MAX_DAYS_RANGE="7"
```

---

## Usage

### Home Page

The home page provides navigation cards to the three main modules:
- **Data Exploration**: ETL & Analytics dashboard
- **Model Laboratory**: ML training workbench
- **Sybil Detector**: Real-time model inference

### Sybil Detector

1. Select **Data Source**:
   - `Mock Data`: Use predefined test data (no BigQuery required)
   - `Real Data (BigQuery)`: Query live Lens Protocol data

2. Enter a **Profile ID** (e.g., `0x1234...`)

3. Click **Analyze**

4. View results:
   - **Prediction**: SYBIL or NON-SYBIL
   - **Confidence Score**: Probability as percentage
   - **Risk Level**: High (>80%), Medium (50-80%), Low (<50%)
   - **Network Graph**: Visual representation of relationships

### Data Exploration

1. Select a **Date Range** (max 7 days)

2. Choose **Visualization Mode**:
   - `Interactive (PyVis)`: For exploration
   - `Static (Matplotlib)`: For reports

3. Click **Load Data**

4. Explore:
   - **Global Statistics**: Node/edge counts, average degree
   - **Network Graph**: 4-layer relationship visualization
   - **Data Tables**: Browse nodes, features, and edges
   - **Deep Insights**: Edge distribution and activity charts
   - **Download**: Export data as CSV

### Model Laboratory

The Model Laboratory is a scientific workbench with four steps:

#### Step 1: Data Ingestion
- **From Exploration**: Load data from the Data Exploration page
- **Upload Files**: Upload custom `nodes.csv` and `edges.csv` files
- Automatic removal of isolated nodes (degree = 0)

#### Step 2: Clustering
- **Auto K-Selection**: Automatically find optimal K using Silhouette Score
- **Manual K**: Specify exact number of clusters
- **Metrics**: View Silhouette Score, Davies-Bouldin Index
- **Visualization**: Cluster distribution chart, optimal K search plot

#### Step 3: Labeling
Configure thresholds for rule-based labeling:
| Threshold | Default | Description |
|-----------|---------|-------------|
| Co-Owner % | 5% | Clusters with high co-owner edges = SYBIL |
| Fuzzy Handle % | 50% | Clusters with similar handles = SYBIL |
| Similarity % | 60% | Combined similarity threshold |
| Max Trust Score | 25 | Low trust clusters are suspicious |

**Labeling Rules (Priority Order):**
1. **Co-owner Ring**: High co-owner edge ratio
2. **Name Pattern Abuse**: High fuzzy handle matching + low trust
3. **Industrial Bot Farm**: Batch creation + low social + low trust

#### Step 4: Training
Configure GAT hyperparameters:
| Parameter | Options | Default |
|-----------|---------|---------|
| Hidden Channels | 16, 32, 64, 128 | 32 |
| Attention Heads | 2, 4, 8 | 4 |
| Dropout | 0.0 - 0.6 | 0.3 |
| Learning Rate | 0.001 - 0.05 | 0.005 |
| Max Epochs | 50 - 500 | 300 |
| Early Stopping Patience | 10 - 100 | 40 |

**Training Features:**
- Live loss and F1 curve visualization
- Class weight balancing for imbalanced data
- Confusion matrix and classification report
- Model export to `assets/` directory

---

## Technical Details

### Dependencies

```
# Core Framework
streamlit>=1.32.0
plotly>=5.18.0          # For live training charts (NEW)

# Deep Learning
torch>=2.0.0
torch-geometric>=2.4.0

# NLP
sentence-transformers>=2.2.0
rapidfuzz>=3.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0     # K-Means, metrics, train_test_split
joblib>=1.3.0

# Visualization
matplotlib>=3.7.0
networkx>=3.1
pyvis>=0.3.2

# BigQuery (Optional)
google-cloud-bigquery>=3.11.0
db-dtypes>=1.1.0
```

### Feature Engineering

The model uses 396-dimensional feature vectors:

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| Text Embedding (S-BERT) | 384 | Encoded handle + name + bio |
| Avatar | 1 | Binary: has custom avatar |
| Statistics | 10 | Normalized activity metrics |
| Account Age | 1 | Days since creation |

**Statistics features:**
- `total_tips`, `total_posts`, `total_quotes`, `total_reacted`
- `total_reactions`, `total_reposts`, `total_collects`, `total_comments`
- `total_followers`, `total_following`

---

## API Reference

### `SybilPredictor`

Main class for running Sybil predictions.

```python
from utils.predictor import SybilPredictor

predictor = SybilPredictor(
    model_path="assets/best_gat_model.pt",      # Optional
    scaler_path="assets/std_scaler.bin",        # Optional
    ref_data_path="assets/processed_sybil_data.pt",  # Optional
    node_info_path="assets/nodes_with_clusters_k21.csv"  # Optional
)

result, edges, types, dirs = predictor.predict(
    profile_id="0x1234...",
    fetch_data_func=mock_bq_fetcher  # or bq_fetcher
)
```

**Returns:**
- `result`: Dict with prediction, confidence, risk level
- `edges`: Tensor of edge indices
- `types`: List of edge type names
- `dirs`: List of edge directions

### `DataLoader`

Class for batch data fetching and processing.

```python
from utils.data_loader import DataLoader

loader = DataLoader()
data = loader.fetch_and_process_data(start_date, end_date)

# Returns:
# {
#     'nodes_df': pd.DataFrame,    # Node metadata
#     'features_df': pd.DataFrame, # Node features
#     'edges_df': pd.DataFrame,    # Edge list
#     'warnings': List[str]        # Any warnings
# }
```

### `ClusteringEngine`

Engine for unsupervised node clustering.

```python
from utils.clustering_engine import ClusteringEngine

engine = ClusteringEngine()

# Full pipeline: Features -> Clustering
result, optimal_k, pyg_data = engine.process_and_cluster(
    nodes_df,
    edges_df,
    n_clusters=None,      # Auto-select optimal K
    k_range=(2, 15),      # Search range
    random_state=42
)

# Returns:
# - result: ClusteringResult with labels, metrics, centroids
# - optimal_k: OptimalKResult with K search metrics
# - pyg_data: PyTorch Geometric Data object
```

### `LabelingEngine`

Engine for rule-based cluster labeling.

```python
from utils.labeling_engine import LabelingEngine, LabelingThresholds

thresholds = LabelingThresholds(
    pct_co_owner=0.05,
    pct_fuzzy_handle=0.50,
    pct_similarity=0.60,
    co_owner_avg_trust=25
)

engine = LabelingEngine(thresholds)

# Profile clusters
profiles = engine.profile_clusters(nodes_df, edges_df, cluster_labels)

# Apply rules
results, summary_df = engine.label_clusters(profiles)

# Generate node-level labels
node_labels = engine.generate_node_labels(nodes_df, cluster_labels, results)
# Returns: np.ndarray with 0 (NON-SYBIL) or 1 (SYBIL)
```

### `GATrainer`

Trainer for GAT model with early stopping.

```python
from utils.trainer import GATrainer, TrainingConfig

config = TrainingConfig(
    hidden_channels=32,
    heads=4,
    dropout=0.3,
    learning_rate=0.005,
    epochs=300,
    patience=40
)

trainer = GATrainer(config)

# Train with callback for live updates
def callback(epoch, metrics):
    print(f"Epoch {epoch}: F1={metrics['val_f1']:.4f}")

history = trainer.train(pyg_data, node_labels, callback=callback)

# Evaluate on test set
eval_result = trainer.evaluate_test(pyg_data)
print(f"Test F1: {eval_result.f1_macro:.4f}")

# Save model
trainer.save_model("assets/my_model.pt")
```

### Visualization Functions

```python
from utils.visualizer import render_static_graph, render_interactive_graph

# Static visualization (Matplotlib)
fig = render_static_graph(
    node_info, edges, types, dirs, df_ref, result, ref_labels
)

# Interactive visualization (PyVis)
html_path = render_interactive_graph(
    node_info, edges, types, dirs, df_ref, result, ref_labels
)
```

---

## Model Architecture

### SybilGAT (Graph Attention Network)

The model supports two modes:
1. **Classification Mode**: Full forward pass with log-softmax output
2. **Embedding Mode**: Return Layer 2 output for clustering (64 dimensions)

```
Input Features (396 dim)
         │
         ▼
┌─────────────────────────────────┐
│  GAT Layer 1 (4 heads)          │
│  396 → 32 × 4 = 128             │
│  Dropout: 0.3, Activation: ELU  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GAT Layer 2 (2 heads)          │
│  128 → 32 × 2 = 64              │◄──── Embedding Output (for clustering)
│  Dropout: 0.3, Activation: ELU  │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GAT Layer 3 (1 head)           │
│  64 → 2 (num_classes)           │
│  Dropout: 0.3                   │
│  Output: Log-Softmax            │
└─────────────────────────────────┘
         │
         ▼
   [Non-Sybil, Sybil] Probabilities
```

### Usage

```python
from models.gat_model import SybilGAT

model = SybilGAT(
    num_features=396,
    hidden_channels=32,
    num_classes=2,
    heads=4,
    dropout=0.3
)

# Classification mode (default)
out = model(x, edge_index)  # [num_nodes, 2]

# Embedding mode (for clustering)
embeddings = model(x, edge_index, return_embeddings=True)  # [num_nodes, 64]

# Or use the helper method
embeddings = model.get_embeddings(x, edge_index)  # [num_nodes, 64]
```

### Edge Weight Configuration

Different edge types have different importance weights for the GNN:

| Layer | Edge Type | Weight |
|-------|-----------|--------|
| Follow | FOLLOW | 2 |
| Interact | UPVOTE, REACTION | 1 |
| Interact | COMMENT, QUOTE | 2 |
| Interact | MIRROR | 3 |
| Interact | COLLECT | 4 |
| Interact | TIP | 5 |
| Co-owner | CO-OWNER | 5 |
| Similarity | SAME_AVATAR | 3 |
| Similarity | FUZZY_HANDLE | 2 |
| Similarity | SIM_BIO | 3 |
| Similarity | CLOSE_CREATION_TIME | 2 |

---

## Data Pipeline

### Single Profile Prediction Flow

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  Profile ID │────▶│  Data Fetcher   │────▶│  Feature Processing │
└─────────────┘     │ (BigQuery/Mock) │     │  - S-BERT encoding  │
                    └─────────────────┘     │  - Stats scaling    │
                                            └──────────┬──────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EDGE DISCOVERY                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Layer 1: Follow        │  Match against reference follows  ││
│  │  Layer 2: Interact      │  Match against reference posts    ││
│  │  Layer 3: Co-owner      │  Same wallet address              ││
│  │  Layer 4: Similarity    │  Bio/Handle/Avatar/Time           ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GAT MODEL INFERENCE                          │
│  Reference Graph (19K+ nodes) + New Node + New Edges            │
│                               │                                 │
│                               ▼                                 │
│              [Sybil Probability, Non-Sybil Probability]         │
└─────────────────────────────────────────────────────────────────┘
```

### Batch Data Exploration Flow

```
┌───────────────────┐     ┌─────────────────────────────────────────┐
│ Date Range Input  │────▶│           BigQuery Queries              │
└───────────────────┘     │  1. Profiles (metadata, handle, wallet) │
                          │  2. Features (post stats, followers)    │
                          │  3. Follows (follow relationships)      │
                          │  4. Interactions (comments, tips, etc.) │
                          └──────────────────┬──────────────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LOCAL COMPUTATION                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Co-owner Edges    │  Vectorized self-join on wallet address   │ │
│  │  Similarity Edges  │  SBERT cosine, Fuzzy handle, Avatar URL   │ │
│  │                    │  (Skipped if > 1000 nodes)                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
┌───────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                    │
│   nodes_df: ID, handle, bio, owned_by, trust_score, created_on    │
│   features_df: ID, total_posts, total_followers, ...              │
│   edges_df: source, target, type, layer                           │
└───────────────────────────────────────────────────────────────────┘
```

### Model Laboratory Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. DATA INGESTION                               │
│  ┌──────────────────────┐    ┌──────────────────────┐               │
│  │  From Exploration    │ OR │   Upload CSV Files   │               │
│  │  (session_state)     │    │ (nodes.csv, edges)   │               │
│  └──────────┬───────────┘    └──────────┬───────────┘               │
│             └────────────────┬──────────┘                           │
│                              ▼                                      │
│             ┌────────────────────────────┐                          │
│             │  Remove Isolated Nodes     │                          │
│             │  (degree = 0)              │                          │
│             └────────────────┬───────────┘                          │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     2. CLUSTERING (Unsupervised)                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Feature Engineering                                           │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │ │
│  │  │  S-BERT   │ │  Avatar   │ │   Stats   │ │  Account  │       │ │
│  │  │  (384d)   │ │   (1d)    │ │  (10d)    │ │  Age (1d) │       │ │
│  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘       │ │
│  │        └─────────────┴─────────────┴─────────────┘             │ │
│  │                              │                                  │ │
│  │                              ▼                                  │ │
│  │                 ┌────────────────────┐                          │ │
│  │                 │ Feature Matrix     │                          │ │
│  │                 │ (396 dimensions)   │                          │ │
│  │                 └─────────┬──────────┘                          │ │
│  └───────────────────────────┼────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  K-Means Clustering                                            │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │ │
│  │  │ Optimal K Search│──│ Silhouette Score│──│ Best K Selected │ │ │
│  │  │  (K = 2...15)   │  │ Davies-Bouldin  │  │ (Auto or Manual)│ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     3. LABELING (Semi-Supervised)                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Cluster Profiling                                             │ │
│  │  - Avg Trust Score       - Pct Co-Owner Edges                  │ │
│  │  - Std Creation Time     - Pct Fuzzy Handle                    │ │
│  │  - Pct Social Activity   - Pct Similarity Edges                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Rule-Based Classification (Priority Order)                    │ │
│  │  ┌─────────────────────────────────────────────────────────┐   │ │
│  │  │ P1: Co-Owner Ring      │ pct_co_owner > threshold       │   │ │
│  │  │ P2: Name Pattern Abuse │ fuzzy_handle + low_trust       │   │ │
│  │  │ P3: Industrial Bot Farm│ batch_creation + low_social    │   │ │
│  │  │ Default: Organic       │ Otherwise = NON-SYBIL          │   │ │
│  │  └─────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│             ┌────────────────────────────┐                          │
│             │  Pseudo-Labels (y_pseudo)  │                          │
│             │  0 = NON-SYBIL, 1 = SYBIL  │                          │
│             └────────────────┬───────────┘                          │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     4. TRAINING (Supervised)                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Data Preparation                                              │ │
│  │  - Stratified Train/Val/Test Split (60/20/20)                  │ │
│  │  - Class Weight Calculation (for imbalance)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  GAT Training Loop                                             │ │
│  │  ┌─────────────────────────────────────────────────────────┐   │ │
│  │  │  Optimizer: Adam (lr=0.005, weight_decay=5e-4)          │   │ │
│  │  │  Loss: NLLLoss with class weights                       │   │ │
│  │  │  Early Stopping: patience=40, monitor=val_f1            │   │ │
│  │  └─────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Evaluation & Export                                           │ │
│  │  - Confusion Matrix   - F1-Score (Macro)   - Accuracy          │ │
│  │  - Classification Report   - Model Checkpoint (.pt)            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Visualization

### Color Coding

**Nodes:**
| Type | Sybil | Non-Sybil |
|------|-------|-----------|
| Target | Red (#dc2626) + White border | Green (#16a34a) + White border |
| Reference | Red (#dc2626) | Green (#16a34a) |

**Edges:**
| Layer | Color | Style | Direction |
|-------|-------|-------|-----------|
| Follow | Blue (#3b82f6) | Solid | Directed → |
| Interact | Cyan (#06b6d4) | Solid | Directed → |
| Co-owner | Red (#dc2626) | Dashed | Undirected |
| Similarity | Purple (#7c3aed) | Dotted | Undirected |

### Interactive Features (PyVis)

- **Zoom**: Mouse wheel or pinch
- **Pan**: Click and drag
- **Hover**: View node/edge details
- **Physics**: Auto-layout with Barnes-Hut algorithm

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for public APIs
- Test with both mock and real data sources

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Lens Protocol](https://lens.xyz/) for the decentralized social graph data
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for GNN implementation
- [Streamlit](https://streamlit.io/) for the web application framework
- [PyVis](https://pyvis.readthedocs.io/) for interactive network visualization
- [Sentence-Transformers](https://www.sbert.net/) for text embeddings

---

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.
