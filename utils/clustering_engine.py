"""Clustering engine for unsupervised node grouping.

This module handles:
1. Feature engineering (S-BERT embeddings, statistics normalization)
2. Graph construction with weighted edges
3. GAT-based embedding generation
4. K-Means clustering with optimal K selection

Logic adapted from colab-code/gae.py (Feature Engineering, K-Means sections)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Optional imports for feature engineering
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

from config import EMBEDDING_MODEL


@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray
    k: int
    silhouette: float
    davies_bouldin: float
    embeddings: np.ndarray
    centroids: np.ndarray


@dataclass
class OptimalKResult:
    """Container for optimal K search results."""
    best_k: int
    k_range: List[int]
    silhouette_scores: List[float]
    davies_bouldin_scores: List[float]
    inertia_values: List[float]


# Edge weights adapted from colab-code/gae.py (lines 162-180)
EDGE_WEIGHTS = {
    # Follow layer
    'FOLLOW': 2,
    # Interact layer
    'UPVOTE': 1,
    'REACTION': 1,
    'COMMENT': 2,
    'QUOTE': 2,
    'MIRROR': 3,
    'COLLECT': 4,
    'TIP': 5,
    # Co-owner layer
    'CO-OWNER': 5,
    # Similarity layer
    'SAME_AVATAR': 3,
    'FUZZY_HANDLE': 2,
    'SIM_BIO': 3,
    'CLOSE_CREATION_TIME': 2,
}


class FeatureEngineer:
    """
    Feature engineering pipeline for node data.
    
    Logic adapted from colab-code/gae.py (lines 58-99)
    """
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self._sbert = None
        self.scaler = MinMaxScaler()
        
    @property
    def sbert(self):
        """Lazy load S-BERT model."""
        if self._sbert is None and SBERT_AVAILABLE:
            self._sbert = SentenceTransformer(self.embedding_model)
        return self._sbert
    
    def create_text_embeddings(
        self, 
        handles: List[str], 
        names: List[str], 
        bios: List[str]
    ) -> np.ndarray:
        """
        Create S-BERT embeddings from text data.
        
        Logic from colab-code/gae.py (lines 60-71)
        """
        if not SBERT_AVAILABLE or self.sbert is None:
            warnings.warn("S-BERT not available. Using zero embeddings.")
            return np.zeros((len(handles), 384))
        
        text_data = [
            f"Handle: {h}. Name: {n}. Bio: {b}"
            for h, n, b in zip(handles, names, bios)
        ]
        
        embeddings = self.sbert.encode(text_data, show_progress_bar=False)
        return embeddings
    
    def create_avatar_features(self, picture_urls: List[str]) -> np.ndarray:
        """
        Create binary avatar features.
        
        Logic from colab-code/gae.py (lines 76-81)
        """
        def check_avatar(url):
            if not isinstance(url, str) or url == "" or "default" in str(url).lower():
                return 0
            return 1
        
        features = [check_avatar(url) for url in picture_urls]
        return np.array(features).reshape(-1, 1)
    
    def create_stat_features(
        self, 
        df: pd.DataFrame, 
        stat_cols: List[str],
        days_active: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize statistical features.
        
        Logic from colab-code/gae.py (lines 83-93)
        """
        stats_data = df[stat_cols].fillna(0).values
        
        if days_active is not None:
            stats_data = np.hstack([stats_data, days_active.reshape(-1, 1)])
        
        stats_norm = self.scaler.fit_transform(stats_data)
        return stats_norm
    
    def process_nodes(self, nodes_df: pd.DataFrame) -> np.ndarray:
        """
        Full feature engineering pipeline.
        
        Args:
            nodes_df: DataFrame with node metadata and features
            
        Returns:
            Feature matrix [num_nodes, num_features]
        """
        # Extract columns
        handles = nodes_df.get('handle', pd.Series([''] * len(nodes_df))).fillna('').tolist()
        names = nodes_df.get('display_name', pd.Series([''] * len(nodes_df))).fillna('').tolist()
        bios = nodes_df.get('bio', pd.Series([''] * len(nodes_df))).fillna('').tolist()
        pictures = nodes_df.get('picture_url', pd.Series([''] * len(nodes_df))).fillna('').tolist()
        
        # Text embeddings (384 dims)
        text_emb = self.create_text_embeddings(handles, names, bios)
        
        # Avatar feature (1 dim)
        img_feat = self.create_avatar_features(pictures)
        
        # Statistical features
        stat_cols = [
            'total_tips', 'total_posts', 'total_quotes', 'total_reacted',
            'total_reactions', 'total_reposts', 'total_collects',
            'total_comments', 'total_followers', 'total_following'
        ]
        
        # Filter to existing columns
        existing_stat_cols = [c for c in stat_cols if c in nodes_df.columns]
        
        # Calculate days active if created_on exists
        days_active = None
        if 'created_on' in nodes_df.columns:
            try:
                created = pd.to_datetime(nodes_df['created_on'], utc=True)
                days_active = (pd.Timestamp.now(tz='UTC') - created).dt.days.fillna(0).values
            except Exception:
                pass
        
        if existing_stat_cols:
            stats_norm = self.create_stat_features(nodes_df, existing_stat_cols, days_active)
        else:
            stats_norm = np.zeros((len(nodes_df), 1))
        
        # Concatenate all features
        x_np = np.hstack([text_emb, img_feat, stats_norm])
        
        return x_np


class GraphBuilder:
    """
    Build PyTorch Geometric Data objects from DataFrames.
    
    Logic adapted from colab-code/train.py (lines 146-176)
    """
    
    def __init__(self):
        self.id_to_idx = {}
        self.idx_to_id = {}
    
    def build_edge_index(
        self, 
        edges_df: pd.DataFrame, 
        node_ids: List[str],
        use_weights: bool = True
    ) -> torch.Tensor:
        """
        Build edge_index tensor from edges DataFrame.
        
        Args:
            edges_df: DataFrame with 'source'/'target' OR 'source_id'/'target_id' columns
            node_ids: List of valid node IDs
            use_weights: If True, replicate edges based on EDGE_WEIGHTS
            
        Returns:
            edge_index tensor [2, num_edges]
        """
        # Build ID mapping
        self.id_to_idx = {uid: i for i, uid in enumerate(node_ids)}
        self.idx_to_id = {i: uid for uid, i in self.id_to_idx.items()}
        
        src_list = []
        dst_list = []
        
        for _, row in edges_df.iterrows():
            # Handle both column naming conventions
            src_id = row.get('source') or row.get('source_id')
            dst_id = row.get('target') or row.get('target_id')
            
            # Skip if nodes not in mapping
            if src_id not in self.id_to_idx or dst_id not in self.id_to_idx:
                continue
            
            src = self.id_to_idx[str(src_id)]  # Ensure string conversion
            dst = self.id_to_idx[str(dst_id)]  # Ensure string conversion
            etype = row.get('type', 'FOLLOW')
            
            weight = EDGE_WEIGHTS.get(etype, 1) if use_weights else 1
            
            # Replicate edge based on weight
            for _ in range(weight):
                src_list.append(src)
                dst_list.append(dst)
                
                # Add reverse for undirected relations
                if etype in ['CO-OWNER', 'SAME_AVATAR', 'FUZZY_HANDLE', 
                            'SIM_BIO', 'CLOSE_CREATION_TIME']:
                    src_list.append(dst)
                    dst_list.append(src)
        
        if not src_list:
            return torch.empty((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return edge_index
    
    def create_data_object(
        self, 
        x: np.ndarray, 
        edge_index: torch.Tensor,
        y: Optional[np.ndarray] = None
    ) -> Data:
        """
        Create PyTorch Geometric Data object.
        """
        x_tensor = torch.tensor(x, dtype=torch.float)
        
        data = Data(x=x_tensor, edge_index=edge_index)
        
        if y is not None:
            data.y = torch.tensor(y, dtype=torch.long)
        
        data.num_classes = 2
        return data


class ClusteringEngine:
    """
    Main clustering engine for node grouping.
    
    Combines feature engineering, graph building, and K-Means clustering.
    Logic adapted from colab-code/gae.py (K-Means section, lines 506-597)
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_engineer = FeatureEngineer()
        self.graph_builder = GraphBuilder()
        self.scaler = MinMaxScaler()
        
    def find_optimal_k(
        self, 
        embeddings: np.ndarray,
        k_range: Tuple[int, int] = (2, 15),
        random_state: int = 42
    ) -> OptimalKResult:
        """
        Find optimal number of clusters using multiple metrics.
        
        Logic from colab-code/gae.py (lines 520-539)
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            k_range: Tuple of (min_k, max_k)
            random_state: Random seed for reproducibility
            
        Returns:
            OptimalKResult with best K and all metrics
            
        Note:
            Silhouette Score requires: 2 <= n_clusters <= n_samples - 1
            If k_range[1] >= n_samples, it will be automatically capped.
        """
        n_samples = embeddings.shape[0]
        
        # Validate and cap k_range to prevent Silhouette Score error
        k_min = max(2, k_range[0])
        k_max = min(k_range[1], n_samples - 1)
        
        if k_max < k_min:
            k_max = k_min
        
        k_values = list(range(k_min, k_max + 1))
        silhouette_scores = []
        davies_bouldin_scores = []
        inertia_values = []
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            inertia_values.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, labels))
            davies_bouldin_scores.append(davies_bouldin_score(embeddings, labels))
        
        # Best K: Highest Silhouette Score
        best_idx = np.argmax(silhouette_scores)
        best_k = k_values[best_idx]
        
        return OptimalKResult(
            best_k=best_k,
            k_range=k_values,
            silhouette_scores=silhouette_scores,
            davies_bouldin_scores=davies_bouldin_scores,
            inertia_values=inertia_values
        )
    
    def cluster(
        self, 
        embeddings: np.ndarray,
        n_clusters: int,
        random_state: int = 42
    ) -> ClusteringResult:
        """
        Perform K-Means clustering on embeddings.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            ClusteringResult with labels and metrics
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        silhouette = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        
        return ClusteringResult(
            labels=labels,
            k=n_clusters,
            silhouette=silhouette,
            davies_bouldin=db_score,
            embeddings=embeddings,
            centroids=kmeans.cluster_centers_
        )
    
    def process_and_cluster(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        n_clusters: Optional[int] = None,
        k_range: Tuple[int, int] = (2, 15),
        random_state: int = 42
    ) -> Tuple[ClusteringResult, Optional[OptimalKResult], Data]:
        """
        Full pipeline: Feature engineering -> Embedding -> Clustering
        
        Args:
            nodes_df: Node metadata DataFrame
            edges_df: Edge list DataFrame
            n_clusters: Number of clusters (if None, auto-select)
            k_range: Range for optimal K search
            random_state: Random seed
            
        Returns:
            Tuple of (ClusteringResult, OptimalKResult or None, PyG Data)
        """
        # Feature engineering
        node_ids = nodes_df['profile_id'].astype(str).tolist()  # Ensure string conversion
        x = self.feature_engineer.process_nodes(nodes_df)
        
        # Build graph
        edge_index = self.graph_builder.build_edge_index(edges_df, node_ids)
        data = self.graph_builder.create_data_object(x, edge_index)
        
        # Use features directly as embeddings for clustering
        # (In a full implementation, you'd run GAE here)
        embeddings = x
        
        # Normalize embeddings
        embeddings_norm = self.scaler.fit_transform(embeddings)
        
        # Find optimal K if not specified
        optimal_k_result = None
        if n_clusters is None:
            optimal_k_result = self.find_optimal_k(
                embeddings_norm, k_range, random_state
            )
            n_clusters = optimal_k_result.best_k
        
        # Cluster
        result = self.cluster(embeddings_norm, n_clusters, random_state)
        
        return result, optimal_k_result, data
