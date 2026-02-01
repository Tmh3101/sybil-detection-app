"""Sybil prediction engine using Graph Attention Networks."""

import os
import warnings
from typing import Dict, List, Tuple, Any, Optional, Callable

import torch
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from models.gat_model import SybilGAT
from config import EMBEDDING_MODEL

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')


class SybilPredictor:
    """
    Sybil detection predictor using GAT-based graph neural network.
    
    This class handles:
        - Loading pre-trained model weights and scalers
        - Processing new node features
        - Finding dynamic edges based on similarity metrics
        - Running inference and returning predictions
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        ref_data_path: Optional[str] = None,
        node_info_path: Optional[str] = None
    ):
        """
        Initialize the Sybil prediction system.
        
        Args:
            model_path: Path to trained GAT model weights (.pth)
            scaler_path: Path to fitted StandardScaler (.bin)
            ref_data_path: Path to reference graph data (.pth)
            node_info_path: Path to node information CSV
        """
        # Default paths from assets directory
        model_path = model_path or os.path.join(ASSETS_DIR, 'best_gat_model.pth')
        scaler_path = scaler_path or os.path.join(ASSETS_DIR, 'std_scaler.bin')
        ref_data_path = ref_data_path or os.path.join(ASSETS_DIR, 'processed_sybil_data.pth')
        node_info_path = node_info_path or os.path.join(ASSETS_DIR, 'nodes_with_clusters_k21.csv')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load reference data
        self.ref_data = torch.load(ref_data_path, map_location=self.device, weights_only=False)
        self.num_ref_nodes = self.ref_data.x.shape[0]
        
        # Load reference info CSV
        self.df_ref = pd.read_csv(node_info_path)
        
        # Convert datetime
        if 'created_on' in self.df_ref.columns:
            self.df_ref['created_on'] = pd.to_datetime(self.df_ref['created_on'], utc=True)
        
        # Load model
        self.model = SybilGAT(
            num_features=self.ref_data.num_features,
            hidden_channels=32,
            num_classes=2,
            heads=4
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize S-BERT and cache embeddings
        self.sbert = SentenceTransformer(EMBEDDING_MODEL)
        
        # Cache bio embeddings
        self.df_ref['metadata'] = self.df_ref['metadata'].fillna("")
        ref_bios = self.df_ref['metadata'].tolist()
        self.ref_bio_embeddings = self.sbert.encode(ref_bios, show_progress_bar=False)
        
        self._initialized = True

    def process_new_node_features(
        self, 
        node_info: Dict[str, Any], 
        node_stats: List[float]
    ) -> torch.Tensor:
        """
        Create feature vector for a new node.
        
        Args:
            node_info: Dictionary with node metadata
            node_stats: List of 10 statistical features
            
        Returns:
            Tensor of normalized features [1, num_features]
        """
        raw_stats_array = np.array(node_stats).reshape(1, -1)
        stats_norm = self.scaler.transform(raw_stats_array)
        return torch.tensor(stats_norm, dtype=torch.float).to(self.device)

    def find_dynamic_edges(
        self, 
        node_info: Dict[str, Any], 
        interactions: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Find edges connecting new node to reference graph.
        
        Args:
            node_info: Dictionary with node metadata
            interactions: List of profile IDs the node has interacted with
            
        Returns:
            Tuple of (edge_index tensor [2, E], list of edge types)
        """
        new_edges = []
        edge_types = []
        new_node_idx = self.num_ref_nodes

        # 1. Social connections (follows/interactions)
        if interactions:
            matched_indices = self.df_ref.index[
                self.df_ref['profile_id'].isin(interactions)
            ].tolist()
            for idx in matched_indices:
                new_edges.append([new_node_idx, idx])
                edge_types.append('social')
                new_edges.append([idx, new_node_idx])
                edge_types.append('social')

        # 2. Co-owner detection
        my_wallet = node_info.get('owned_by')
        if my_wallet:
            co_owners = self.df_ref.index[
                self.df_ref['owned_by'] == my_wallet
            ].tolist()
            for idx in co_owners:
                new_edges.append([new_node_idx, idx])
                edge_types.append('co_owner')
                new_edges.append([idx, new_node_idx])
                edge_types.append('co_owner')

        # 3. Bio similarity (S-BERT)
        my_bio = str(node_info.get('bio', ''))
        if len(my_bio) > 3:
            my_bio_emb = self.sbert.encode([my_bio])
            sim_scores = cosine_similarity(my_bio_emb, self.ref_bio_embeddings)[0]
            high_sim_indices = np.where(sim_scores > 0.85)[0]
            for idx in high_sim_indices:
                new_edges.append([new_node_idx, idx])
                edge_types.append('similarity')
                new_edges.append([idx, new_node_idx])
                edge_types.append('similarity')

        # 4. Handle similarity (fuzzy matching)
        my_handle = str(node_info.get('handle', ''))
        if len(my_handle) > 3:
            for idx, row in self.df_ref.iterrows():
                if fuzz.ratio(my_handle, str(row['handle'])) > 90:
                    new_edges.append([new_node_idx, idx])
                    edge_types.append('similarity')
                    new_edges.append([idx, new_node_idx])
                    edge_types.append('similarity')

        # 5. Avatar similarity
        my_picture = str(node_info.get('picture_url', ''))
        if len(my_picture) > 5 and "default" not in my_picture.lower():
            same_avatar_indices = self.df_ref.index[
                self.df_ref['metadata'].apply(lambda x: my_picture in str(x))
            ].tolist()
            for idx in same_avatar_indices:
                new_edges.append([new_node_idx, idx])
                edge_types.append('similarity')
                new_edges.append([idx, new_node_idx])
                edge_types.append('similarity')

        # 6. Time proximity (within 5 seconds)
        my_created_on = node_info.get('created_on')
        if my_created_on:
            try:
                ts_new = pd.to_datetime(my_created_on, utc=True)
                time_diffs = (self.df_ref['created_on'] - ts_new).dt.total_seconds().abs()
                close_time_indices = np.where(time_diffs <= 5)[0]
                for idx in close_time_indices:
                    new_edges.append([new_node_idx, idx])
                    edge_types.append('similarity')
                    new_edges.append([idx, new_node_idx])
                    edge_types.append('similarity')
            except Exception:
                pass

        # Return results
        if not new_edges:
            return torch.empty((2, 0), dtype=torch.long).to(self.device), []

        return (
            torch.tensor(new_edges, dtype=torch.long).t().contiguous().to(self.device),
            edge_types
        )

    def predict(
        self, 
        profile_id: str, 
        fetch_data_func: Callable[[str], Optional[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], torch.Tensor, List[str]]:
        """
        Run Sybil prediction for a profile.
        
        Args:
            profile_id: The target profile ID
            fetch_data_func: Function to fetch profile data
            
        Returns:
            Tuple of (result dict, edge index tensor, edge types list)
        """
        # Fetch data
        raw_data = fetch_data_func(profile_id)
        if raw_data is None:
            return {"error": "Profile not found"}, torch.empty((2, 0)), []

        node_info = raw_data['info']
        node_stats = raw_data['stats']
        interactions = raw_data['interactions']

        # Process features
        x_new = self.process_new_node_features(node_info, node_stats)
        x_combined = torch.cat([self.ref_data.x, x_new], dim=0)

        # Find edges
        new_edge_index, new_edge_types = self.find_dynamic_edges(node_info, interactions)

        if new_edge_index.shape[1] > 0:
            edge_index_combined = torch.cat(
                [self.ref_data.edge_index, new_edge_index], dim=1
            )
            edge_count = new_edge_index.shape[1] // 2
        else:
            edge_index_combined = self.ref_data.edge_index
            edge_count = 0

        # Run inference
        with torch.no_grad():
            out = self.model(x_combined, edge_index_combined)

        # Process result
        prediction = out[-1]
        probs = torch.exp(prediction)
        sybil_prob = probs[1].item()

        result = {
            "profile_id": profile_id,
            "handle": node_info.get('handle'),
            "prediction": "SYBIL" if sybil_prob > 0.5 else "NON-SYBIL",
            "sybil_probability": sybil_prob,
            "sybil_probability_formatted": f"{sybil_prob * 100:.2f}%",
            "node_info": node_info,
            "analysis": {
                "edges_found": edge_count,
                "has_co_owner": 'co_owner' in new_edge_types,
                "risk_level": (
                    "High" if sybil_prob > 0.8 
                    else ("Medium" if sybil_prob > 0.5 else "Low")
                )
            }
        }

        return result, new_edge_index, new_edge_types
