import torch
import numpy as np
import pandas as pd
import joblib
import os
import ast
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from models.gat_model import SybilGAT
from config import EMBEDDING_MODEL


def _parse_metadata(meta_str: str) -> Tuple[str, str]:
    """Parse metadata JSON string to extract bio and picture_url."""
    try:
        if pd.isna(meta_str) or not meta_str:
            return '', ''
        meta = ast.literal_eval(meta_str)
        lens_data = meta.get('lens', {})
        bio = lens_data.get('bio', '') or ''
        picture = lens_data.get('picture', '')
        if isinstance(picture, dict):
            picture = picture.get('item', '') or picture.get('url', '') or ''
        return bio, picture or ''
    except Exception:
        return '', ''

EDGE_WEIGHTS = {
    # Layer 1: Follow - có hướng
    'FOLLOW': 2,

    # Layer 2: Interact - có hướng
    'UPVOTE': 1,
    'REACTION': 1,
    'COMMENT': 2,
    'QUOTE': 2,
    'MIRROR': 3, # Repost
    'COLLECT': 4,
    'TIP': 5,    # High value action
    'DEFAULT_INTERACT': 1, # Dự phòng nếu không rõ loại

    # Layer 3: Co-owner - vô hướng
    'CO-OWNER': 5,

    # Layer 4: Similarity - vô hướng
    'SAME_AVATAR': 3,
    'FUZZY_HANDLE': 2,
    'SIM_BIO': 3,
    'CLOSE_CREATION_TIME': 2,
}

ASSETS_DIR = 'assets/' 

class SybilPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        ref_data_path: Optional[str] = None,
        node_info_path: Optional[str] = None
    ):
        model_path = model_path or os.path.join(ASSETS_DIR, 'best_gat_model.pt')
        scaler_path = scaler_path or os.path.join(ASSETS_DIR, 'std_scaler.bin')
        ref_data_path = ref_data_path or os.path.join(ASSETS_DIR, 'processed_sybil_data.pt')
        node_info_path = node_info_path or os.path.join(ASSETS_DIR, 'nodes_with_clusters_k21.csv')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 Initializing Predictor on {self.device}...")

        # Load Resources
        self.scaler = joblib.load(scaler_path)
        self.ref_data = torch.load(ref_data_path, map_location=self.device, weights_only=False)
        self.num_ref_nodes = self.ref_data.x.shape[0]
        self.ref_x = self.ref_data.x
        self.ref_edge_index = self.ref_data.edge_index
        
        # Load Metadata DataFrame
        self.df_ref = pd.read_csv(node_info_path)
        self.df_ref = self.df_ref.reset_index(drop=True)

        if 'created_on' in self.df_ref.columns:
            self.df_ref['created_on'] = pd.to_datetime(self.df_ref['created_on'], utc=True)
        
        # Parse metadata to extract bio and picture_url
        self.df_ref['metadata'] = self.df_ref['metadata'].fillna("")
        parsed_meta = [_parse_metadata(m) for m in self.df_ref['metadata']]
        self.df_ref['bio'] = [x[0] for x in parsed_meta]
        self.df_ref['picture_url'] = [x[1] for x in parsed_meta]
        
        # Load S-BERT & Cache Embeddings
        self.sbert = SentenceTransformer(EMBEDDING_MODEL)
        
        # Use parsed bio for embeddings
        ref_bios = self.df_ref['bio'].tolist()
        self.ref_bio_embeddings = self.sbert.encode(ref_bios, show_progress_bar=False)

        # Load Model Architecture
        num_features = self.ref_x.shape[1]
        
        self.model = SybilGAT(
            num_features=num_features,
            hidden_channels=32,
            num_classes=2,
            heads=4
        ).to(self.device)
        
        # Load Weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def process_new_node_features(
        self, 
        node_info: Dict[str, Any], 
        node_stats: List[float]
    ) -> torch.Tensor:
        """
        Tái tạo quy trình Feature Engineering giống hệt lúc Train:
        1. S-BERT (Handle + Name + Bio) -> 384 dim
        2. Avatar (0/1) -> 1 dim
        3. Stats (MinMax Scaled) -> 11 dim
        Tổng: ~396 dims
        """
        # Text Embedding
        handle = node_info.get('handle', '') or ''
        name = node_info.get('display_name', '') or ''
        bio = node_info.get('bio', '') or ''
        
        text_raw = f"Handle: {handle}. Name: {name}. Bio: {bio}"
        text_emb = self.sbert.encode([text_raw])  # Shape (1, 384)

        # Avatar Feature
        pic = str(node_info.get('picture_url', ''))
        has_avatar = 0 if (not pic or 'default' in pic.lower()) else 1
        img_feat = np.array([[has_avatar]])

        # Stats & Time Feature
        created_at = pd.to_datetime(node_info.get('created_on', pd.Timestamp.now(tz='UTC')))
        if created_at.tz is None:
            created_at = created_at.tz_localize('UTC')
        days_active = (pd.Timestamp.now(tz='UTC') - created_at).days

        # Build stats array - ensure no NaN values
        raw_stats_list = [float(s) if s is not None and not np.isnan(s) else 0.0 for s in node_stats]
        raw_stats_list.append(float(days_active))
        
        # Scale - reshape to 2D array for scaler
        stats_array = np.array(raw_stats_list).reshape(1, -1)
        stats_norm = self.scaler.transform(stats_array)

        x_new = np.hstack([text_emb, img_feat, stats_norm])
        return torch.tensor(x_new, dtype=torch.float).to(self.device)

    def _add_weighted_edge(
        self, 
        src_list, dst_list, type_list, dir_list,
        u, v, weight, edge_type_name, is_directed=True
    ):
        """
        Helper: Add weighted edges for GNN (bidirectional) and track direction for visualization.
        
        Args:
            src_list, dst_list: Edge source/destination lists
            type_list: Edge type list
            dir_list: Edge direction list ('outgoing', 'incoming', 'undirected')
            u, v: Source and target node indices
            weight: Edge weight (number of times to duplicate)
            edge_type_name: Type name for logging
            is_directed: If True, track as directed edge; if False, track as undirected
        """
        for _ in range(weight):
            # Forward edge (u -> v)
            src_list.append(u)
            dst_list.append(v)
            type_list.append(edge_type_name)
            dir_list.append('outgoing' if is_directed else 'undirected')
            
            # Reverse edge (v -> u) - needed for GNN message passing
            src_list.append(v)
            dst_list.append(u)
            type_list.append(edge_type_name)
            dir_list.append('incoming' if is_directed else 'undirected')

    def find_dynamic_edges(
        self, 
        node_info: Dict[str, Any], 
        interactions: List[Dict[str, str]],
        queried_profile_id: str = None
    ) -> Tuple[torch.Tensor, List[str], List[str]]:
        """
        Find dynamic edges between new node and reference nodes.
        
        Args:
            node_info: Target node information
            interactions: List of interactions with 'source', 'target', 'type' keys
            queried_profile_id: The profile ID being queried (to determine edge direction)
        
        Returns:
            Tuple of (edge_tensor, edge_types, edge_directions)
            - edge_directions: 'outgoing' (new->ref), 'incoming' (ref->new), 'undirected'
        """
        src_list = []
        dst_list = []
        edge_types_log = []
        edge_directions = []
        
        new_node_idx = self.num_ref_nodes

        # Process FOLLOW and INTERACT layers together
        if interactions:
            for interact in interactions:
                i_type = interact.get('type', 'DEFAULT_INTERACT')
                source_id = interact.get('source')
                target_id = interact.get('target')
                
                weight = EDGE_WEIGHTS.get(i_type, 1)
                
                # Determine which profile is the reference node
                # If source == queried profile: new_node -> ref_node (outgoing)
                # If target == queried profile: ref_node -> new_node (incoming)
                if source_id == queried_profile_id:
                    # Outgoing: new_node is the source, look up target in ref_data
                    ref_profile_id = target_id
                    matches = self.df_ref.index[self.df_ref['profile_id'] == ref_profile_id].tolist()
                    for idx in matches:
                        self._add_weighted_edge(
                            src_list, dst_list, edge_types_log, edge_directions,
                            new_node_idx, idx,  # new_node -> ref_node
                            weight=weight,
                            edge_type_name=i_type,
                            is_directed=True
                        )
                elif target_id == queried_profile_id:
                    # Incoming: ref_node is the source, look up source in ref_data
                    ref_profile_id = source_id
                    matches = self.df_ref.index[self.df_ref['profile_id'] == ref_profile_id].tolist()
                    for idx in matches:
                        self._add_weighted_edge(
                            src_list, dst_list, edge_types_log, edge_directions,
                            idx, new_node_idx,  # ref_node -> new_node
                            weight=weight,
                            edge_type_name=i_type,
                            is_directed=True
                        )

        # CO-OWNER LAYER (undirected - mutual relationship)
        my_wallet = node_info.get('owned_by')
        if my_wallet:
            co_owners = self.df_ref.index[self.df_ref['owned_by'] == my_wallet].tolist()
            for idx in co_owners:
                self._add_weighted_edge(
                    src_list, dst_list, edge_types_log, edge_directions,
                    new_node_idx, idx,
                    weight=EDGE_WEIGHTS['CO-OWNER'],
                    edge_type_name='CO-OWNER',
                    is_directed=False  # Undirected
                )

        # SIMILARITY LAYER (all undirected - mutual relationships)
        
        # Sim Bio (Cosine Similarity > 0.85)
        my_bio = str(node_info.get('bio', ''))
        if len(my_bio) > 10:
            my_bio_emb = self.sbert.encode([my_bio])
            sim_scores = cosine_similarity(my_bio_emb, self.ref_bio_embeddings)[0]
            high_sim_indices = np.where(sim_scores > 0.85)[0]
            for idx in high_sim_indices:
                self._add_weighted_edge(
                    src_list, dst_list, edge_types_log, edge_directions,
                    new_node_idx, idx,
                    weight=EDGE_WEIGHTS['SIM_BIO'],
                    edge_type_name='SIM_BIO',
                    is_directed=False
                )

        # Fuzzy Handle (Trùng tên > 90%)
        my_handle = str(node_info.get('handle', ''))
        if len(my_handle) > 3:
            for idx, row in self.df_ref.iterrows():
                if fuzz.ratio(my_handle, str(row['handle'])) > 90:
                    self._add_weighted_edge(
                        src_list, dst_list, edge_types_log, edge_directions,
                        new_node_idx, idx,
                        weight=EDGE_WEIGHTS['FUZZY_HANDLE'],
                        edge_type_name='FUZZY_HANDLE',
                        is_directed=False
                    )

        # Same Avatar (URL matching)
        my_picture = str(node_info.get('picture_url', ''))
        if len(my_picture) > 5 and 'default' not in my_picture.lower():
            same_avatar_indices = self.df_ref.index[
                self.df_ref['picture_url'] == my_picture
            ].tolist()
            for idx in same_avatar_indices:
                self._add_weighted_edge(
                    src_list, dst_list, edge_types_log, edge_directions,
                    new_node_idx, idx,
                    weight=EDGE_WEIGHTS['SAME_AVATAR'],
                    edge_type_name='SAME_AVATAR',
                    is_directed=False
                )

        # Close Creation Time (Trong vòng 5 giây)
        my_created_on = node_info.get('created_on')
        if my_created_on:
            try:
                ts_new = pd.to_datetime(my_created_on, utc=True)
                time_diffs = (self.df_ref['created_on'] - ts_new).dt.total_seconds().abs()
                close_time_indices = np.where(time_diffs <= 5)[0]
                
                for idx in close_time_indices:
                    self._add_weighted_edge(
                        src_list, dst_list, edge_types_log, edge_directions,
                        new_node_idx, idx,
                        weight=EDGE_WEIGHTS['CLOSE_CREATION_TIME'],
                        edge_type_name='CLOSE_CREATION_TIME',
                        is_directed=False
                    )
            except Exception:
                pass

        # Return results
        if not src_list:
            return torch.empty((2, 0), dtype=torch.long).to(self.device), [], []

        new_edges_tensor = torch.tensor([src_list, dst_list], dtype=torch.long).to(self.device)
        return new_edges_tensor, edge_types_log, edge_directions

    def predict(
        self, 
        profile_id: str, 
        fetch_data_func: Callable[[str], Optional[Dict[str, Any]]]
    ) -> Tuple[Dict[str, Any], torch.Tensor, List[str], List[str]]:
        """
        Run prediction for a profile.
        
        Returns:
            Tuple of (result_dict, edge_tensor, edge_types, edge_directions)
        """
        # Fetch Data
        raw_data = fetch_data_func(profile_id)
        if raw_data is None:
            return {"error": "Profile not found"}, torch.empty((2, 0)), [], []

        node_info = raw_data['info']
        node_stats = raw_data['stats']
        interactions = raw_data.get('interactions', [])

        # Process features
        x_new = self.process_new_node_features(node_info, node_stats)
        
        # Combine feature node mới vào feature cũ
        x_combined = torch.cat([self.ref_x, x_new], dim=0)

        # Find edges
        new_edge_index, new_edge_types, new_edge_dirs = self.find_dynamic_edges(
            node_info, interactions, queried_profile_id=profile_id
        )

        if new_edge_index.shape[1] > 0:
            edge_index_combined = torch.cat(
                [self.ref_edge_index, new_edge_index], dim=1
            )
            # Count unique edges for report
            edge_count_report = len(set([(u.item(), v.item()) for u, v in new_edge_index.t() if u.item() < v.item()]))
        else:
            edge_index_combined = self.ref_edge_index
            edge_count_report = 0

        # Inference
        with torch.no_grad():
            out = self.model(x_combined, edge_index_combined)

        # Result Interpretation
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
                "edges_found": edge_count_report,
                "weighted_edges_processed": new_edge_index.shape[1],
                "risk_types": list(set(new_edge_types)),
                "risk_level": (
                    "High" if sybil_prob > 0.8 
                    else ("Medium" if sybil_prob > 0.5 else "Low")
                )
            }
        }

        return result, new_edge_index, new_edge_types, new_edge_dirs