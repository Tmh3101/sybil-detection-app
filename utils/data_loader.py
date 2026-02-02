"""Data loader for fetching and processing Lens Protocol data from BigQuery.

This module implements the ETL logic from Build_Datasets.ipynb for the
Data Exploration dashboard.
"""

import os
import ast
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from itertools import combinations

from config import DATASET_ID, EMBEDDING_MODEL

# BigQuery imports (optional)
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Similarity imports (optional)
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def _get_bigquery_client() -> Optional[Any]:
    """Initialize and return a BigQuery client."""
    if not BIGQUERY_AVAILABLE:
        return None
    
    creds_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "creds", 
        "service-account-key.json"
    )
    
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    try:
        return bigquery.Client(location="US")
    except Exception:
        return None


def _parse_metadata(meta_str: str) -> Tuple[str, str]:
    """Parse metadata JSON to extract bio and picture_url."""
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


class DataLoader:
    """Handles data fetching and processing for the Data Exploration page."""
    
    def __init__(self):
        self.client = _get_bigquery_client()
        self._sbert_model = None
    
    @property
    def sbert_model(self):
        """Lazy load SBERT model."""
        if self._sbert_model is None and SBERT_AVAILABLE:
            self._sbert_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._sbert_model
    
    def fetch_nodes(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch node data (profiles) from BigQuery."""
        if self.client is None:
            raise RuntimeError("BigQuery client not available")
        
        query = f"""
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(meta.account) as profile_id,
            ANY_VALUE(meta.created_on) as created_on,
            ANY_VALUE(meta.name) as display_name,
            ANY_VALUE(meta.metadata) as metadata,
            ANY_VALUE(`{DATASET_ID}.app.FORMAT_HEX`(ksw.owned_by)) as owned_by,
            ARRAY_AGG(usr.local_name ORDER BY usr.timestamp DESC LIMIT 1)[OFFSET(0)] as handle,
            ARRAY_AGG(score.score ORDER BY score.generated_at DESC LIMIT 1)[OFFSET(0)] as trust_score
        FROM `{DATASET_ID}.account.metadata` as meta
        LEFT JOIN `{DATASET_ID}.username.record` as usr
            ON meta.account = usr.account
        LEFT JOIN `{DATASET_ID}.account.known_smart_wallet` as ksw
            ON meta.account = ksw.address
        LEFT JOIN `{DATASET_ID}.ml.account_score` as score
            ON meta.account = score.account
        WHERE meta.created_on >= '{start_date}'
          AND meta.created_on < '{end_date}'
        GROUP BY 1
        """
        
        df = self.client.query(query).to_dataframe()
        
        # Parse metadata to extract bio and picture_url
        parsed = [_parse_metadata(m) for m in df['metadata'].fillna('')]
        df['bio'] = [x[0] for x in parsed]
        df['picture_url'] = [x[1] for x in parsed]
        
        # Parse datetime
        df['created_on'] = pd.to_datetime(df['created_on'], utc=True)
        
        # Fill missing values
        df['handle'] = df['handle'].fillna('Unknown')
        df['owned_by'] = df['owned_by'].fillna('')
        df['trust_score'] = df['trust_score'].fillna(0)
        
        return df
    
    def fetch_node_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch node features (post and follower stats) from BigQuery."""
        if self.client is None:
            raise RuntimeError("BigQuery client not available")
        
        # Post features
        query_post = f"""
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(account) as profile_id,
            SUM(total_tips) as total_tips,
            SUM(total_posts) as total_posts,
            SUM(total_quotes) as total_quotes,
            SUM(total_reacted) as total_reacted,
            SUM(total_reactions) as total_reactions,
            SUM(total_reposts) as total_reposts,
            SUM(total_collects) as total_collects,
            SUM(total_comments) as total_comments
        FROM `{DATASET_ID}.account.post_summary`
        WHERE account IN (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        GROUP BY 1
        """
        
        # Follower features
        query_follower = f"""
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(account) as profile_id,
            SUM(total_followers) as total_followers,
            SUM(total_following) as total_following
        FROM `{DATASET_ID}.account.follower_summary`
        WHERE account IN (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        GROUP BY 1
        """
        
        df_post = self.client.query(query_post).to_dataframe()
        df_follower = self.client.query(query_follower).to_dataframe()
        
        # Merge features
        df_features = pd.merge(df_post, df_follower, on='profile_id', how='outer')
        df_features = df_features.fillna(0)
        
        return df_features
    
    def fetch_follow_edges(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch follow edges (Layer 1)."""
        if self.client is None:
            raise RuntimeError("BigQuery client not available")
        
        query = f"""
        WITH TargetUsers AS (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        SELECT DISTINCT
            `{DATASET_ID}.app.FORMAT_HEX`(f.account_follower) as source,
            `{DATASET_ID}.app.FORMAT_HEX`(f.account_following) as target,
            'FOLLOW' as type,
            'follow' as layer
        FROM `{DATASET_ID}.account.follower` as f
        JOIN TargetUsers as t1 ON f.account_follower = t1.account
        JOIN TargetUsers as t2 ON f.account_following = t2.account
        """
        
        return self.client.query(query).to_dataframe()
    
    def fetch_interaction_edges(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch interaction edges (Layer 2)."""
        if self.client is None:
            raise RuntimeError("BigQuery client not available")
        
        # Comment & Quote
        query_cq = f"""
        WITH TargetUsers AS (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(p.account) as source,
            `{DATASET_ID}.app.FORMAT_HEX`(parent.account) as target,
            CASE WHEN p.parent_post IS NOT NULL THEN 'COMMENT' ELSE 'QUOTE' END as type,
            'interact' as layer
        FROM `{DATASET_ID}.post.record` as p
        JOIN `{DATASET_ID}.post.record` as parent
            ON (p.parent_post = parent.id OR p.quoted_post = parent.id)
        JOIN TargetUsers t1 ON p.account = t1.account
        JOIN TargetUsers t2 ON parent.account = t2.account
        WHERE p.timestamp >= '{start_date}' AND p.timestamp < '{end_date}'
          AND p.account != parent.account
        """
        
        # Reactions
        query_rx = f"""
        WITH TargetUsers AS (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(r.account) as source,
            `{DATASET_ID}.app.FORMAT_HEX`(p.account) as target,
            'UPVOTE' as type,
            'interact' as layer
        FROM `{DATASET_ID}.post.reaction` as r
        JOIN `{DATASET_ID}.post.record` as p ON r.post = p.id
        JOIN TargetUsers t1 ON r.account = t1.account
        JOIN TargetUsers t2 ON p.account = t2.account
        WHERE r.action_at >= '{start_date}' AND r.action_at < '{end_date}'
          AND r.account != p.account
        """
        
        # Actions (TIP, COLLECT)
        query_act = f"""
        WITH TargetUsers AS (
            SELECT account FROM `{DATASET_ID}.account.metadata`
            WHERE created_on >= '{start_date}' AND created_on < '{end_date}'
        )
        SELECT
            `{DATASET_ID}.app.FORMAT_HEX`(a.account) as source,
            `{DATASET_ID}.app.FORMAT_HEX`(p.account) as target,
            CASE 
                WHEN a.type = 'TippingPostAction' THEN 'TIP'
                WHEN a.type = 'SimpleCollectAction' THEN 'COLLECT'
                ELSE 'UNKNOWN'
            END as type,
            'interact' as layer
        FROM `{DATASET_ID}.post.action_executed` as a
        JOIN `{DATASET_ID}.post.record` as p ON a.post_id = p.id
        JOIN TargetUsers t1 ON a.account = t1.account
        JOIN TargetUsers t2 ON p.account = t2.account
        WHERE a.timestamp >= '{start_date}' AND a.timestamp < '{end_date}'
          AND a.account != p.account
        """
        
        df_cq = self.client.query(query_cq).to_dataframe()
        df_rx = self.client.query(query_rx).to_dataframe()
        df_act = self.client.query(query_act).to_dataframe()
        
        # Filter out UNKNOWN types
        df_act = df_act[df_act['type'] != 'UNKNOWN']
        
        return pd.concat([df_cq, df_rx, df_act], ignore_index=True)
    
    def compute_coowner_edges(self, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """Compute co-owner edges (Layer 3) using vectorized operations."""
        # Filter nodes with valid owned_by
        df_subset = nodes_df[['profile_id', 'owned_by']].copy()
        df_subset = df_subset[df_subset['owned_by'].str.len() > 0]
        
        if df_subset.empty:
            return pd.DataFrame(columns=['source', 'target', 'type', 'layer'])
        
        # Self-join on owned_by
        df_co = pd.merge(
            df_subset, df_subset,
            on='owned_by',
            suffixes=('_source', '_target')
        )
        
        # Remove self-loops
        df_co = df_co[df_co['profile_id_source'] != df_co['profile_id_target']]
        
        # Rename columns
        df_co = df_co.rename(columns={
            'profile_id_source': 'source',
            'profile_id_target': 'target'
        })
        
        df_co['type'] = 'CO-OWNER'
        df_co['layer'] = 'co_owner'
        
        return df_co[['source', 'target', 'type', 'layer']]
    
    def compute_similarity_edges(
        self, 
        nodes_df: pd.DataFrame,
        max_nodes: int = 1000
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compute similarity edges (Layer 4).
        
        Returns:
            Tuple of (edges_df, warnings_list)
        """
        warnings = []
        edges = []
        
        n_nodes = len(nodes_df)
        if n_nodes > max_nodes:
            warnings.append(f"Node count ({n_nodes}) exceeds {max_nodes}. Similarity calculation skipped.")
            return pd.DataFrame(columns=['source', 'target', 'type', 'layer']), warnings
        
        if n_nodes < 2:
            return pd.DataFrame(columns=['source', 'target', 'type', 'layer']), warnings
        
        # 1. Same Avatar (URL matching)
        df_valid_avatar = nodes_df[nodes_df['picture_url'].str.len() > 5]
        if len(df_valid_avatar) > 1:
            groups = df_valid_avatar.groupby('picture_url').apply(
                lambda x: x.index.tolist() if len(x) > 1 else []
            )
            for indices in groups:
                if len(indices) > 1:
                    for i, j in combinations(indices, 2):
                        edges.append({
                            'source': nodes_df.loc[i, 'profile_id'],
                            'target': nodes_df.loc[j, 'profile_id'],
                            'type': 'SAME_AVATAR',
                            'layer': 'similarity'
                        })
        
        # 2. Close Creation Time (within 5 seconds)
        df_sorted = nodes_df.sort_values('created_on').reset_index(drop=True)
        times = df_sorted['created_on'].values
        window = pd.Timedelta(seconds=5)
        
        for i in range(len(df_sorted)):
            for j in range(i + 1, len(df_sorted)):
                time_diff = times[j] - times[i]
                if time_diff > window:
                    break
                edges.append({
                    'source': df_sorted.loc[i, 'profile_id'],
                    'target': df_sorted.loc[j, 'profile_id'],
                    'type': 'CLOSE_CREATION_TIME',
                    'layer': 'similarity'
                })
        
        # 3. Fuzzy Handle (> 90% similarity)
        if RAPIDFUZZ_AVAILABLE:
            handles = nodes_df['handle'].astype(str).tolist()
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    h1, h2 = handles[i], handles[j]
                    if len(h1) < 4 or len(h2) < 4:
                        continue
                    score = fuzz.ratio(h1, h2)
                    if score >= 90:
                        edges.append({
                            'source': nodes_df.iloc[i]['profile_id'],
                            'target': nodes_df.iloc[j]['profile_id'],
                            'type': 'FUZZY_HANDLE',
                            'layer': 'similarity'
                        })
        else:
            warnings.append("rapidfuzz not available. FUZZY_HANDLE skipped.")
        
        # 4. Similar Bio (SBERT cosine > 0.85)
        if SBERT_AVAILABLE and self.sbert_model is not None:
            valid_bio_mask = nodes_df['bio'].str.len() > 10
            df_valid_bio = nodes_df[valid_bio_mask]
            
            if len(df_valid_bio) > 1:
                bios = df_valid_bio['bio'].tolist()
                embeddings = self.sbert_model.encode(bios, show_progress_bar=False)
                sim_matrix = cosine_similarity(embeddings)
                
                valid_indices = df_valid_bio.index.tolist()
                for i in range(len(valid_indices)):
                    for j in range(i + 1, len(valid_indices)):
                        if sim_matrix[i][j] >= 0.85:
                            edges.append({
                                'source': nodes_df.loc[valid_indices[i], 'profile_id'],
                                'target': nodes_df.loc[valid_indices[j], 'profile_id'],
                                'type': 'SIM_BIO',
                                'layer': 'similarity'
                            })
        else:
            warnings.append("SentenceTransformer not available. SIM_BIO skipped.")
        
        if not edges:
            return pd.DataFrame(columns=['source', 'target', 'type', 'layer']), warnings
        
        return pd.DataFrame(edges), warnings
    
    def fetch_and_process_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Main entry point: Fetch all data and construct the 4-layer graph.
        
        Returns:
            Dictionary with:
            - nodes_df: DataFrame with ID, Metadata, Features
            - edges_df: DataFrame with source, target, type, layer
            - warnings: List of warning messages
        """
        # Format dates for SQL
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        
        warnings = []
        
        # 1. Fetch nodes
        nodes_df = self.fetch_nodes(start_str, end_str)
        if nodes_df.empty:
            return {
                'nodes_df': pd.DataFrame(),
                'features_df': pd.DataFrame(),
                'edges_df': pd.DataFrame(),
                'warnings': ['No nodes found in the specified date range.']
            }
        
        # 2. Fetch features
        features_df = self.fetch_node_features(start_str, end_str)
        
        # Merge nodes with features
        nodes_df = pd.merge(nodes_df, features_df, on='profile_id', how='left')
        stat_cols = [
            'total_tips', 'total_posts', 'total_quotes', 'total_reacted',
            'total_reactions', 'total_reposts', 'total_collects', 'total_comments',
            'total_followers', 'total_following'
        ]
        for col in stat_cols:
            if col not in nodes_df.columns:
                nodes_df[col] = 0
        nodes_df[stat_cols] = nodes_df[stat_cols].fillna(0).astype(int)
        
        # 3. Fetch edges (L1 Follow)
        edges_follow = self.fetch_follow_edges(start_str, end_str)
        
        # 4. Fetch edges (L2 Interact)
        edges_interact = self.fetch_interaction_edges(start_str, end_str)
        
        # 5. Compute edges (L3 Co-owner)
        edges_coowner = self.compute_coowner_edges(nodes_df)
        
        # 6. Compute edges (L4 Similarity)
        edges_similarity, sim_warnings = self.compute_similarity_edges(nodes_df)
        warnings.extend(sim_warnings)
        
        # Combine all edges
        all_edges = [edges_follow, edges_interact, edges_coowner, edges_similarity]
        edges_df = pd.concat([e for e in all_edges if not e.empty], ignore_index=True)
        
        # Create features DataFrame
        features_df = nodes_df[['profile_id'] + stat_cols].copy()
        
        return {
            'nodes_df': nodes_df,
            'features_df': features_df,
            'edges_df': edges_df,
            'warnings': warnings
        }


def fetch_and_process_data(start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Convenience function for fetching and processing data."""
    loader = DataLoader()
    return loader.fetch_and_process_data(start_date, end_date)
