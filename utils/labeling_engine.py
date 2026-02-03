"""Labeling engine for semi-supervised cluster labeling.

This module applies heuristic rules to cluster profiles to generate
pseudo-labels for supervised training.

Logic adapted from colab-code/labeling.py (Rule-based classifier, lines 120-172)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class LabelingThresholds:
    """Configuration thresholds for labeling rules."""
    pct_co_owner: float = 0.05
    pct_fuzzy_handle: float = 0.50
    pct_similarity: float = 0.60
    std_creation_hours: float = 2.0
    pct_social: float = 0.20
    co_owner_avg_trust: float = 25.0
    industrial_avg_trust: float = 20.0


@dataclass 
class ClusterProfile:
    """Profile statistics for a single cluster."""
    cluster_id: int
    size: int
    avg_trust: float
    std_creation_hours: float
    pct_co_owner: float
    pct_fuzzy_handle: float
    pct_similarity: float
    pct_social: float


@dataclass
class LabelingResult:
    """Container for labeling results."""
    cluster_id: int
    label: str
    rule_applied: str
    reason: str
    profile: ClusterProfile


class LabelingEngine:
    """
    Engine for applying rule-based labeling to cluster profiles.
    
    Logic adapted from colab-code/labeling.py (lines 64-172)
    """
    
    def __init__(self, thresholds: Optional[LabelingThresholds] = None):
        self.thresholds = thresholds or LabelingThresholds()
        
        # Edge type categories
        self.co_owner_types = ['CO-OWNER']
        self.fuzzy_handle_types = ['FUZZY_HANDLE']
        self.similarity_types = ['SAME_AVATAR', 'SIM_BIO', 'CLOSE_CREATION_TIME', 'FUZZY_HANDLE']
        self.social_types = ['FOLLOW', 'UPVOTE', 'COMMENT', 'COLLECT', 'QUOTE', 'MIRROR']
    
    def calculate_cluster_stats(
        self,
        cluster_id: int,
        nodes_in_cluster: pd.DataFrame,
        edges_in_cluster: pd.DataFrame
    ) -> ClusterProfile:
        """
        Calculate statistics for a single cluster.
        
        Logic from colab-code/labeling.py (lines 66-111)
        """
        total_edges = max(len(edges_in_cluster), 1)  # Avoid division by zero
        
        # Average trust score
        avg_trust = nodes_in_cluster['trust_score'].mean() if 'trust_score' in nodes_in_cluster.columns else 0
        
        # Standard deviation of creation time (in hours)
        std_hours = 0
        if 'created_on' in nodes_in_cluster.columns and len(nodes_in_cluster) > 1:
            try:
                created_times = pd.to_datetime(nodes_in_cluster['created_on'])
                std_seconds = created_times.std().total_seconds()
                std_hours = std_seconds / 3600
            except Exception:
                std_hours = float('inf')  # Unknown = assume not suspicious
        
        # Edge type percentages
        type_counts = edges_in_cluster['type'].value_counts() if 'type' in edges_in_cluster.columns else pd.Series()
        
        def get_pct(type_list):
            count = type_counts[type_counts.index.isin(type_list)].sum()
            return count / total_edges
        
        pct_co_owner = get_pct(self.co_owner_types)
        pct_fuzzy_handle = get_pct(self.fuzzy_handle_types)
        pct_similarity = get_pct(self.similarity_types)
        pct_social = get_pct(self.social_types)
        
        return ClusterProfile(
            cluster_id=cluster_id,
            size=len(nodes_in_cluster),
            avg_trust=avg_trust,
            std_creation_hours=std_hours,
            pct_co_owner=pct_co_owner,
            pct_fuzzy_handle=pct_fuzzy_handle,
            pct_similarity=pct_similarity,
            pct_social=pct_social
        )
    
    def apply_rules(self, profile: ClusterProfile) -> LabelingResult:
        """
        Apply rule-based classifier to a cluster profile.
        
        Logic from colab-code/labeling.py (lines 131-172)
        
        Priority order:
        1. Co-owner Ring (strongest signal)
        2. Name Pattern Abuse (fuzzy handle spam)
        3. Industrial Bot Farm (batch creation + low social + low trust)
        4. Default: Organic Community
        """
        reasons = []
        label = "NON-SYBIL"
        rule_applied = "Default"
        
        t = self.thresholds  # Shorthand
        
        # Priority 1: Co-owner Ring
        if profile.pct_co_owner > t.pct_co_owner:
            label = "SYBIL"
            rule_applied = "Priority 1: Co-owner Ring"
            reasons.append(f"Co-owner edges > {t.pct_co_owner * 100:.1f}% ({profile.pct_co_owner:.1%})")
            return LabelingResult(
                cluster_id=profile.cluster_id,
                label=label,
                rule_applied=rule_applied,
                reason="; ".join(reasons),
                profile=profile
            )
        
        # Priority 2: Name Pattern Abuse
        if (profile.pct_fuzzy_handle == 1 or 
            (profile.pct_fuzzy_handle >= t.pct_fuzzy_handle and 
             profile.avg_trust <= t.co_owner_avg_trust)):
            label = "SYBIL"
            rule_applied = "Priority 2: Name Pattern Abuse"
            reasons.append(f"Fuzzy Handle edges >= {t.pct_fuzzy_handle * 100:.1f}% ({profile.pct_fuzzy_handle:.1%})")
            reasons.append(f"Low Trust ({profile.avg_trust:.2f} <= {t.co_owner_avg_trust})")
            return LabelingResult(
                cluster_id=profile.cluster_id,
                label=label,
                rule_applied=rule_applied,
                reason="; ".join(reasons),
                profile=profile
            )
        
        # Priority 3: Industrial Bot Farm
        cond_batch = (profile.pct_similarity >= t.pct_similarity or 
                      profile.std_creation_hours < t.std_creation_hours)
        cond_social = profile.pct_social <= t.pct_social
        cond_trust = profile.avg_trust <= t.industrial_avg_trust
        
        if cond_batch and cond_social and cond_trust:
            label = "SYBIL"
            rule_applied = "Priority 3: Industrial Bot Farm"
            if profile.pct_similarity >= t.pct_similarity:
                reasons.append(f"High Similarity ({profile.pct_similarity:.1%})")
            if profile.std_creation_hours < t.std_creation_hours:
                reasons.append(f"Batch Creation (< {t.std_creation_hours}h std)")
            reasons.append("Low Social Activity")
            reasons.append("Low Trust Score")
            return LabelingResult(
                cluster_id=profile.cluster_id,
                label=label,
                rule_applied=rule_applied,
                reason="; ".join(reasons),
                profile=profile
            )
        
        # Default: Organic Community
        return LabelingResult(
            cluster_id=profile.cluster_id,
            label="NON-SYBIL",
            rule_applied="Default: Organic Community",
            reason="Safe Profile",
            profile=profile
        )
    
    def profile_clusters(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        cluster_labels: np.ndarray
    ) -> List[ClusterProfile]:
        """
        Generate profiles for all clusters.
        
        Args:
            nodes_df: Node metadata with cluster assignments
            edges_df: Edge list with type information
            cluster_labels: Cluster assignment for each node
            
        Returns:
            List of ClusterProfile objects
        """
        # Add cluster labels to nodes
        nodes_df = nodes_df.copy()
        nodes_df['cluster_label'] = cluster_labels
        
        # Create node to cluster mapping
        node_to_cluster = dict(zip(nodes_df['profile_id'], cluster_labels))
        
        # Add cluster info to edges
        edges_df = edges_df.copy()
        edges_df['src_cluster'] = edges_df['source'].map(node_to_cluster)
        edges_df['dst_cluster'] = edges_df['target'].map(node_to_cluster)
        
        # Filter internal edges (both nodes in same cluster)
        internal_edges = edges_df[edges_df['src_cluster'] == edges_df['dst_cluster']]
        
        # Calculate stats for each cluster
        unique_clusters = sorted(set(cluster_labels))
        profiles = []
        
        for cluster_id in unique_clusters:
            nodes_in_cluster = nodes_df[nodes_df['cluster_label'] == cluster_id]
            edges_in_cluster = internal_edges[internal_edges['src_cluster'] == cluster_id]
            
            profile = self.calculate_cluster_stats(
                cluster_id, nodes_in_cluster, edges_in_cluster
            )
            profiles.append(profile)
        
        return profiles
    
    def label_clusters(
        self,
        profiles: List[ClusterProfile]
    ) -> Tuple[List[LabelingResult], pd.DataFrame]:
        """
        Apply labeling rules to all cluster profiles.
        
        Args:
            profiles: List of ClusterProfile objects
            
        Returns:
            Tuple of (labeling results, summary DataFrame)
        """
        results = []
        
        for profile in profiles:
            result = self.apply_rules(profile)
            results.append(result)
        
        # Create summary DataFrame
        summary_data = []
        for r in results:
            summary_data.append({
                'cluster_id': r.cluster_id,
                'size': r.profile.size,
                'label': r.label,
                'rule_applied': r.rule_applied,
                'reason': r.reason,
                'avg_trust': r.profile.avg_trust,
                'std_creation_hours': r.profile.std_creation_hours,
                'pct_co_owner': r.profile.pct_co_owner,
                'pct_fuzzy_handle': r.profile.pct_fuzzy_handle,
                'pct_similarity': r.profile.pct_similarity,
                'pct_social': r.profile.pct_social
            })
        
        summary_df = pd.DataFrame(summary_data)
        return results, summary_df
    
    def generate_node_labels(
        self,
        nodes_df: pd.DataFrame,
        cluster_labels: np.ndarray,
        labeling_results: List[LabelingResult]
    ) -> np.ndarray:
        """
        Generate node-level labels from cluster labels.
        
        Args:
            nodes_df: Node DataFrame
            cluster_labels: Cluster assignment per node
            labeling_results: Labeling results per cluster
            
        Returns:
            Array of labels (0=NON-SYBIL, 1=SYBIL) per node
        """
        # Create cluster -> label mapping
        cluster_to_label = {}
        for result in labeling_results:
            cluster_to_label[result.cluster_id] = 1 if result.label == "SYBIL" else 0
        
        # Map to nodes
        node_labels = np.array([
            cluster_to_label.get(c, 0) for c in cluster_labels
        ])
        
        return node_labels


def create_labeling_summary(
    labeling_results: List[LabelingResult]
) -> Dict[str, Any]:
    """
    Create summary statistics for labeling results.
    """
    total_clusters = len(labeling_results)
    sybil_clusters = sum(1 for r in labeling_results if r.label == "SYBIL")
    nonsybil_clusters = total_clusters - sybil_clusters
    
    total_nodes = sum(r.profile.size for r in labeling_results)
    sybil_nodes = sum(r.profile.size for r in labeling_results if r.label == "SYBIL")
    nonsybil_nodes = total_nodes - sybil_nodes
    
    # Count by rule
    rules_count = {}
    for r in labeling_results:
        rule = r.rule_applied
        if rule not in rules_count:
            rules_count[rule] = {'clusters': 0, 'nodes': 0}
        rules_count[rule]['clusters'] += 1
        rules_count[rule]['nodes'] += r.profile.size
    
    return {
        'total_clusters': total_clusters,
        'sybil_clusters': sybil_clusters,
        'nonsybil_clusters': nonsybil_clusters,
        'total_nodes': total_nodes,
        'sybil_nodes': sybil_nodes,
        'nonsybil_nodes': nonsybil_nodes,
        'sybil_ratio': sybil_nodes / total_nodes if total_nodes > 0 else 0,
        'rules_breakdown': rules_count
    }
