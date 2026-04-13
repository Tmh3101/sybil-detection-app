export type RiskClassification =
  | "BENIGN"
  | "LOW_RISK"
  | "HIGH_RISK"
  | "MALICIOUS";

export interface SybilNode {
  id: string;
  risk_label: RiskClassification;
  risk_score: number;
  trust_score: number;
  cluster_id?: number;
  x?: number;
  y?: number;
  z?: number;
  attributes: {
    handle?: string;
    bio?: string;
    created_on?: string;
    days_active?: number;
    total_tips?: number;
    total_posts?: number;
    total_quotes?: number;
    total_reacted?: number;
    total_reactions?: number;
    total_reposts?: number;
    total_collects?: number;
    total_comments?: number;
    total_followers?: number;
    total_following?: number;
    follower_count?: number;
    following_count?: number;
    post_count?: number;
    picture_url?: string;
    owned_by?: string;
    reasons?: string[];
    reason?: string;
    [key: string]: string | number | boolean | string[] | undefined;
  };
}

export interface SybilEdge {
  id?: string;
  source: string;
  target: string;
  edge_type:
    | "FOLLOW"
    | "UPVOTE"
    | "REACTION"
    | "COMMENT"
    | "QUOTE"
    | "MIRROR"
    | "COLLECT"
    | "CO-OWNER"
    | "SAME_AVATAR"
    | "FUZZY_HANDLE"
    | "SIM_BIO"
    | "CLOSE_CREATION_TIME"
    | "SIMILARITY"
    | "FOLLOW_REV"
    | "UPVOTE_REV"
    | "REACTION_REV"
    | "COMMENT_REV"
    | "QUOTE_REV"
    | "MIRROR_REV"
    | "COLLECT_REV"
    | "UNKNOWN";
  weight: number;
  gat_attention?: number;
  violations?: string[];
}

export interface Analysis {
  predict_label: RiskClassification;
  predict_proba: Record<RiskClassification, number>;
  reasoning: string[];
}

export interface InspectorResponse {
  profile_info: {
    id: string;
    handle: string;
    picture_url?: string;
    owned_by: string;
  };
  analysis: Analysis;
  local_graph: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
}

export interface TimeRange {
  start_date: string;
  end_date: string;
}

export interface Hyperparameters {
  max_epochs?: number;
  patience?: number;
  learning_rate?: number;
}

export interface DiscoveryStartRequest {
  time_range: TimeRange;
  max_nodes?: number;
  hyperparameters?: Hyperparameters;
}

export interface DiscoveryStartResponse {
  task_id: string;
}

export interface DiscoveryStatusResponse {
  task_id: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  progress: number;
  current_step: string;
  graph_data: {
    cluster_count: number;
    num_nodes?: number;
    num_edges?: number;
    nodes: SybilNode[];
    links: SybilEdge[];
  } | null;
  message: string | null;
}

// Statistics Types
export interface EdgeDistributionItem {
  layer: string;
  count: number;
  percentage: number;
}

export interface NetworkOverview {
  total_nodes: number;
  total_edges: number;
  total_clusters: number;
  avg_cluster_size: number;
  edge_distribution: EdgeDistributionItem[];
}

export interface RiskDistributionItem {
  label: RiskClassification;
  count: number;
}

export interface RiskDistribution {
  distribution: RiskDistributionItem[];
}

export interface ClusterStats {
  total_clusters: number;
  avg_cluster_size: number;
  largest_cluster: number;
  smallest_cluster: number;
}

// History Types
export interface InspectorHistoryItem {
  id: number;
  timestamp: string;
  target_address: string;
  predict_label: RiskClassification;
  confidence_score: number;
  depth_filter: number;
}

export type InspectorHistoryResponse = InspectorHistoryItem[];

export interface DiscoveryHistoryItem {
  id: number;
  task_id: string;
  timestamp: string;
  start_date: string;
  end_date: string;
  cluster_count: number;
  node_count: number;
  edge_count: number;
  status: "COMPLETED" | "FAILED" | "PENDING";
}

export type DiscoveryHistoryResponse = DiscoveryHistoryItem[];
