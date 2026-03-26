export type RiskClassification =
  | "BENIGN"
  | "LOW_RISK"
  | "HIGH_RISK"
  | "MALICIOUS";

export interface ProfileInfo {
  id: string;
  handle: string;
  picture_url: string;
  owned_by: string;
}

export interface InferenceReasoning {
  type: string; // e.g., "SIM_BIO", "PATTERN_XYZ"
  description: string;
  severity: "low" | "medium" | "high" | "critical";
}

export interface Analysis {
  predict_label: RiskClassification;
  predict_proba: Record<RiskClassification, number>;
  reasoning: string[];
}

export interface SybilNode {
  id: string;
  risk_label: RiskClassification;
  trust_score: number;
  is_sybil: boolean;
  cluster_id: number;
  risk_score: number;
  x?: number;
  y?: number;
  z?: number;
  attributes: {
    follower_count: number;
    post_count: number;
    trust_score: number;
    reasons: string[];
    handle?: string;
    following_count?: number;
    account_age?: string;
    picture_url?: string;
    total_reposts?: number;
    [key: string]: string | number | boolean | undefined;
  };
}

export interface SybilEdge {
  source: string;
  target: string;
  edge_type: "comment" | "follow" | "upvote" | "transfer" | string;
  weight: number;
  gat_attention?: number;
}

export interface LocalGraph {
  nodes: SybilNode[];
  links: SybilEdge[];
}

export interface InspectorResponse {
  profile_info: ProfileInfo;
  analysis: Analysis;
  local_graph: LocalGraph;
}

export interface TimeRange {
  start_date: string; // YYYY-MM-DD
  end_date: string; // YYYY-MM-DD
}

export interface DiscoveryHyperparameter {
  max_epochs?: number;
  patience?: number;
  learning_rate?: number;
}

export interface DiscoveryStartRequest {
  time_range: TimeRange;
  max_nodes: number;
  hyperparameters?: DiscoveryHyperparameter;
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
    nodes: SybilNode[];
    links: SybilEdge[];
    cluster_count: number;
    num_nodes: number;
    num_edges: number;
    start_date: string;
    end_date: string;
  } | null;
  message: string | null;
}
