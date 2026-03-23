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
  sybil_probability: number;
  classification: "BENIGN" | "WARNING" | "SYBIL";
  reasoning: string[];
}

export interface SybilNode {
  id: string;
  label: string;
  trust_score: number;
  is_sybil: boolean;
  cluster_id?: number;
  is_high_risk?: boolean;
  x?: number;
  y?: number;
  z?: number;
  attributes: {
    follower_count?: number;
    following_count?: number;
    account_age?: string;
    [key: string]: string | number | boolean | undefined;
  };
}

export interface SybilEdge {
  source: string;
  target: string;
  type: "comment" | "follow" | "upvote" | "transfer";
  weight: number;
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

export interface DiscoveryStartRequest {
  time_range: TimeRange;
  max_nodes: number;
}

export interface DiscoveryStartResponse {
  task_id: string;
}

export interface DiscoveryStatusResponse {
  task_id: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  progress: number;
  current_step: string;
  graph_data: { nodes: SybilNode[]; links: SybilEdge[] } | null;
  message: string | null;
}
