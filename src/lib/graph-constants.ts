// FIX 2: Ensure LABEL_COLORS keys match exactly the RiskClassification enum values
// from the API: "BENIGN" | "LOW_RISK" | "HIGH_RISK" | "MALICIOUS"
export const LABEL_COLORS: Record<string, string> = {
  BENIGN: "#00f2ff", // Cyan  - clean/safe
  LOW_RISK: "#4ade80", // Green - mild concern
  HIGH_RISK: "#fb923c", // Orange - serious
  MALICIOUS: "#ef4444", // Red   - dangerous
  UNKNOWN: "#94a3b8", // Slate - unknown/unclassified
};

// FIX 5: Edge types mapped to color. Grouped by relation layer.
export const RELATION_COLORS: Record<string, string> = {
  // Follow Layer (directed) - Blue
  FOLLOW: "#3b82f6",

  // Interact Layer (directed) - Green/Teal
  UPVOTE: "#10b981",
  REACTION: "#10b981",
  COMMENT: "#10b981",
  QUOTE: "#10b981",
  MIRROR: "#06b6d4",
  COLLECT: "#06b6d4",
  TIP: "#10b981",
  INTERACT: "#10b981",

  // Co-Owner Layer (undirected) - Orange/Amber
  "CO-OWNER": "#f97316",

  // Similarity Layer (undirected) - Purple/Violet
  SAME_AVATAR: "#a855f7",
  FUZZY_HANDLE: "#a855f7",
  SIM_BIO: "#a855f7",
  CLOSE_CREATION_TIME: "#8b5cf6",
  SIMILARITY: "#a855f7",

  UNKNOWN: "#64748b",
};

export const RELATION_GROUPS = [
  { label: "Co-Owner (undirected)", type: "CO-OWNER" },
  { label: "Follow (directed)", type: "FOLLOW" },
  { label: "Interact (directed)", type: "INTERACT" },
  { label: "Similarity (undirected)", type: "SIMILARITY" },
];

export const LABEL_GROUPS = [
  { label: "Benign", key: "BENIGN" },
  { label: "Low Risk", key: "LOW_RISK" },
  { label: "High Risk", key: "HIGH_RISK" },
  { label: "Malicious", key: "MALICIOUS" },
];

export const MIN_LINK_WIDTH = 0.5;
export const MAX_LINK_WIDTH = 5;
export const DEFAULT_LINK_WIDTH = 1.5;

// FIX 5: Edge direction classification
export const DIRECTED_EDGE_LAYERS = new Set(["Follow", "Interact"]);
export const DIRECTED_EDGE_TYPES = new Set([
  "FOLLOW",
  "UPVOTE",
  "REACTION",
  "COMMENT",
  "QUOTE",
  "MIRROR",
  "COLLECT",
  "TIP",
]);
