export const LABEL_COLORS: Record<string, string> = {
  BENIGN: "#00f2ff",
  LOW_RISK: "#4ade80",
  HIGH_RISK: "#fb923c",
  MALICIOUS: "#ef4444",
  UNKNOWN: "#94a3b8",
};

export const RELATION_COLORS: Record<string, string> = {
  // Follow Layer
  FOLLOW: "#3b82f6",

  // Interact Layer
  UPVOTE: "#10b981",
  REACTION: "#10b981",
  COMMENT: "#10b981",
  QUOTE: "#10b981",
  MIRROR: "#10b981",
  COLLECT: "#10b981",
  TIP: "#10b981",
  INTERACT: "#10b981",

  // Co-Owner Layer
  "CO-OWNER": "#f97316",

  // Similarity Layer
  SAME_AVATAR: "#a855f7",
  FUZZY_HANDLE: "#a855f7",
  SIM_BIO: "#a855f7",
  CLOSE_CREATION_TIME: "#a855f7",
  SIMILARITY: "#a855f7",

  UNKNOWN: "#64748b",
};

export const RELATION_GROUPS = [
  { label: "Co-Owner", type: "CO-OWNER" },
  { label: "Follow", type: "FOLLOW" },
  { label: "Interact", type: "INTERACT" },
  { label: "Similarity", type: "SIMILARITY" },
];

export const LABEL_GROUPS = [
  { label: "Benign", key: "BENIGN" },
  { label: "Low Risk", key: "LOW_RISK" },
  { label: "High Risk", key: "HIGH_RISK" },
  { label: "Malicious", key: "MALICIOUS" },
];
