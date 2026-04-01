import { SybilEdge } from "@/types/api";

// ─── Risk label colors ───
export const LABEL_COLORS: Record<string, string> = {
  BENIGN: "#00f2ff", // cyan
  LOW_RISK: "#4ade80", // green
  HIGH_RISK: "#fb923c", // orange
  MALICIOUS: "#ef4444", // red
  UNKNOWN: "#94a3b8", // slate
};

// ─── Edge type colors by relation layer ───
export const RELATION_COLORS: Record<string, string> = {
  // Follow layer (directed) — blue
  FOLLOW: "#3b82f6",
  FOLLOW_REV: "#3b82f6",
  // Interact layer (directed) — emerald / cyan
  UPVOTE: "#10b981",
  UPVOTE_REV: "#10b981",
  REACTION: "#10b981",
  REACTION_REV: "#10b981",
  COMMENT: "#10b981",
  COMMENT_REV: "#10b981",
  QUOTE: "#10b981",
  QUOTE_REV: "#10b981",
  MIRROR: "#06b6d4",
  MIRROR_REV: "#06b6d4",
  COLLECT: "#06b6d4",
  COLLECT_REV: "#06b6d4",
  TIP: "#10b981",
  TIP_REV: "#10b981",
  INTERACT: "#10b981",
  // Co-Owner layer (undirected) — orange
  "CO-OWNER": "#f97316",
  // Similarity layer (undirected) — violet
  SAME_AVATAR: "#a855f7",
  FUZZY_HANDLE: "#8b5cf6",
  SIM_BIO: "#a855f7",
  CLOSE_CREATION_TIME: "#7c3aed",
  SIMILARITY: "#a855f7",
  UNKNOWN: "#475569",
};

// ─── Which edge_type values are DIRECTED (Follow + Interact layers) ───
export const DIRECTED_EDGE_TYPES = new Set([
  "FOLLOW",
  "FOLLOW_REV",
  "UPVOTE",
  "UPVOTE_REV",
  "REACTION",
  "REACTION_REV",
  "COMMENT",
  "COMMENT_REV",
  "QUOTE",
  "QUOTE_REV",
  "MIRROR",
  "MIRROR_REV",
  "COLLECT",
  "COLLECT_REV",
  "TIP",
  "TIP_REV",
]);

// ─── Layer grouping for legend ───
export const EDGE_LAYERS: {
  key: string;
  label: string;
  color: string;
  directed: boolean;
  types: string[];
}[] = [
  {
    key: "CO-OWNER",
    label: "Co-Owner",
    color: RELATION_COLORS["CO-OWNER"],
    directed: false,
    types: ["CO-OWNER"],
  },
  {
    key: "FOLLOW",
    label: "Follow",
    color: RELATION_COLORS["FOLLOW"],
    directed: true,
    types: ["FOLLOW", "FOLLOW_REV"],
  },
  {
    key: "INTERACT",
    label: "Interact",
    color: RELATION_COLORS["INTERACT"],
    directed: true,
    types: [
      "UPVOTE",
      "UPVOTE_REV",
      "REACTION",
      "REACTION_REV",
      "COMMENT",
      "COMMENT_REV",
      "QUOTE",
      "QUOTE_REV",
      "MIRROR",
      "MIRROR_REV",
      "COLLECT",
      "COLLECT_REV",
      "TIP",
      "TIP_REV",
    ],
  },
  {
    key: "SIMILARITY",
    label: "Similarity",
    color: RELATION_COLORS["SIMILARITY"],
    directed: false,
    types: ["SAME_AVATAR", "FUZZY_HANDLE", "SIM_BIO", "CLOSE_CREATION_TIME"],
  },
];

// ─── Legacy compat exports ───
export const RELATION_GROUPS = EDGE_LAYERS.map(({ key, label }) => ({
  type: key,
  label,
}));

export const LABEL_GROUPS = [
  { label: "Benign", key: "BENIGN" },
  { label: "Low Risk", key: "LOW_RISK" },
  { label: "High Risk", key: "HIGH_RISK" },
  { label: "Malicious", key: "MALICIOUS" },
];

export const MIN_LINK_WIDTH = 0.5;
export const MAX_LINK_WIDTH = 5;
export const DEFAULT_LINK_WIDTH = 1.5;

// ─── Compute edge counts per layer from a links array ───
export function computeEdgeCounts(links: SybilEdge[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const link of links) {
    const et = link.edge_type || "UNKNOWN";
    let layerKey = "UNKNOWN";
    for (const layer of EDGE_LAYERS) {
      if (layer.types.includes(et)) {
        layerKey = layer.key;
        break;
      }
    }
    counts[layerKey] = (counts[layerKey] || 0) + 1;
  }
  return counts;
}
