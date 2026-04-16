import { useMemo } from "react";
import { SybilNode, SybilEdge } from "@/types/api";
import { NodeObject } from "react-force-graph-2d";
import { LABEL_COLORS, LIGHT_LABEL_COLORS } from "@/lib/graph-constants";
import { useThemeStore } from "@/store/theme-store";

export interface AggregatedLink extends Omit<SybilEdge, "source" | "target"> {
  source: string | NodeObject<SybilNode>;
  target: string | NodeObject<SybilNode>;
  aggregated_weight?: number;
  gat_attention?: number;
  multiLinkIndex?: number;
  multiLinkCount?: number;
  is_merged_multiple?: boolean;
}

interface ProcessedGraphData {
  nodes: SybilNode[];
  links: AggregatedLink[];
}

interface GraphProcessorOptions {
  aggregateEdges?: boolean;
  mergeEdges?: boolean;
  targetId?: string;
}

export function useGraphProcessor(
  graphData: { nodes: SybilNode[]; links: SybilEdge[] },
  options: GraphProcessorOptions = {}
): ProcessedGraphData {
  const { aggregateEdges = true, mergeEdges = false, targetId } = options;
  const { theme } = useThemeStore();
  const isDark = theme === "dark";

  return useMemo(() => {
    // 1. Process nodes — inject __color and __isTarget for reliable canvas access
    const nodes = graphData.nodes.map((n) => {
      const isTarget = !!(targetId && String(n.id) === String(targetId));
      const riskLabel = String(n.risk_label || "UNKNOWN")
        .trim()
        .toUpperCase();

      const palette = isDark ? LABEL_COLORS : LIGHT_LABEL_COLORS;
      const nodeColor = palette[riskLabel] || palette.UNKNOWN;

      const enriched = {
        ...n,
        risk_label: riskLabel as SybilNode["risk_label"],
        fx: isTarget ? 0 : undefined,
        fy: isTarget ? 0 : undefined,
        // These two are the KEY FIX — set before d3 processes nodes
        __color: nodeColor,
        __isTarget: isTarget,
      };

      return enriched as SybilNode;
    });

    if (!aggregateEdges) {
      const processedLinks = graphData.links.map((link, index) => {
        let sId =
          typeof link.source === "object"
            ? (link.source as NodeObject<SybilNode>).id
            : (link.source as string);
        let tId =
          typeof link.target === "object"
            ? (link.target as NodeObject<SybilNode>).id
            : (link.target as string);

        const type = link.edge_type || "UNKNOWN";
        if (type.endsWith("_REV")) {
          const temp = sId;
          sId = tId;
          tId = temp;
        }

        return {
          ...link,
          id: link.id || `${sId}-${tId}-${type}-${index}`,
          source: sId as string,
          target: tId as string,
        } as AggregatedLink;
      });
      return { nodes, links: processedLinks };
    }

    // 2. Pre-calculate true GAT attention strictly from raw links where target is targetId
    const incomingAttentionMap = new Map<string, number>();
    if (targetId) {
      const tid = String(targetId).toLowerCase();
      graphData.links.forEach((link) => {
        const rawS = String(
          typeof link.source === "object"
            ? (link.source as NodeObject<SybilNode>).id
            : (link.source as string)
        ).toLowerCase();
        const rawT = String(
          typeof link.target === "object"
            ? (link.target as NodeObject<SybilNode>).id
            : (link.target as string)
        ).toLowerCase();

        if (rawT === tid) {
          incomingAttentionMap.set(
            rawS,
            (incomingAttentionMap.get(rawS) || 0) + (link.gat_attention || 0)
          );
        }
      });
    }

    // 3. Aggregate links by source-target-type
    const linkMap = new Map<string, AggregatedLink>();

    graphData.links.forEach((link) => {
      const rawS = String(
        typeof link.source === "object"
          ? (link.source as NodeObject<SybilNode>).id
          : (link.source as string)
      );
      const rawT = String(
        typeof link.target === "object"
          ? (link.target as NodeObject<SybilNode>).id
          : (link.target as string)
      );

      let sId = rawS;
      let tId = rawT;
      const type = link.edge_type || "UNKNOWN";

      // FIX: If it's a reverse edge, swap source and target so the arrow points correctly
      // in the visualization (towards the actual target of the relationship).
      if (type.endsWith("_REV")) {
        const temp = sId;
        sId = tId;
        tId = temp;
      }

      const key = mergeEdges ? `${sId}-${tId}` : `${sId}-${tId}-${type}`;

      if (linkMap.has(key)) {
        const existing = linkMap.get(key)!;
        existing.aggregated_weight =
          (existing.aggregated_weight || 0) + (link.weight || 1);

        if (mergeEdges) {
          if (
            targetId &&
            String(tId).toLowerCase() === String(targetId).toLowerCase()
          ) {
            existing.gat_attention =
              incomingAttentionMap.get(String(sId).toLowerCase()) || 0;
          } else {
            existing.gat_attention =
              (existing.gat_attention || 0) + (link.gat_attention || 0);
          }
          existing.is_merged_multiple = true;
        } else {
          // FIX: Do NOT use incomingAttentionMap here. Keep individual max/sum for parallel edges of the SAME exact type.
          existing.gat_attention = Math.max(
            existing.gat_attention || 0,
            link.gat_attention || 0
          );
        }
      } else {
        let initialAttention = link.gat_attention || 0;

        // FIX: Only use the total summed attention if we are actually merging edges.
        if (
          mergeEdges &&
          targetId &&
          String(tId).toLowerCase() === String(targetId).toLowerCase()
        ) {
          initialAttention =
            incomingAttentionMap.get(String(sId).toLowerCase()) || 0;
        }

        linkMap.set(key, {
          ...link,
          id: link.id || key,
          source: sId as string,
          target: tId as string,
          aggregated_weight: link.weight || 1,
          edge_type: type,
          gat_attention: initialAttention,
        } as AggregatedLink);
      }
    });

    const aggregatedLinks = Array.from(linkMap.values());

    // 3. Multi-link curvature indices
    const pairGroups: Record<string, AggregatedLink[]> = {};
    aggregatedLinks.forEach((link) => {
      const sId =
        typeof link.source === "object"
          ? (link.source as NodeObject<SybilNode>).id
          : link.source;
      const tId =
        typeof link.target === "object"
          ? (link.target as NodeObject<SybilNode>).id
          : link.target;
      const id = [sId as string, tId as string].sort().join("-");
      if (!pairGroups[id]) pairGroups[id] = [];
      pairGroups[id].push(link);
    });

    Object.values(pairGroups).forEach((group) => {
      const count = group.length;
      group.forEach((link, i) => {
        link.multiLinkIndex = i;
        link.multiLinkCount = count;
      });
    });

    return { nodes, links: aggregatedLinks };
  }, [graphData, aggregateEdges, mergeEdges, targetId, isDark]);
}
