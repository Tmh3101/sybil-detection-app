import { useMemo } from "react";
import { SybilNode, SybilEdge } from "@/types/api";
import { NodeObject } from "react-force-graph-2d";
import { LABEL_COLORS } from "@/lib/graph-constants";

export interface AggregatedLink extends Omit<SybilEdge, "source" | "target"> {
  source: string | NodeObject<SybilNode>;
  target: string | NodeObject<SybilNode>;
  aggregated_weight?: number;
  gat_attention?: number;
  multiLinkIndex?: number;
  multiLinkCount?: number;
}

interface ProcessedGraphData {
  nodes: SybilNode[];
  links: AggregatedLink[];
}

interface GraphProcessorOptions {
  aggregateEdges?: boolean;
  targetId?: string;
}

export function useGraphProcessor(
  graphData: { nodes: SybilNode[]; links: SybilEdge[] },
  options: GraphProcessorOptions = {}
): ProcessedGraphData {
  const { aggregateEdges = true, targetId } = options;

  return useMemo(() => {
    // 1. Process nodes — inject __color and __isTarget for reliable canvas access
    const nodes = graphData.nodes.map((n) => {
      const isTarget = !!(targetId && String(n.id) === String(targetId));
      const riskLabel = String(n.risk_label || "UNKNOWN")
        .trim()
        .toUpperCase();
      const nodeColor = LABEL_COLORS[riskLabel] || LABEL_COLORS.UNKNOWN;

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

    // 2. Aggregate links by source-target-type
    const linkMap = new Map<string, AggregatedLink>();

    graphData.links.forEach((link) => {
      let sId =
        typeof link.source === "object"
          ? (link.source as NodeObject<SybilNode>).id
          : (link.source as string);
      let tId =
        typeof link.target === "object"
          ? (link.target as NodeObject<SybilNode>).id
          : (link.target as string);

      const type = link.edge_type || "UNKNOWN";

      // FIX: If it's a reverse edge, swap source and target so the arrow points correctly
      // in the visualization (towards the actual target of the relationship).
      if (type.endsWith("_REV")) {
        const temp = sId;
        sId = tId;
        tId = temp;
      }

      const key = `${sId}-${tId}-${type}`;

      if (linkMap.has(key)) {
        const existing = linkMap.get(key)!;
        existing.aggregated_weight =
          (existing.aggregated_weight || 0) + (link.weight || 1);
        existing.gat_attention = Math.max(
          existing.gat_attention || 0,
          link.gat_attention || 0
        );
      } else {
        linkMap.set(key, {
          ...link,
          id: link.id || key,
          source: sId as string,
          target: tId as string,
          aggregated_weight: link.weight || 1,
          edge_type: type,
          gat_attention: link.gat_attention || 0,
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
  }, [graphData, aggregateEdges, targetId]);
}
