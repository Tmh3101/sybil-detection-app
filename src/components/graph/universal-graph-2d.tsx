"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
  LinkObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge, RiskClassification } from "@/types/api";
import { resolvePictureUrl } from "@/lib/utils";
import {
  LABEL_COLORS,
  RELATION_COLORS,
  MIN_LINK_WIDTH,
} from "@/lib/graph-constants";
import GraphLegend from "./graph-legend";
import { useGraphProcessor, AggregatedLink } from "@/hooks/use-graph-processor";

// Edge types that are directed (one-way)
const DIRECTED_EDGE_TYPES = new Set([
  "FOLLOW",
  "UPVOTE",
  "REACTION",
  "COMMENT",
  "QUOTE",
  "MIRROR",
  "COLLECT",
  "TIP",
]);

interface ExtendedNode extends SybilNode {
  __img?: HTMLImageElement;
}

export interface UniversalGraph2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
  mode: "EGO" | "CLUSTER";
  targetId?: string;
  risk_label?: RiskClassification;
}

const UniversalGraph2D: React.FC<UniversalGraph2DProps> = ({
  graphData,
  mode,
  targetId,
  risk_label,
}) => {
  const fgRef = useRef<
    ForceGraphMethods<ExtendedNode, AggregatedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [imagesLoaded, setImagesLoaded] = useState(0);
  const imgCache = useRef<Record<string, HTMLImageElement>>({});

  const processedData = useGraphProcessor(graphData, {
    targetId: mode === "EGO" ? targetId : undefined,
    aggregateEdges: true,
  });

  useEffect(() => {
    if (!containerRef.current) return;
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height });
      }
    });
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  useEffect(() => {
    if (!fgRef.current) return;

    if (mode === "EGO") {
      fgRef.current.d3Force("radial", d3.forceRadial(150, 0, 0));
      const charge = fgRef.current.d3Force("charge");
      if (charge) {
        (charge as d3.ForceManyBody<ExtendedNode>).strength(-200);
      }
    } else {
      fgRef.current.d3Force("x", d3.forceX(0).strength(0.05));
      fgRef.current.d3Force("y", d3.forceY(0).strength(0.05));
      fgRef.current.d3Force("center", d3.forceCenter(0, 0));
      fgRef.current.d3Force("charge", d3.forceManyBody().strength(-100));
      fgRef.current.d3Force("link")?.distance(30);
    }

    fgRef.current.d3ReheatSimulation();
  }, [processedData, mode, dimensions.width, dimensions.height]);

  const getTargetColor = useCallback(() => {
    if (risk_label === "MALICIOUS" || risk_label === "HIGH_RISK")
      return LABEL_COLORS["MALICIOUS"];
    return LABEL_COLORS["BENIGN"];
  }, [risk_label]);

  // FIX 1: Use String() comparison to avoid type mismatch between string|number
  const isTargetNode = useCallback(
    (node: NodeObject<ExtendedNode>) => {
      return mode === "EGO" && String(node.id) === String(targetId);
    },
    [mode, targetId]
  );

  // FIX 2: Ensure risk_label is read correctly - fallback chain
  const getNodeColor = useCallback((node: NodeObject<ExtendedNode>) => {
    // risk_label is top-level on SybilNode, but NodeObject can shadow it
    const label = (node as ExtendedNode).risk_label;
    return (label && LABEL_COLORS[label]) || LABEL_COLORS.UNKNOWN;
  }, []);

  const drawNode = useCallback(
    (
      node: NodeObject<ExtendedNode>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const isTarget = isTargetNode(node);
      const isMalicious = (node as ExtendedNode).risk_label === "MALICIOUS";
      const isHighRisk = (node as ExtendedNode).risk_label === "HIGH_RISK";

      let size = 5;
      if (mode === "EGO") {
        // FIX 6: Target node is significantly larger for clear highlighting
        size = isTarget ? 14 : 6;
      } else {
        size = isMalicious ? 8 : 5;
      }

      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const color = isTarget ? getTargetColor() : getNodeColor(node);

      // FIX 6: Multi-ring glow for target node
      if (isTarget) {
        // Outer glow ring 3
        ctx.beginPath();
        ctx.arc(x, y, size + 10, 0, 2 * Math.PI, false);
        ctx.fillStyle = `${color}15`;
        ctx.fill();
        // Outer glow ring 2
        ctx.beginPath();
        ctx.arc(x, y, size + 6, 0, 2 * Math.PI, false);
        ctx.fillStyle = `${color}25`;
        ctx.fill();
        // Inner glow ring
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, 2 * Math.PI, false);
        ctx.fillStyle = `${color}40`;
        ctx.fill();
      } else if (mode === "CLUSTER" && (isMalicious || isHighRisk)) {
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, 2 * Math.PI, false);
        ctx.fillStyle = isMalicious
          ? "rgba(239, 68, 68, 0.3)"
          : "rgba(251, 146, 60, 0.2)";
        ctx.fill();
      }

      // Avatar rendering
      const skipImages = mode === "CLUSTER" && processedData.nodes.length > 500;
      const rawImgUrl = (node as ExtendedNode).attributes?.picture_url;
      const safeUrl = rawImgUrl ? resolvePictureUrl(String(rawImgUrl)) : "";
      let img: HTMLImageElement | null = null;

      if (!skipImages && safeUrl) {
        if (imgCache.current[safeUrl]) {
          img = imgCache.current[safeUrl];
        } else {
          const newImg = new Image();
          newImg.crossOrigin = "anonymous"; // FIX 3: Enable CORS for canvas
          newImg.src = safeUrl;
          newImg.onload = () => {
            imgCache.current[safeUrl] = newImg;
            setImagesLoaded((prev) => prev + 1);
          };
          newImg.onerror = () => {
            // Mark as failed so we don't retry
            imgCache.current[safeUrl] = new Image(); // placeholder
          };
          imgCache.current[safeUrl] = newImg; // set before load to prevent duplicate requests
        }
      }

      // Draw node base
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.clip();

      if (img && img.complete && img.naturalWidth > 0) {
        try {
          ctx.drawImage(img, x - size, y - size, size * 2, size * 2);
        } catch {
          ctx.fillStyle = "#1e293b";
          ctx.fill();
        }
      } else {
        // Fallback: fill with a darker version of the node color
        ctx.fillStyle = isTarget ? "#0a0f1a" : "#1e293b";
        ctx.fill();
      }
      ctx.restore();

      // Border ring
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.strokeStyle = color;
      ctx.lineWidth = isTarget ? 2.5 : isMalicious || isHighRisk ? 2 : 1;
      ctx.stroke();

      // FIX 6: Cross marker on target node for extra visibility
      if (isTarget) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.6;
        // Dashed orbit circle
        ctx.beginPath();
        ctx.setLineDash([3, 3]);
        ctx.arc(x, y, size + 7, 0, 2 * Math.PI, false);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }

      // Labels
      const extNode = node as ExtendedNode;
      const label = extNode.attributes?.handle || String(node.id).slice(0, 8);
      if (mode === "EGO" && (isTarget || globalScale > 2)) {
        const fontSize = isTarget ? 11 / globalScale : 10 / globalScale;
        ctx.font = `${isTarget ? "bold " : ""}${fontSize}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(String(label), x, y + size + 2);
      } else if (mode === "CLUSTER" && globalScale > 3) {
        ctx.font = `${8 / globalScale}px monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(String(label).slice(0, 6), x, y + size + 1);
      }
    },
    [
      mode,
      isTargetNode,
      getNodeColor,
      getTargetColor,
      processedData.nodes.length,
    ]
  );

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[400px] w-full bg-[#050608]"
    >
      <ForceGraph2D
        key={`fg-${imagesLoaded}-${mode}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "always"}
        // FIX 5: Directed arrows for FOLLOW / INTERACT layer edges
        linkDirectionalArrowLength={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          const edgeType = link.edge_type as string;
          if (DIRECTED_EDGE_TYPES.has(edgeType)) {
            return mode === "EGO" ? 5 : 3;
          }
          return 0; // No arrow for CO-OWNER and SIMILARITY (undirected)
        }}
        linkDirectionalArrowRelPos={1}
        linkDirectionalArrowColor={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          const edgeType = link.edge_type as string;
          const baseColor =
            (edgeType && RELATION_COLORS[edgeType]) || RELATION_COLORS.UNKNOWN;
          return baseColor;
        }}
        linkColor={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          const relationType = link.edge_type as string;
          const baseColor =
            (relationType && RELATION_COLORS[relationType]) ||
            RELATION_COLORS.UNKNOWN;
          const weight = link.aggregated_weight || 1;

          if (mode === "EGO") {
            const opacity = Math.min(0.4 + Math.log10(weight) * 0.2, 0.8);
            const r = parseInt(baseColor.slice(1, 3), 16);
            const g = parseInt(baseColor.slice(3, 5), 16);
            const b = parseInt(baseColor.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${opacity})`;
          } else {
            return `${baseColor}66`;
          }
        }}
        linkWidth={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          if (mode === "CLUSTER") return MIN_LINK_WIDTH;
          const weight = link.aggregated_weight || 1;
          return Math.max(MIN_LINK_WIDTH, Math.sqrt(weight));
        }}
        linkDirectionalParticles={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          if (mode === "CLUSTER") return 0;
          const edgeType = link.edge_type as string;
          // Only show particles for directed edges in EGO mode
          if (!DIRECTED_EDGE_TYPES.has(edgeType)) return 0;
          const weight = link.aggregated_weight || 1;
          return weight > 1
            ? Math.min(Math.floor(Math.log2(weight)) + 1, 4)
            : 0;
        }}
        linkDirectionalParticleWidth={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          const weight = link.aggregated_weight || 1;
          return weight > 5 ? 2.2 : 1.2;
        }}
        linkCurvature={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          if (!link.multiLinkCount || link.multiLinkCount <= 1) return 0;
          const index = link.multiLinkIndex ?? 0;
          const count = link.multiLinkCount;
          return (index - (count - 1) / 2) * 0.15;
        }}
        // FIX 1 & 7: node label hover tooltip with cluster_id
        nodeLabel={(node: NodeObject<ExtendedNode>) => {
          if (mode === "CLUSTER" && processedData.nodes.length > 500) return "";

          const extNode = node as ExtendedNode;
          const isTarget = isTargetNode(node);
          const riskLabel = extNode.risk_label || "UNKNOWN";
          const isHighRisk =
            riskLabel === "MALICIOUS" || riskLabel === "HIGH_RISK";

          // FIX 7: Added cluster_id display
          return `
            <div style="background:rgba(2,6,23,0.97);border:1px solid #1e293b;padding:12px;font-family:'JetBrains Mono',monospace;font-size:10px;box-shadow:0 4px 20px rgba(0,0,0,0.5);min-width:220px;max-width:320px;">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                <div style="color:#00f2ff;font-weight:bold;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;">
                  ${extNode.attributes?.handle || "Unknown"}
                  ${isTarget ? '<span style="margin-left:6px;font-size:8px;padding:1px 4px;background:rgba(0,242,255,0.15);border:1px solid rgba(0,242,255,0.4);color:#00f2ff;">[TARGET]</span>' : ""}
                </div>
                <div style="font-size:8px;font-weight:bold;padding:2px 6px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:${isHighRisk ? "#ef4444" : "#64748b"};text-transform:uppercase;">${riskLabel}</div>
              </div>
              <div style="color:#475569;margin-bottom:8px;font-size:9px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">ID: ${node.id}</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-bottom:8px;">
                <div style="background:rgba(255,255,255,0.03);padding:4px 6px;border:1px solid #1e293b;">
                  <div style="color:#475569;font-size:8px;margin-bottom:2px;">RISK SCORE</div>
                  <div style="color:${isHighRisk ? "#ef4444" : "#22c55e"};font-weight:bold;font-size:13px;">${(extNode.risk_score || 0).toFixed(2)}</div>
                </div>
                <div style="background:rgba(255,255,255,0.03);padding:4px 6px;border:1px solid #1e293b;">
                  <div style="color:#475569;font-size:8px;margin-bottom:2px;">CLUSTER ID</div>
                  <div style="color:#94a3b8;font-weight:bold;font-size:13px;">#${extNode.cluster_id ?? "—"}</div>
                </div>
              </div>
              ${
                extNode.attributes?.trust_score !== undefined
                  ? `
              <div style="margin-bottom:8px;background:rgba(255,255,255,0.03);padding:4px 6px;border:1px solid #1e293b;">
                <div style="color:#475569;font-size:8px;margin-bottom:2px;">TRUST SCORE</div>
                <div style="color:#94a3b8;font-size:11px;">${Number(extNode.attributes.trust_score).toFixed(2)}</div>
              </div>`
                  : ""
              }
              ${
                ((extNode.attributes?.reasons as string[]) || []).length > 0
                  ? `
              <div style="border-top:1px solid #1e293b;padding-top:6px;">
                <div style="color:#334155;font-size:8px;font-weight:bold;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Detection Reasons</div>
                <div style="display:flex;flex-wrap:wrap;gap:3px;">
                  ${((extNode.attributes?.reasons as string[]) || [])
                    .slice(0, 4)
                    .map(
                      (r) =>
                        `<span style="background:rgba(255,255,255,0.05);border:1px solid #1e293b;padding:2px 5px;font-size:8px;color:#94a3b8;text-transform:uppercase;">${r}</span>`
                    )
                    .join("")}
                </div>
              </div>`
                  : ""
              }
            </div>
          `;
        }}
        onNodeClick={(node: NodeObject<ExtendedNode>) => {
          if (fgRef.current && node.x !== undefined && node.y !== undefined) {
            fgRef.current.centerAt(node.x, node.y, 1000);
            fgRef.current.zoom(4, 1000);
          }
        }}
        cooldownTicks={100}
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
      />

      <GraphLegend
        extraItems={
          mode === "EGO" ? (
            <div className="mb-1 flex items-center gap-3">
              <div className="h-3 w-3 animate-pulse rounded-full bg-[#00f2ff] shadow-[0_0_8px_rgba(0,242,255,0.6)]" />
              <span className="text-accent-cyan font-mono text-[9px] font-bold uppercase italic">
                Target Node
              </span>
            </div>
          ) : undefined
        }
      />
    </div>
  );
};

export default UniversalGraph2D;
