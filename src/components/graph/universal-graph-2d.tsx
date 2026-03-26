"use client";

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo,
} from "react";
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
  DIRECTED_EDGE_TYPES,
} from "@/lib/graph-constants";
import GraphLegend from "./graph-legend";
import { useGraphProcessor, AggregatedLink } from "@/hooks/use-graph-processor";
import { Maximize2, ZoomIn, ZoomOut, Weight } from "lucide-react";

type EnrichedNode = SybilNode & {
  __color?: string;
  __isTarget?: boolean;
};

function getNodePictureUrl(
  node: Partial<SybilNode> & { picture_url?: string }
) {
  const attrPicture =
    typeof node.attributes?.picture_url === "string"
      ? node.attributes.picture_url
      : "";
  const rootPicture =
    typeof node.picture_url === "string" ? node.picture_url : "";
  return attrPicture || rootPicture || "";
}

function getNodeHandle(node: Partial<SybilNode> & { handle?: string }) {
  const attrHandle =
    typeof node.attributes?.handle === "string" ? node.attributes.handle : "";
  const rootHandle = typeof node.handle === "string" ? node.handle : "";
  return attrHandle || rootHandle || String(node.id || "?");
}

function getNodeRiskLabel(node: Partial<SybilNode>) {
  return String(node.risk_label || "UNKNOWN")
    .trim()
    .toUpperCase();
}

export interface UniversalGraph2DProps {
  graphData: { nodes: SybilNode[]; links: SybilEdge[] };
  mode: "EGO" | "CLUSTER";
  targetId?: string;
  risk_label?: RiskClassification;
  depthFilter?: 1 | 2;
  onClusterNodeClick?: (clusterId: number, nodes: SybilNode[]) => void;
  allNodes?: SybilNode[];
}

export default function UniversalGraph2D({
  graphData,
  mode,
  targetId,
  depthFilter = 2,
  onClusterNodeClick,
  allNodes,
}: UniversalGraph2DProps) {
  const fgRef = useRef<
    ForceGraphMethods<EnrichedNode, AggregatedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [showWeights, setShowWeights] = useState(false);
  const [, setImageVersion] = useState(0);

  const imgCache = useRef<
    Record<string, HTMLImageElement | "error" | "pending">
  >({});

  // ─── Depth filter (frontend, EGO only) ───
  const depthFilteredData = useMemo(() => {
    if (mode !== "EGO" || depthFilter === 2 || !targetId) return graphData;
    const direct = new Set<string>([String(targetId)]);
    graphData.links.forEach((l) => {
      const s = String(
        typeof l.source === "object"
          ? (l.source as { id: string }).id
          : l.source
      );
      const t = String(
        typeof l.target === "object"
          ? (l.target as { id: string }).id
          : l.target
      );
      if (s === String(targetId)) direct.add(t);
      if (t === String(targetId)) direct.add(s);
    });
    return {
      nodes: graphData.nodes.filter((n) => direct.has(String(n.id))),
      links: graphData.links.filter((l) => {
        const s = String(
          typeof l.source === "object"
            ? (l.source as { id: string }).id
            : l.source
        );
        const t = String(
          typeof l.target === "object"
            ? (l.target as { id: string }).id
            : l.target
        );
        return direct.has(s) && direct.has(t);
      }),
    };
  }, [graphData, mode, depthFilter, targetId]);

  const processedData = useGraphProcessor(depthFilteredData, {
    targetId: mode === "EGO" ? targetId : undefined,
    aggregateEdges: true,
  });

  // ─── Resize ───
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        setDimensions({ width, height });
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // ─── D3 forces ───
  useEffect(() => {
    if (!fgRef.current) return;
    if (mode === "EGO") {
      fgRef.current.d3Force("radial", d3.forceRadial(190, 0, 0));
      (
        fgRef.current.d3Force("charge") as d3.ForceManyBody<EnrichedNode>
      )?.strength(-260);
    } else {
      fgRef.current.d3Force("x", d3.forceX(0).strength(0.05));
      fgRef.current.d3Force("y", d3.forceY(0).strength(0.05));
      fgRef.current.d3Force("center", d3.forceCenter(0, 0));
      fgRef.current.d3Force("charge", d3.forceManyBody().strength(-120));
      fgRef.current.d3Force("link")?.distance(40);
    }
    fgRef.current.d3ReheatSimulation();
  }, [processedData, mode, dimensions.width, dimensions.height]);

  const getOrLoadImage = useCallback(
    (rawUrl: string | undefined): HTMLImageElement | null => {
      if (!rawUrl) return null;
      const url = resolvePictureUrl(rawUrl);
      if (!url) return null;

      const cached = imgCache.current[url];
      if (cached === "error" || cached === "pending") return null;
      if (cached instanceof HTMLImageElement) {
        return cached.complete && cached.naturalWidth > 0 ? cached : null;
      }

      // First request
      imgCache.current[url] = "pending";
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        imgCache.current[url] = img;
        setImageVersion((v) => v + 1);
      };
      img.onerror = () => {
        imgCache.current[url] = "error";
      };
      img.src = url;
      return null;
    },
    []
  );

  // ─── NODE RENDERER ───
  const drawNode = useCallback(
    (
      node: NodeObject<EnrichedNode>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const n = node as EnrichedNode;

      const isTarget = !!n.__isTarget;
      const baseColor = n.__color || LABEL_COLORS.UNKNOWN;
      const color = baseColor;

      const riskLabel = getNodeRiskLabel(n);
      const isMalicious = riskLabel === "MALICIOUS";
      const isHighRisk = riskLabel === "HIGH_RISK";

      const size =
        mode === "EGO"
          ? isTarget
            ? 12
            : 6
          : 6;

      const x = n.x ?? 0;
      const y = n.y ?? 0;

      // ── Glow aura ──
      if (isTarget) {
        ([size + 12, size + 7, size + 3] as number[]).forEach((r, i) => {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = color + (["0d", "20", "38"] as string[])[i];
          ctx.fill();
        });
      } else if (mode === "CLUSTER" && isMalicious) {
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(239,68,68,0.2)";
        ctx.fill();
      } else if (mode === "CLUSTER" && isHighRisk) {
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(251,146,60,0.18)";
        ctx.fill();
      }

      // ── Node body: background + clip + avatar or letter ──
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);

      ctx.fillStyle = color + "15";
      ctx.fill();

      // Now clip for avatar/letter
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.clip();

      const skipImg = mode === "CLUSTER" && processedData.nodes.length > 500;
      const rawUrl = getNodePictureUrl(n);
      const handle = getNodeHandle(n);
      const img = skipImg ? null : getOrLoadImage(rawUrl);

      if (img) {
        try {
          ctx.drawImage(img, x - size, y - size, size * 2, size * 2);
        } catch {
          // CORS taint — letter fallback
          drawLetter(ctx, x, y, size, color, handle);
        }
      } else {
        drawLetter(ctx, x, y, size, color, handle);
      }

      ctx.restore();

      // ── Border ring ──
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = isTarget ? 2.5 : isMalicious || isHighRisk ? 1.8 : 1;
      ctx.stroke();

      // ── Target: dashed orbit ──
      if (isTarget) {
        ctx.save();
        ctx.strokeStyle = color + "70";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(x, y, size + 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // ── Labels ──
      const showLabel =
        (mode === "EGO" && (isTarget || globalScale > 1.8)) ||
        (mode === "CLUSTER" && globalScale > 2.8);

      if (showLabel) {
        const fs =
          mode === "EGO" && isTarget ? 12 / globalScale : 9 / globalScale;
        ctx.font = `${isTarget ? "bold " : ""}${Math.max(fs, 4)}px "JetBrains Mono",monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(handle.slice(0, 14), x, y + size + 2);
      }
    },
    [mode, processedData.nodes.length, getOrLoadImage]
  );

  // ── GAT Weight label on edges (shown at high zoom) ──
  const drawEdgeWeight = useCallback(
    (
      link: LinkObject<EnrichedNode, AggregatedLink>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      if (!showWeights || globalScale < 2.5) return;

      const src = link.source as { x?: number; y?: number };
      const tgt = link.target as { x?: number; y?: number };
      if (src.x === undefined || tgt.x === undefined) return;

      const midX = ((src.x || 0) + (tgt.x || 0)) / 2;
      const midY = ((src.y || 0) + (tgt.y || 0)) / 2;

      const w = (link as AggregatedLink).aggregated_weight || 1;
      const label = w.toFixed(0);

      const fs = Math.max(2.5, 8 / globalScale);
      ctx.font = `bold ${fs}px monospace`;

      const edgeType = ((link as AggregatedLink).edge_type as string) || "";
      const edgeColor = RELATION_COLORS[edgeType] || RELATION_COLORS.UNKNOWN;

      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(midX - tw / 2 - 2, midY - fs / 2 - 1, tw + 4, fs + 2);

      ctx.fillStyle = edgeColor + "dd";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, midX, midY);
    },
    [showWeights]
  );

  // ── Tooltip ──
  const nodeLabel = useCallback(
    (node: NodeObject<EnrichedNode>) => {
      if (mode === "CLUSTER" && processedData.nodes.length > 600) return "";
      const n = node as EnrichedNode;
      const rl = getNodeRiskLabel(n);
      const c = n.__color || LABEL_COLORS.UNKNOWN;
      const isHigh = rl === "MALICIOUS" || rl === "HIGH_RISK";
      const reasons = (n.attributes?.reasons as string[]) || [];
      const handle = getNodeHandle(n);

      // Explicit cluster_id access from root or attributes
      const clusterId =
        n.cluster_id !== undefined && n.cluster_id !== null
          ? n.cluster_id
          : n.attributes?.cluster_id;

      return `
      <div style="background:#020617;border:1px solid #1e293b;padding:12px;font-family:'JetBrains Mono',monospace;font-size:10px;min-width:230px;max-width:320px;box-shadow:0 8px 32px rgba(0,0,0,0.7);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#00f2ff;font-weight:bold;font-size:12px;max-width:155px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${handle}</span>
          <span style="font-size:8px;padding:2px 5px;border:1px solid ${c}44;color:${c};background:${c}11;text-transform:uppercase;">${rl}</span>
        </div>
        <div style="color:#64748b;font-size:8px;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${node.id}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;margin-bottom:8px;">
          <div style="background:#0a0f1a;padding:4px 6px;border:1px solid #1e293b;">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">RISK</div>
            <div style="color:${isHigh ? "#ef4444" : "#22c55e"};font-weight:bold;font-size:13px;">${((n.risk_score || 0) * 100).toFixed(0)}%</div>
          </div>
          <div style="background:#0a0f1a;padding:4px 6px;border:1px solid #1e293b;">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">CLUSTER</div>
            <div style="color:#f1f5f9;font-weight:bold;font-size:13px;">#${clusterId ?? "-"}</div>
          </div>
          <div style="background:#0a0f1a;padding:4px 6px;border:1px solid #1e293b;">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">TRUST</div>
            <div style="color:#f1f5f9;font-size:11px;">${Number(n.attributes?.trust_score || 0).toFixed(1)}</div>
          </div>
        </div>
        ${
          reasons.length > 0
            ? `
        <div style="border-top:1px solid #0f172a;padding-top:6px;">
          <div style="color:#64748b;font-size:7px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Detection Flags</div>
          <div style="display:flex;flex-wrap:wrap;gap:3px;">${reasons
            .slice(0, 3)
            .map(
              (r) =>
                `<span style="background:#0f172a;border:1px solid #1e293b;padding:2px 5px;font-size:8px;color:#94a3b8;">${r}</span>`
            )
            .join("")}</div>
        </div>`
            : ""
        }
      </div>`;
    },
    [mode, processedData.nodes.length]
  );

  // ── Node click ──
  const handleNodeClick = useCallback(
    (node: NodeObject<EnrichedNode>) => {
      if (mode === "CLUSTER" && onClusterNodeClick) {
        const cid = (node as EnrichedNode).cluster_id;
        if (cid !== undefined && cid !== null) {
          const clusterNodes = (allNodes || processedData.nodes).filter(
            (n) => (n as SybilNode).cluster_id === cid
          ) as SybilNode[];
          onClusterNodeClick(cid, clusterNodes);
        }
      } else if (
        fgRef.current &&
        node.x !== undefined &&
        node.y !== undefined
      ) {
        fgRef.current.centerAt(node.x, node.y, 700);
        fgRef.current.zoom(5, 700);
      }
    },
    [mode, onClusterNodeClick, allNodes, processedData.nodes]
  );

  const zoomToFit = () => fgRef.current?.zoomToFit(400, 50);
  const zoomIn = () =>
    fgRef.current?.zoom((fgRef.current?.zoom() ?? 1) * 1.45, 200);
  const zoomOut = () =>
    fgRef.current?.zoom((fgRef.current?.zoom() ?? 1) / 1.45, 200);

  // Target node color for legend — read from enriched node
  const targetColor = useMemo(() => {
    if (!targetId) return LABEL_COLORS.BENIGN;
    const found = processedData.nodes.find(
      (n) => String(n.id) === String(targetId)
    ) as EnrichedNode | undefined;
    return found?.__color || LABEL_COLORS.BENIGN;
  }, [processedData.nodes, targetId]);

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[400px] w-full bg-[#050608]"
    >
      <ForceGraph2D
        // ─── FIX: NO imagesLoaded in key — prevents remount that clears imgCache ───
        key={`fg-${mode}-${depthFilter}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "replace"}
        nodeLabel={nodeLabel}
        onNodeClick={handleNodeClick}
        // ─── Directed arrows for Follow/Interact layers ───
        linkDirectionalArrowLength={(
          link: LinkObject<EnrichedNode, AggregatedLink>
        ) =>
          DIRECTED_EDGE_TYPES.has((link.edge_type as string) || "")
            ? mode === "EGO"
              ? 5
              : 3
            : 0
        }
        linkDirectionalArrowRelPos={0.95}
        linkDirectionalArrowColor={(
          link: LinkObject<EnrichedNode, AggregatedLink>
        ) =>
          RELATION_COLORS[(link.edge_type as string) || ""] ||
          RELATION_COLORS.UNKNOWN
        }
        linkColor={(link: LinkObject<EnrichedNode, AggregatedLink>) => {
          const t = (link.edge_type as string) || "";
          const base = RELATION_COLORS[t] || RELATION_COLORS.UNKNOWN;
          if (mode === "CLUSTER") return base + "50";
          const w = link.aggregated_weight || 1;
          const op = Math.min(0.35 + Math.log10(w) * 0.18, 0.8);
          const r = parseInt(base.slice(1, 3), 16);
          const g = parseInt(base.slice(3, 5), 16);
          const b = parseInt(base.slice(5, 7), 16);
          return `rgba(${r},${g},${b},${op})`;
        }}
        linkWidth={(link: LinkObject<EnrichedNode, AggregatedLink>) =>
          mode === "CLUSTER"
            ? MIN_LINK_WIDTH
            : Math.max(MIN_LINK_WIDTH, Math.sqrt(link.aggregated_weight || 1))
        }
        linkDirectionalParticles={(
          link: LinkObject<EnrichedNode, AggregatedLink>
        ) => {
          if (mode === "EGO") return 0; // disabled for inspector
          if (!DIRECTED_EDGE_TYPES.has((link.edge_type as string) || ""))
            return 0;
          const w = link.aggregated_weight || 1;
          return w > 1 ? Math.min(Math.floor(Math.log2(w)) + 1, 4) : 0;
        }}
        linkDirectionalParticleWidth={() => 1.5}
        linkCurvature={(link: LinkObject<EnrichedNode, AggregatedLink>) => {
          if (!link.multiLinkCount || link.multiLinkCount <= 1) return 0;
          return (
            ((link.multiLinkIndex ?? 0) - (link.multiLinkCount - 1) / 2) * 0.18
          );
        }}
        // ─── GAT weight labels (drawn AFTER default link) ───
        linkCanvasObject={showWeights ? drawEdgeWeight : undefined}
        linkCanvasObjectMode={showWeights ? () => "after" : undefined}
        cooldownTicks={120}
        d3AlphaDecay={0.018}
        d3VelocityDecay={0.35}
      />

      {/* ── Controls (zoom + weight toggle) ── */}
      <div className="absolute right-6 bottom-6 z-20 flex flex-col gap-1.5">
        <button
          onClick={() => setShowWeights((v) => !v)}
          title={showWeights ? "Hide edge weights" : "Show edge weights"}
          className={`flex h-8 w-8 items-center justify-center border backdrop-blur-sm transition-all active:scale-95 ${
            showWeights
              ? "border-accent-cyan/50 bg-accent-cyan/10 text-accent-cyan"
              : "hover:border-accent-cyan/30 hover:text-accent-cyan border-slate-700/80 bg-black/80 text-slate-500"
          }`}
        >
          <Weight size={12} />
        </button>
        <button
          onClick={zoomToFit}
          title="Zoom to fit"
          className="hover:border-accent-cyan/40 hover:text-accent-cyan flex h-8 w-8 items-center justify-center border border-slate-700/80 bg-black/80 text-slate-500 backdrop-blur-sm transition-all active:scale-95"
        >
          <Maximize2 size={12} />
        </button>
        <button
          onClick={zoomIn}
          title="Zoom in"
          className="hover:border-accent-cyan/40 hover:text-accent-cyan flex h-8 w-8 items-center justify-center border border-slate-700/80 bg-black/80 text-slate-500 backdrop-blur-sm transition-all active:scale-95"
        >
          <ZoomIn size={12} />
        </button>
        <button
          onClick={zoomOut}
          title="Zoom out"
          className="hover:border-accent-cyan/40 hover:text-accent-cyan flex h-8 w-8 items-center justify-center border border-slate-700/80 bg-black/80 text-slate-500 backdrop-blur-sm transition-all active:scale-95"
        >
          <ZoomOut size={12} />
        </button>
      </div>

      {/* Weight info hint */}
      {showWeights && (
        <div className="border-accent-cyan/20 absolute bottom-6 left-4 z-10 border bg-black/80 px-3 py-1.5 font-mono text-[8px] text-slate-500 backdrop-blur-sm">
          Showing semantic edge weights (1–5 scale)
          <br />
          <span className="text-slate-700">Zoom in to see labels on edges</span>
        </div>
      )}

      <GraphLegend
        graphData={depthFilteredData}
        extraItems={
          mode === "EGO" ? (
            <div className="mb-2 flex items-center gap-3">
              <div
                className="h-2.5 w-2.5 animate-pulse rounded-full"
                style={{
                  backgroundColor: targetColor,
                  boxShadow: `0 0 8px ${targetColor}99`,
                }}
              />
              <span
                className="font-mono text-[9px] font-bold uppercase italic"
                style={{ color: targetColor }}
              >
                Target Node
              </span>
            </div>
          ) : undefined
        }
      />
    </div>
  );
}

// ─── Letter avatar (drawn within canvas clip context) ───
function drawLetter(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  color: string,
  handle: string
) {
  const letter = (handle || "?").charAt(0).toUpperCase();
  const fs = Math.max(size * 0.85, 5);
  ctx.font = `bold ${fs}px "JetBrains Mono",monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color + "cc";
  ctx.fillText(letter, x, y);
}
