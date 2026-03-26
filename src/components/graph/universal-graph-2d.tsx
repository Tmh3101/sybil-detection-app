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
  DEFAULT_LINK_WIDTH,
  DIRECTED_EDGE_TYPES,
} from "@/lib/graph-constants";
import GraphLegend from "./graph-legend";
import { useGraphProcessor, AggregatedLink } from "@/hooks/use-graph-processor";
import { Maximize2, ZoomIn, ZoomOut } from "lucide-react";

type ExtendedNode = SybilNode;

export interface UniversalGraph2DProps {
  graphData: { nodes: SybilNode[]; links: SybilEdge[] };
  mode: "EGO" | "CLUSTER";
  targetId?: string;
  risk_label?: RiskClassification; // kept for compat, color derived from data
  depthFilter?: 1 | 2;
  onClusterNodeClick?: (clusterId: number, nodes: SybilNode[]) => void;
  allNodes?: SybilNode[];
}

// ─── Letter-avatar: drawn directly on canvas, no CORS needed ───
function drawLetterAvatar(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  size: number,
  color: string,
  handle: string
) {
  // Fill circle background
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.fillStyle = color + "22"; // slightly more opaque
  ctx.fill();

  // Draw letter
  const letter = (handle || "?").charAt(0).toUpperCase();
  const fs = Math.max(size * 0.85, 5);
  ctx.font = `bold ${fs}px "JetBrains Mono",monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = color + "bb";
  ctx.fillText(letter, x, y);
}

const UniversalGraph2D: React.FC<UniversalGraph2DProps> = ({
  graphData,
  mode,
  targetId,
  depthFilter = 2,
  onClusterNodeClick,
  allNodes,
}) => {
  const fgRef = useRef<
    ForceGraphMethods<ExtendedNode, AggregatedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // THÊM: State này dùng để ép re-render component khi ảnh tải xong
  const [, setAvatarTrigger] = useState(0);

  // ─── FIX 1: Image cache — stable ref, no state, reheat on load ───
  const imgCache = useRef<
    Record<string, HTMLImageElement | "error" | "pending">
  >({});

  // ─── Depth filter (frontend, EGO only) ───
  const depthFilteredData = useMemo(() => {
    if (mode !== "EGO" || depthFilter === 2 || !targetId) return graphData;
    const tid = targetId.toLowerCase();
    const direct = new Set<string>([tid]);
    graphData.links.forEach((l) => {
      const s = String(
        typeof l.source === "object"
          ? (l.source as { id: string }).id
          : l.source
      ).toLowerCase();
      const t = String(
        typeof l.target === "object"
          ? (l.target as { id: string }).id
          : l.target
      ).toLowerCase();
      if (s === tid) direct.add(t);
      if (t === tid) direct.add(s);
    });
    return {
      nodes: graphData.nodes.filter((n) =>
        direct.has(String(n.id).toLowerCase())
      ),
      links: graphData.links.filter((l) => {
        const s = String(
          typeof l.source === "object"
            ? (l.source as { id: string }).id
            : l.source
        ).toLowerCase();
        const t = String(
          typeof l.target === "object"
            ? (l.target as { id: string }).id
            : l.target
        ).toLowerCase();
        return direct.has(s) && direct.has(t);
      }),
    };
  }, [graphData, mode, depthFilter, targetId]);

  const processedData = useGraphProcessor(depthFilteredData, {
    targetId: mode === "EGO" ? targetId : undefined,
    aggregateEdges: true,
  });

  // Target color for legend
  const targetNodeColor = useMemo(() => {
    if (!targetId) return LABEL_COLORS.BENIGN;
    const found = processedData.nodes.find(
      // SỬA: Ép về chữ thường để so sánh an toàn ID và
      (n) => String(n.id).toLowerCase() === String(targetId).toLowerCase()
    );
    // SỬA: Loại bỏ khoảng trắng và in hoa risk_label để map đúng với LABEL_COLORS
    const rl = String(found?.risk_label || "UNKNOWN")
      .toUpperCase()
      .trim();
    return LABEL_COLORS[rl] || LABEL_COLORS.UNKNOWN;
  }, [processedData.nodes, targetId]);

  // Resize
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

  // Forces
  useEffect(() => {
    if (!fgRef.current) return;
    if (mode === "EGO") {
      fgRef.current.d3Force("radial", d3.forceRadial(190, 0, 0));
      (
        fgRef.current.d3Force("charge") as d3.ForceManyBody<ExtendedNode>
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

  // ─── FIX 3: Load image properly, reheat on success ───
  const getOrLoadImage = useCallback(
    (rawUrl: string | undefined): HTMLImageElement | null => {
      if (!rawUrl) return null;
      const url = resolvePictureUrl(rawUrl);
      if (!url) return null;

      console.log(`url: ${url}`);

      const cached = imgCache.current[url];
      if (cached === "error" || cached === "pending") return null;
      if (cached instanceof HTMLImageElement) {
        return cached.complete && cached.naturalWidth > 0 ? cached : null;
      }

      // First request
      imgCache.current[url] = "pending";
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        imgCache.current[url] = img;
        // THÊM: Kích hoạt React vẽ lại đồ thị với ảnh đã cache
        setAvatarTrigger((prev) => prev + 1);
        fgRef.current?.d3ReheatSimulation();
      };
      img.onerror = () => {
        imgCache.current[url] = "error";
        console.error("Failed to load node image:", url);
      };
      img.src = url;
      return null;
    },
    []
  );

  // ─── NODE CANVAS RENDERER ───
  const drawNode = useCallback(
    (
      node: NodeObject<ExtendedNode>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const ext = node as ExtendedNode;
      // SỬA: Đảm bảo string chuẩn xác 100% (ví dụ: " MALICIOUS " -> "MALICIOUS")
      const rl = String(ext.risk_label || "UNKNOWN")
        .toUpperCase()
        .trim();

      // SỬA: Phải check !!targetId để tránh lỗi undefined
      const isTarget =
        mode === "EGO" &&
        !!targetId &&
        String(node.id).toLowerCase() === String(targetId).toLowerCase();
      const isMalicious = rl === "MALICIOUS";
      const isHighRisk = rl === "HIGH_RISK";

      const size =
        mode === "EGO"
          ? isTarget
            ? 20 // Increased from 16
            : 10 // Increased from 6
          : isMalicious
            ? 9
            : isHighRisk
              ? 7
              : 5;

      const x = node.x ?? 0;
      const y = node.y ?? 0;

      // Color from label
      const color = LABEL_COLORS[rl] || LABEL_COLORS.UNKNOWN;

      // ── Glow aura ──
      if (isTarget) {
        // Larger, more vibrant aura for target
        ([size + 18, size + 10, size + 4] as number[]).forEach(
          (r: number, i: number) => {
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = color + (["0a", "1a", "30"] as string[])[i];
            ctx.fill();
          }
        );

        // Vibrant outer ring
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else if (isMalicious && mode === "CLUSTER") {
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(239,68,68,0.22)";
        ctx.fill();
      } else if (isHighRisk && mode === "CLUSTER") {
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(251,146,60,0.18)";
        ctx.fill();
      }

      // ── Node body ──
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.clip();

      const skipImg = mode === "CLUSTER" && processedData.nodes.length > 1500;
      const rawUrl = ext.attributes?.picture_url
        ? String(ext.attributes.picture_url)
        : undefined;
      const handle = String(ext.attributes?.handle || node.id || "?");
      const img = skipImg ? null : getOrLoadImage(rawUrl);

      if (img) {
        try {
          ctx.drawImage(img, x - size, y - size, size * 2, size * 2);
        } catch {
          drawLetterAvatar(ctx, x, y, size, color, handle);
        }
      } else {
        drawLetterAvatar(ctx, x, y, size, color, handle);
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
        if (isTarget) {
          ctx.font = `bold ${Math.max(10 / globalScale, 6)}px "JetBrains Mono",monospace`;
          ctx.fillStyle = color;
          ctx.fillText("[TARGET]", x, y - size - 14 / globalScale);
        }

        const fs =
          mode === "EGO" && isTarget ? 14 / globalScale : 9 / globalScale;
        ctx.font = `${isTarget ? "bold " : ""}${Math.max(fs, 4)}px "JetBrains Mono",monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(handle.slice(0, 14), x, y + size + 2);
      }
    },
    [mode, targetId, processedData.nodes.length, getOrLoadImage]
  );

  // ── Tooltip ──
  const nodeLabel = useCallback(
    (node: NodeObject<ExtendedNode>) => {
      if (mode === "CLUSTER" && processedData.nodes.length > 600) return "";
      const ext = node as ExtendedNode;
      // SỬA: Chuẩn hóa cho Tooltip
      const rl = String(ext.risk_label || "UNKNOWN")
        .toUpperCase()
        .trim();
      const c = LABEL_COLORS[rl] || LABEL_COLORS.UNKNOWN;
      const isHigh = rl === "MALICIOUS" || rl === "HIGH_RISK";
      const reasons = (ext.attributes?.reasons as string[]) || [];

      return `
      <div style="background:#020617;border:1px solid #1e293b;padding:12px;font-family:'JetBrains Mono',monospace;font-size:10px;min-width:230px;max-width:320px;box-shadow:0 8px 32px rgba(0,0,0,0.7);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="color:#00f2ff;font-weight:bold;font-size:12px;max-width:155px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${ext.attributes?.handle || "—"}</span>
          <span style="font-size:8px;padding:2px 5px;border:1px solid ${c}44;color:${c};background:${c}11;text-transform:uppercase;">${rl}</span>
        </div>
        <div style="color:#1e293b;font-size:8px;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${node.id}</div>
        <div style="display:grid;grid-template-columns:repeat(${mode === "CLUSTER" ? 3 : 2}, 1fr);gap:3px;margin-bottom:8px;">
          <div style="background:#0a0f1a;padding:4px 6px;border:1px solid #1e293b;">
            <div style="color:#334155;font-size:7px;margin-bottom:2px;">RISK SCORE</div>
            <div style="color:${isHigh ? "#ef4444" : "#22c55e"};font-weight:bold;font-size:13px;">${((ext.risk_score || 0) * 100).toFixed(0)}%</div>
          </div>
          ${
            mode === "CLUSTER"
              ? `
          <div style="background:#0a0f1a;padding:4px 6px;border:1px solid #1e293b;">
            <div style="color:#334155;font-size:7px;margin-bottom:2px;">CLUSTER</div>
            <div style="color:#64748b;font-weight:bold;font-size:13px;">#${ext.cluster_id ?? "-"}</div>
          </div>`
              : ""
          }
        </div>
        ${
          reasons.length > 0
            ? `
        <div style="border-top:1px solid #0f172a;padding-top:6px;">
          <div style="color:#1e293b;font-size:7px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Detection Flags</div>
          <div style="display:flex;flex-wrap:wrap;gap:3px;">${reasons
            .slice(0, 3)
            .map(
              (r) =>
                `<span style="background:#0f172a;border:1px solid #1e293b;padding:2px 5px;font-size:8px;color:#475569;">${r}</span>`
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
    (node: NodeObject<ExtendedNode>) => {
      if (mode === "CLUSTER" && onClusterNodeClick) {
        const cid = (node as ExtendedNode).cluster_id;
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

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[400px] w-full bg-[#050608]"
    >
      <ForceGraph2D
        key={`fg-${mode}-${depthFilter}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "always"}
        nodeLabel={nodeLabel}
        onNodeClick={handleNodeClick}
        linkDirectionalArrowLength={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) =>
          DIRECTED_EDGE_TYPES.has((link.edge_type as string) || "")
            ? mode === "EGO"
              ? 5
              : 3
            : 0
        }
        linkDirectionalArrowRelPos={0.95}
        linkDirectionalArrowColor={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) =>
          RELATION_COLORS[(link.edge_type as string) || ""] ||
          RELATION_COLORS.UNKNOWN
        }
        linkColor={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
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
        linkWidth={(link: LinkObject<ExtendedNode, AggregatedLink>) =>
          mode === "CLUSTER"
            ? DEFAULT_LINK_WIDTH
            : Math.max(MIN_LINK_WIDTH, Math.sqrt(link.aggregated_weight || 1))
        }
        linkDirectionalParticles={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          if (mode === "CLUSTER") return 0;
          if (!DIRECTED_EDGE_TYPES.has((link.edge_type as string) || ""))
            return 0;
          const w = link.aggregated_weight || 1;
          return w > 1 ? Math.min(Math.floor(Math.log2(w)) + 1, 4) : 0;
        }}
        linkDirectionalParticleWidth={() => 1.5}
        linkCurvature={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          if (!link.multiLinkCount || link.multiLinkCount <= 1) return 0;
          return (
            ((link.multiLinkIndex ?? 0) - (link.multiLinkCount - 1) / 2) * 0.18
          );
        }}
        cooldownTicks={120}
        d3AlphaDecay={0.018}
        d3VelocityDecay={0.35}
      />

      {/* ── Zoom Controls ── */}
      <div className="absolute right-6 bottom-6 z-20 flex flex-col gap-1.5">
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

      <GraphLegend
        graphData={depthFilteredData}
        extraItems={
          mode === "EGO" ? (
            <div className="mb-2 flex items-center gap-3">
              <div
                className="h-2.5 w-2.5 animate-pulse rounded-full"
                style={{
                  backgroundColor: targetNodeColor,
                  boxShadow: `0 0 8px ${targetNodeColor}99`,
                }}
              />
              <span
                className="font-mono text-[9px] font-bold uppercase italic"
                style={{ color: targetNodeColor }}
              >
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
