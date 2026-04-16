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
  LIGHT_LABEL_COLORS,
  RELATION_COLORS,
  LIGHT_RELATION_COLORS,
  DIRECTED_EDGE_TYPES,
  EDGE_LAYERS,
} from "@/lib/graph-constants";
import GraphLegend from "./graph-legend";
import { useGraphProcessor, AggregatedLink } from "@/hooks/use-graph-processor";
import { useThemeStore } from "@/store/theme-store";
import {
  Maximize2,
  ZoomIn,
  ZoomOut,
  Brain,
  Share2,
  Combine,
} from "lucide-react";
import { useTranslations } from "next-intl";

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
  label?: RiskClassification;
  depthFilter?: 1 | 2;
  onClusterNodeClick?: (clusterId: number, nodes: SybilNode[]) => void;
  onNodeClick?: (node: SybilNode) => void;
  onLinkClick?: (link: AggregatedLink) => void;
  allNodes?: SybilNode[];
}

export default function UniversalGraph2D({
  graphData,
  mode,
  targetId,
  depthFilter = 2,
  onClusterNodeClick,
  onNodeClick,
  onLinkClick,
  allNodes,
}: UniversalGraph2DProps) {
  const { theme } = useThemeStore();
  const isDark = theme === "dark";
  const tLegend = useTranslations("GraphLegend");
  const fgRef = useRef<
    ForceGraphMethods<EnrichedNode, AggregatedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [showAttention, setShowAttention] = useState(false);
  const [showAllEdges, setShowAllEdges] = useState(false);
  const [isMerged, setIsMerged] = useState(false);
  const [, setImageVersion] = useState(0);

  // ─── Layer Toggling (Phase 1) ───
  const [visibleLayers, setVisibleLayers] = useState<string[]>(
    EDGE_LAYERS.map((l) => l.key)
  );

  const handleToggleLayer = useCallback((layerKey: string) => {
    setVisibleLayers((prev) =>
      prev.includes(layerKey)
        ? prev.filter((k) => k !== layerKey)
        : [...prev, layerKey]
    );
  }, []);

  // ─── Hover States (Phase 2) ───
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [highlightLinks, setHighlightLinks] = useState<Set<string>>(new Set());
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());

  const imgCache = useRef<
    Record<string, HTMLImageElement | "error" | "pending">
  >({});

  // ─── Depth and Edge Filtering (frontend, EGO only) ───
  const filteredData = useMemo(() => {
    // Keep all edges, including *_REV, so bidirectional relations are visible.
    let links = graphData.links;

    if (mode === "EGO" && targetId) {
      const tid = String(targetId).toLowerCase();

      // 1. Calculate BFS distance (Tier) for each node
      const distances = new Map<string, number>();
      distances.set(tid, 0);

      const adjacency = new Map<string, Set<string>>();
      links.forEach((l) => {
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

        if (!adjacency.has(s)) adjacency.set(s, new Set());
        if (!adjacency.has(t)) adjacency.set(t, new Set());
        adjacency.get(s)!.add(t);
        adjacency.get(t)!.add(s);
      });

      const queue: [string, number][] = [[tid, 0]];
      const visited = new Set<string>([tid]);

      while (queue.length > 0) {
        const [currId, dist] = queue.shift()!;
        if (dist >= 2) continue; // Only care about up to Depth 2

        const neighbors = adjacency.get(currId);
        if (neighbors) {
          for (const neighbor of neighbors) {
            if (!visited.has(neighbor)) {
              visited.add(neighbor);
              const nextDist = dist + 1;
              distances.set(neighbor, nextDist);
              queue.push([neighbor, nextDist]);
            }
          }
        }
      }

      // 2. Filter nodes based on Depth Filter
      const nodes = graphData.nodes.filter((n) => {
        const d = distances.get(String(n.id).toLowerCase());
        if (d === undefined) return false;
        return d <= depthFilter;
      });

      const nodeIds = new Set(nodes.map((n) => String(n.id).toLowerCase()));

      // 3. Filter links based on depth and radial logic
      links = links.filter((l) => {
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
        return nodeIds.has(s) && nodeIds.has(t);
      });

      if (!showAllEdges) {
        links = links.filter((l) => {
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

          const distS = distances.get(s);
          const distT = distances.get(t);

          if (distS === undefined || distT === undefined) return false;

          // Radial logic: show only edges between different tiers
          // T0-T1, T1-T2, etc. (diff === 1)
          const diff = Math.abs(distS - distT);
          return diff === 1;
        });
      }

      // 4. Layer Filtering (Phase 1)
      links = links.filter((l) => {
        const et = l.edge_type || "UNKNOWN";
        let layerKey = "UNKNOWN";
        for (const layer of EDGE_LAYERS) {
          if (layer.types.includes(et)) {
            layerKey = layer.key;
            break;
          }
        }
        return visibleLayers.includes(layerKey);
      });

      // 5. Directed Edge Filtering (Only allow directed edges pointing to the target node)
      links = links.filter((l) => {
        const et = l.edge_type || "UNKNOWN";
        if (DIRECTED_EDGE_TYPES.has(et)) {
          const rawS = String(
            typeof l.source === "object"
              ? (l.source as { id: string }).id
              : l.source
          ).toLowerCase();
          const rawT = String(
            typeof l.target === "object"
              ? (l.target as { id: string }).id
              : l.target
          ).toLowerCase();
          const visualTarget = et.endsWith("_REV") ? rawS : rawT;
          return visualTarget === tid;
        }
        return true; // Keep undirected edges
      });

      // 6. Filter nodes that have no remaining edges
      const connectedNodeIds = new Set<string>([tid]);
      links.forEach((l) => {
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
        connectedNodeIds.add(s);
        connectedNodeIds.add(t);
      });

      const finalNodes = nodes.filter((n) =>
        connectedNodeIds.has(String(n.id).toLowerCase())
      );

      return { nodes: finalNodes, links };
    } else {
      // CLUSTER mode or no target
      // Apply Layer Filtering only
      links = links.filter((l) => {
        const et = l.edge_type || "UNKNOWN";
        let layerKey = "UNKNOWN";
        for (const layer of EDGE_LAYERS) {
          if (layer.types.includes(et)) {
            layerKey = layer.key;
            break;
          }
        }
        return visibleLayers.includes(layerKey);
      });
      return { nodes: graphData.nodes, links };
    }
  }, [graphData, mode, depthFilter, targetId, showAllEdges, visibleLayers]);

  const processedData = useGraphProcessor(filteredData, {
    targetId: mode === "EGO" ? targetId : undefined,
    aggregateEdges: true,
    mergeEdges: isMerged,
  });

  // ─── Hover Handler (Phase 2) ───
  const handleNodeHover = useCallback(
    (node: NodeObject<EnrichedNode> | null) => {
      if (node) {
        const nodeId = String(node.id);
        setHoverNode(nodeId);

        const newHighlightLinks = new Set<string>();
        const newHighlightNodes = new Set<string>();

        processedData.links.forEach((link: AggregatedLink) => {
          const s =
            typeof link.source === "object" ? link.source.id : link.source;
          const t =
            typeof link.target === "object" ? link.target.id : link.target;

          if (String(s) === nodeId || String(t) === nodeId) {
            if (link.id) newHighlightLinks.add(link.id);
            newHighlightNodes.add(String(s));
            newHighlightNodes.add(String(t));
          }
        });

        setHighlightLinks(newHighlightLinks);
        setHighlightNodes(newHighlightNodes);
      } else {
        setHoverNode(null);
        setHighlightLinks(new Set());
        setHighlightNodes(new Set());
      }
    },
    [processedData.links]
  );

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
      fgRef.current.d3Force("radial", d3.forceRadial(220, 0, 0));
      (
        fgRef.current.d3Force("charge") as d3.ForceManyBody<EnrichedNode>
      )?.strength(-350);
      fgRef.current.d3Force("link")?.distance(80);
    } else {
      fgRef.current.d3Force("x", d3.forceX(0).strength(0.1));
      fgRef.current.d3Force("y", d3.forceY(0).strength(0.1));
      fgRef.current.d3Force("center", d3.forceCenter(0, 0));
      // Push nodes further apart (anti-hairball)
      fgRef.current.d3Force("charge", d3.forceManyBody().strength(-150));
      // Stretch the edges to prevent tight clustering
      fgRef.current.d3Force("link")?.distance(50);
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
      const nodeId = String(n.id);

      const isTarget = !!n.__isTarget;
      const baseColor = n.__color || LABEL_COLORS.UNKNOWN;
      const color = baseColor;

      const riskLabel = getNodeRiskLabel(n);
      const isMalicious = riskLabel === "MALICIOUS";
      const isHighRisk = riskLabel === "HIGH_RISK";

      const size = mode === "EGO" ? (isTarget ? 10 : 6) : 6;

      const x = n.x ?? 0;
      const y = n.y ?? 0;

      // ── Visual Focus (Phase 2) ──
      ctx.save();
      const isHovering = hoverNode !== null;
      if (isHovering && hoverNode !== nodeId && !highlightNodes.has(nodeId)) {
        ctx.globalAlpha = 0.1;
      }

      // ── Glow aura ──
      if (isTarget) {
        ([size + 8, size + 4] as number[]).forEach((r, i) => {
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = color + (["0a", "18"] as string[])[i];
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
      ctx.lineWidth = isTarget ? 2 : isMalicious || isHighRisk ? 1.8 : 1;
      ctx.stroke();

      // ── Labels ──
      const showLabel =
        (mode === "EGO" && (isTarget || globalScale > 1.8)) ||
        (mode === "CLUSTER" && globalScale > 2.8);

      if (showLabel) {
        const fs =
          mode === "EGO" && isTarget ? 11 / globalScale : 9 / globalScale;
        ctx.font = `${isTarget ? "bold " : ""}${Math.max(fs, 4)}px "JetBrains Mono",monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = isDark ? color : "#1e293b";
        ctx.fillText(handle.slice(0, 14), x, y + size + 2);
      }
      ctx.restore();
    },
    [
      mode,
      processedData.nodes.length,
      getOrLoadImage,
      hoverNode,
      highlightNodes,
      isDark,
    ]
  );

  // ── GAT Attention label on edges (shown at high zoom) ──
  const drawEdgeAttention = useCallback(
    (
      link: LinkObject<EnrichedNode, AggregatedLink>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      if (!showAttention) return;
      if (!link.gat_attention) return;

      const src = link.source as { x?: number; y?: number };
      const tgt = link.target as { x?: number; y?: number };
      if (
        src.x === undefined ||
        src.y === undefined ||
        tgt.x === undefined ||
        tgt.y === undefined
      )
        return;

      // Tính vector hướng từ Nguồn đến Đích
      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;

      // Tính trung điểm gốc của đường thẳng
      let midX = src.x + dx / 2;
      let midY = src.y + dy / 2;

      // Tính toán độ lệch (offset) dựa trên curvature để text bám theo đường cong
      const curvature =
        link.multiLinkCount && link.multiLinkCount > 1
          ? ((link.multiLinkIndex ?? 0) - (link.multiLinkCount - 1) / 2) * 0.18
          : 0;

      if (curvature !== 0) {
        // Dịch chuyển trung điểm vuông góc với đường thẳng
        const distance = Math.sqrt(dx * dx + dy * dy);
        const nx = -dy / distance; // Normal vector X
        const ny = dx / distance; // Normal vector Y

        // Khoảng cách dịch chuyển phụ thuộc vào độ cong và chiều dài đoạn thẳng
        midX += nx * curvature * distance;
        midY += ny * curvature * distance;
      }

      const label = `${link.gat_attention?.toFixed(4) || "0.0000"}`;

      const fs = Math.max(2.5, 8 / globalScale);
      ctx.font = `bold ${fs}px "JetBrains Mono", monospace`;

      const tw = ctx.measureText(label).width;
      ctx.fillStyle = isDark
        ? "rgba(2, 6, 23, 0.85)"
        : "rgba(255, 255, 255, 0.85)";
      ctx.fillRect(midX - tw / 2 - 2, midY - fs / 2 - 1, tw + 4, fs + 2);

      ctx.fillStyle = isDark ? "#ef4444" : "#dc2626"; // AI Focus Red
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, midX, midY);
    },
    [showAttention, isDark]
  );

  // ── LINK RENDERER (Phase 3 LOD) ──
  const drawLink = useCallback(
    (
      link: LinkObject<EnrichedNode, AggregatedLink>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const l = link as AggregatedLink;
      const src = l.source as { x?: number; y?: number };
      const tgt = l.target as { x?: number; y?: number };

      if (
        !src ||
        !tgt ||
        src.x === undefined ||
        src.y === undefined ||
        tgt.x === undefined ||
        tgt.y === undefined
      )
        return;

      const type = (l.edge_type as string) || "";
      const palette = isDark ? RELATION_COLORS : LIGHT_RELATION_COLORS;
      const baseColor = palette[type] || palette.UNKNOWN;

      // ── Opacity Logic (Phase 2) ──
      let alpha = 0.25;
      if (
        type === "CO-OWNER" ||
        type.includes("SIMILARITY") ||
        type === "SAME_AVATAR"
      ) {
        alpha = 0.5;
      } else if (mode === "EGO") {
        alpha = 0.8; // Uniform opacity for EGO mode
      }

      const isHovering = hoverNode !== null;
      if (isHovering) {
        if (highlightLinks.has(l.id || "")) {
          alpha = 0.8;
        } else {
          alpha = 0.05;
        }
      }

      const r = parseInt(baseColor.slice(1, 3), 16);
      const g = parseInt(baseColor.slice(3, 5), 16);
      const b = parseInt(baseColor.slice(5, 7), 16);
      const color = `rgba(${r},${g},${b},${alpha})`;

      const width = 0.5; // Uniform thickness independent of weight

      ctx.save();
      if (type.endsWith("_REV")) {
        ctx.setLineDash([4, 3]);
      }
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.globalAlpha = 1.0; // We use rgba in strokeStyle instead of globalAlpha to keep it precise

      // ── LOD Rendering (Phase 3) ──
      if (globalScale < 2.0) {
        // Low Detail: Simple straight line
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
        ctx.stroke();
      } else {
        // High Detail: Curved line
        const curvature =
          l.multiLinkCount && l.multiLinkCount > 1
            ? ((l.multiLinkIndex ?? 0) - (l.multiLinkCount - 1) / 2) * 0.18
            : 0;

        if (curvature === 0) {
          ctx.moveTo(src.x, src.y);
          ctx.lineTo(tgt.x, tgt.y);
        } else {
          const dx = tgt.x - src.x;
          const dy = tgt.y - src.y;
          const cp = {
            x: (src.x + tgt.x) / 2 + curvature * -dy,
            y: (src.y + tgt.y) / 2 + curvature * dx,
          };
          ctx.moveTo(src.x, src.y);
          ctx.quadraticCurveTo(cp.x, cp.y, tgt.x, tgt.y);
        }
        ctx.stroke();
      }

      // ── Text Labels: Always show when feature is enabled ──
      const shouldDrawTextHover = !hoverNode || highlightLinks.has(l.id || "");
      if (shouldDrawTextHover) {
        drawEdgeAttention(l, ctx, globalScale);
      }

      ctx.restore();
    },
    [mode, hoverNode, highlightLinks, drawEdgeAttention, isDark]
  );

  const paintLinkPointerArea = useCallback(
    (
      link: LinkObject<EnrichedNode, AggregatedLink>,
      color: string,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const l = link as AggregatedLink;
      const src = l.source as { x?: number; y?: number };
      const tgt = l.target as { x?: number; y?: number };

      if (
        !src ||
        !tgt ||
        src.x === undefined ||
        src.y === undefined ||
        tgt.x === undefined ||
        tgt.y === undefined
      )
        return;

      // Làm cho vùng nhận diện chuột dày hơn một chút để người dùng dễ hover
      ctx.lineWidth = Math.max(4, 10 / globalScale);
      ctx.strokeStyle = color; // Bắt buộc phải dùng tham số color này của thư viện truyền vào
      ctx.beginPath();

      // Bắt chước Y HỆT logic vẽ đường cong ở Phase 3 LOD
      if (globalScale < 2.0) {
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(tgt.x, tgt.y);
      } else {
        const curvature =
          l.multiLinkCount && l.multiLinkCount > 1
            ? ((l.multiLinkIndex ?? 0) - (l.multiLinkCount - 1) / 2) * 0.18
            : 0;

        if (curvature === 0) {
          ctx.moveTo(src.x, src.y);
          ctx.lineTo(tgt.x, tgt.y);
        } else {
          const dx = tgt.x - src.x;
          const dy = tgt.y - src.y;
          const cp = {
            x: (src.x + tgt.x) / 2 + curvature * -dy,
            y: (src.y + tgt.y) / 2 + curvature * dx,
          };
          ctx.moveTo(src.x, src.y);
          ctx.quadraticCurveTo(cp.x, cp.y, tgt.x, tgt.y);
        }
      }
      ctx.stroke();
    },
    []
  );

  // ─── Tooltip ───
  const nodeLabel = useCallback(
    (node: NodeObject<EnrichedNode>) => {
      if (mode === "CLUSTER" && processedData.nodes.length > 600) return "";
      const n = node as EnrichedNode;
      const rl = getNodeRiskLabel(n);
      const palette = isDark ? LABEL_COLORS : LIGHT_LABEL_COLORS;
      const c = n.__color || palette.UNKNOWN;
      const handle = getNodeHandle(n);

      const bg = isDark ? "#020617" : "#ffffff";
      const border = isDark ? "#1e293b" : "#e2e8f0";
      const textMain = isDark ? "#00f2ff" : "#0284c7";
      const textSub = isDark ? "#64748b" : "#475569";
      const cardBg = isDark ? "#0a0f1a" : "#f8fafc";

      if (mode === "EGO") {
        return `
        <div style="background:${bg};border:1px solid ${border};padding:12px;font-family:'JetBrains Mono',monospace;font-size:10px;min-width:180px;box-shadow:0 8px 32px rgba(0,0,0,0.15);">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <span style="color:${textMain};font-weight:bold;font-size:12px;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${handle}</span>
            <span style="font-size:8px;padding:2px 5px;border:1px solid ${c}44;color:${c};background:${c}11;text-transform:uppercase;">${rl}</span>
          </div>
          <div style="color:${textSub};font-size:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${node.id}</div>
        </div>`;
      }

      const isHigh = rl === "MALICIOUS" || rl === "HIGH_RISK";
      const reasons = (n.attributes?.reasons as unknown as string[]) || [];
      const clusterId =
        n.cluster_id !== undefined && n.cluster_id !== null
          ? n.cluster_id
          : n.attributes?.cluster_id;

      return `
      <div style="background:${bg};border:1px solid ${border};padding:12px;font-family:'JetBrains Mono',monospace;font-size:10px;min-width:230px;max-width:320px;box-shadow:0 8px 32px rgba(0,0,0,0.15);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
          <span style="color:${textMain};font-weight:bold;font-size:12px;max-width:155px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${handle}</span>
          <span style="font-size:8px;padding:2px 5px;border:1px solid ${c}44;color:${c};background:${c}11;text-transform:uppercase;">${rl}</span>
        </div>
        <div style="color:${textSub};font-size:8px;margin-bottom:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${node.id}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:3px;margin-bottom:8px;">
          <div style="background:${cardBg};padding:4px 6px;border:1px solid ${border};">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">RISK</div>
            <div style="color:${isHigh ? "#ef4444" : "#22c55e"};font-weight:bold;font-size:13px;">${((n.risk_score || 0) * 100).toFixed(0)}%</div>
          </div>
          <div style="background:${cardBg};padding:4px 6px;border:1px solid ${border};">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">CLUSTER</div>
            <div style="color:${isDark ? "#f1f5f9" : "#1e293b"};font-weight:bold;font-size:13px;">#${clusterId ?? "-"}</div>
          </div>
          <div style="background:${cardBg};padding:4px 6px;border:1px solid ${border};">
            <div style="color:#475569;font-size:7px;margin-bottom:2px;">TRUST</div>
            <div style="color:${isDark ? "#f1f5f9" : "#1e293b"};font-size:11px;">${Number(n.attributes?.trust_score || 0).toFixed(1)}</div>
          </div>
        </div>
        ${
          reasons.length > 0
            ? `
        <div style="border-top:1px solid ${isDark ? "#0f172a" : "#f1f5f9"};padding-top:6px;">
          <div style="color:${textSub};font-size:7px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:4px;">Detection Flags</div>
          <div style="display:flex;flex-wrap:wrap;gap:3px;">${reasons
            .slice(0, 3)
            .map(
              (r) =>
                `<span style="background:${isDark ? "#0f172a" : "#f8fafc"};border:1px solid ${border};padding:2px 5px;font-size:8px;color:${textSub};">${r}</span>`
            )
            .join("")}</div>
        </div>`
            : ""
        }
      </div>`;
    },
    [mode, processedData.nodes.length, isDark]
  );

  const linkLabel = useCallback(
    (link: LinkObject<EnrichedNode, AggregatedLink>) => {
      const l = link as AggregatedLink;
      const type = (l.edge_type as string) || "UNKNOWN";
      const isMergedMultiple = l.is_merged_multiple;
      const displayType = isMergedMultiple ? `${type} (MERGED)` : type;
      const weight = l.aggregated_weight || 1;
      const violations = l.violations || [];
      const attention = l.gat_attention
        ? `<br/><span style="color: #ef4444; font-weight: bold;">AI Attention: ${l.gat_attention.toFixed(
            4
          )}</span>`
        : "";

      const violationsHtml =
        violations.length > 0
          ? `<div style="margin-top: 4px; border-top: 1px solid #1e293b; padding-top: 4px;">
             <span style="color: #ef4444; font-size: 8px; font-weight: bold; text-transform: uppercase;">Violations:</span>
             <div style="display: flex; flex-wrap: wrap; gap: 2px; margin-top: 2px;">
               ${violations
                 .map(
                   (v) =>
                     `<span style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); padding: 1px 4px; color: #ef4444; font-size: 7px;">${v}</span>`
                 )
                 .join("")}
             </div>
           </div>`
          : "";

      const bg = isDark ? "rgba(2, 6, 23, 0.95)" : "rgba(255, 255, 255, 0.95)";
      const border = isDark ? "#1e293b" : "#e2e8f0";
      const textMain = isDark ? "#f1f5f9" : "#1e293b";
      const textSub = isDark ? "#64748b" : "#475569";
      const accent = isDark ? "#00f2ff" : "#0284c7";

      return `<div style="background: ${bg}; border: 1px solid ${border}; padding: 6px 10px; border-radius: 4px; font-size: 10px; font-family: 'JetBrains Mono', monospace; box-shadow: 0 4px 12px rgba(0,0,0,0.1); min-width: 140px;">
      <span style="color: ${textSub}; text-transform: uppercase; font-size: 8px;">Relationship</span>
      <div style="color: ${textMain}; margin-top: 2px;">Type: <span style="color: ${accent};">${displayType}</span></div>
      <div style="color: ${textMain};">Weight: <span style="color: ${accent};">${weight.toFixed(
        2
      )}</span></div>
      ${attention}
      ${violationsHtml}
    </div>`;
    },
    [isDark]
  );

  // ── Node click ──
  const handleNodeClick = useCallback(
    (node: NodeObject<EnrichedNode>) => {
      // Trigger side panel for inspector if prop provided
      if (onNodeClick) {
        onNodeClick(node as SybilNode);
      }

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
    [mode, onClusterNodeClick, onNodeClick, allNodes, processedData.nodes]
  );

  const zoomToFit = () => fgRef.current?.zoomToFit(400, 50);
  const zoomIn = () =>
    fgRef.current?.zoom((fgRef.current?.zoom() ?? 1) * 1.45, 200);
  const zoomOut = () =>
    fgRef.current?.zoom((fgRef.current?.zoom() ?? 1) / 1.45, 200);

  // Target node color for legend — read from enriched node
  const targetColor = useMemo(() => {
    const palette = isDark ? LABEL_COLORS : LIGHT_LABEL_COLORS;
    if (!targetId) return palette.BENIGN;
    const found = processedData.nodes.find(
      (n) => String(n.id) === String(targetId)
    ) as EnrichedNode | undefined;
    return found?.__color || palette.BENIGN;
  }, [processedData.nodes, targetId, isDark]);

  const buttonBaseClass =
    "flex h-8 w-8 items-center justify-center border backdrop-blur-sm transition-all active:scale-95";

  return (
    <div
      ref={containerRef}
      className="relative h-full min-h-[400px] w-full overflow-hidden transition-colors duration-300"
      style={{ backgroundColor: isDark ? "#050608" : "#f8fafc" }}
    >
      <ForceGraph2D
        key={`fg-${mode}-${depthFilter}-${theme}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor={isDark ? "#050608" : "#f8fafc"}
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "replace"}
        nodeLabel={nodeLabel}
        linkLabel={linkLabel}
        onNodeHover={handleNodeHover}
        onNodeClick={handleNodeClick}
        onLinkClick={(link) => onLinkClick?.(link as AggregatedLink)}
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
        ) => {
          const type = (link.edge_type as string) || "";
          const palette = isDark ? RELATION_COLORS : LIGHT_RELATION_COLORS;
          return palette[type] || palette.UNKNOWN;
        }}
        linkLineDash={(link: LinkObject<EnrichedNode, AggregatedLink>) =>
          ((link.edge_type as string) || "").endsWith("_REV") ? [4, 3] : null
        }
        linkPointerAreaPaint={paintLinkPointerArea}
        // ─── Phase 3: LOD Link Renderer ───
        linkCanvasObject={drawLink}
        linkCanvasObjectMode={() => "replace"}
        cooldownTicks={120}
        d3AlphaDecay={0.018}
        d3VelocityDecay={0.35}
      />

      {/* ── Controls (zoom + weight toggle) ── */}
      <div className="absolute right-6 bottom-6 z-20 flex flex-col gap-1.5">
        {mode === "EGO" && (
          <>
            <button
              onClick={() => setShowAllEdges((v) => !v)}
              title={showAllEdges ? "Show Target Edges Only" : "Show All Edges"}
              className={buttonBaseClass}
              style={{
                backgroundColor: showAllEdges
                  ? "rgba(0, 242, 255, 0.1)"
                  : isDark
                    ? "rgba(0, 0, 0, 0.8)"
                    : "rgba(255, 255, 255, 0.8)",
                borderColor: showAllEdges
                  ? "rgba(0, 242, 255, 0.5)"
                  : isDark
                    ? "rgba(51, 65, 85, 0.8)"
                    : "rgba(226, 232, 240, 1)",
                color: showAllEdges
                  ? "#00f2ff"
                  : isDark
                    ? "#64748b"
                    : "#334155",
              }}
            >
              <Share2 size={12} />
            </button>

            <button
              onClick={() => setShowAttention((v) => !v)}
              title={showAttention ? "Hide AI Attention" : "Show AI Attention"}
              className={buttonBaseClass}
              style={{
                backgroundColor: showAttention
                  ? "rgba(239, 68, 68, 0.1)"
                  : isDark
                    ? "rgba(0, 0, 0, 0.8)"
                    : "rgba(255, 255, 255, 0.8)",
                borderColor: showAttention
                  ? "rgba(239, 68, 68, 0.5)"
                  : isDark
                    ? "rgba(51, 65, 85, 0.8)"
                    : "rgba(226, 232, 240, 1)",
                color: showAttention
                  ? "#ef4444"
                  : isDark
                    ? "#64748b"
                    : "#334155",
              }}
            >
              <Brain size={12} />
            </button>
          </>
        )}
        <button
          onClick={() => setIsMerged((v) => !v)}
          title={isMerged ? "Separate Edges" : "Merge Edges"}
          className={buttonBaseClass}
          style={{
            backgroundColor: isMerged
              ? "rgba(0, 242, 255, 0.1)"
              : isDark
                ? "rgba(0, 0, 0, 0.8)"
                : "rgba(255, 255, 255, 0.8)",
            borderColor: isMerged
              ? "rgba(0, 242, 255, 0.5)"
              : isDark
                ? "rgba(51, 65, 85, 0.8)"
                : "rgba(226, 232, 240, 1)",
            color: isMerged ? "#00f2ff" : isDark ? "#64748b" : "#334155",
          }}
        >
          <Combine size={12} />
        </button>
        <button
          onClick={zoomToFit}
          title="Zoom to fit"
          className={buttonBaseClass}
          style={{
            backgroundColor: isDark
              ? "rgba(0, 0, 0, 0.8)"
              : "rgba(255, 255, 255, 0.8)",
            borderColor: isDark
              ? "rgba(51, 65, 85, 0.8)"
              : "rgba(226, 232, 240, 1)",
            color: isDark ? "#64748b" : "#334155",
          }}
        >
          <Maximize2 size={12} />
        </button>
        <button
          onClick={zoomIn}
          title="Zoom in"
          className={buttonBaseClass}
          style={{
            backgroundColor: isDark
              ? "rgba(0, 0, 0, 0.8)"
              : "rgba(255, 255, 255, 0.8)",
            borderColor: isDark
              ? "rgba(51, 65, 85, 0.8)"
              : "rgba(226, 232, 240, 1)",
            color: isDark ? "#64748b" : "#334155",
          }}
        >
          <ZoomIn size={12} />
        </button>
        <button
          onClick={zoomOut}
          title="Zoom out"
          className={buttonBaseClass}
          style={{
            backgroundColor: isDark
              ? "rgba(0, 0, 0, 0.8)"
              : "rgba(255, 255, 255, 0.8)",
            borderColor: isDark
              ? "rgba(51, 65, 85, 0.8)"
              : "rgba(226, 232, 240, 1)",
            color: isDark ? "#64748b" : "#334155",
          }}
        >
          <ZoomOut size={12} />
        </button>
      </div>

      {/* Weight info hint */}
      {showAttention && (
        <div
          className="absolute bottom-6 left-4 z-10 border px-3 py-1.5 font-mono text-[8px] backdrop-blur-sm"
          style={{
            backgroundColor: isDark
              ? "rgba(0, 0, 0, 0.8)"
              : "rgba(255, 255, 255, 0.8)",
            borderColor: isDark
              ? "rgba(239, 68, 68, 0.2)"
              : "rgba(239, 68, 68, 0.2)",
            color: isDark ? "#64748b" : "#64748b",
          }}
        >
          {tLegend("attention_weights_title")}
          <br />
          <span
            style={{
              color: isDark ? "rgba(51, 65, 85, 1)" : "rgba(100, 116, 139, 1)",
            }}
          >
            {tLegend("attention_weights_desc")}
          </span>
        </div>
      )}

      <GraphLegend
        graphData={filteredData}
        visibleLayers={visibleLayers}
        onToggleLayer={handleToggleLayer}
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
                {tLegend("target_node")}
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
