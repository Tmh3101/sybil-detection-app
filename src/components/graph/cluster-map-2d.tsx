"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge } from "@/types/api";
import { resolvePictureUrl } from "@/lib/utils";
import { LABEL_COLORS, RELATION_COLORS } from "@/lib/graph-constants";
import GraphLegend from "./graph-legend";

interface ClusterMap2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
}

const ClusterMap2D: React.FC<ClusterMap2DProps> = ({ graphData }) => {
  const fgRef = useRef<ForceGraphMethods<SybilNode, SybilEdge> | undefined>(
    undefined
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [imagesLoaded, setImagesLoaded] = useState(0); // Trigger re-render when images load

  // Thêm cache cho avatar để tối ưu render giống ego-graph
  const imgCache = useRef<Record<string, HTMLImageElement>>({});

  useEffect(() => {
    if (!containerRef.current) return;
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    handleResize();
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(containerRef.current);
    return () => resizeObserver.disconnect();
  }, []);

  // Tune D3 Forces for compactness
  useEffect(() => {
    if (!fgRef.current) return;

    // Pull nodes toward the center - especially important for disconnected clusters
    fgRef.current
      .d3Force("center")
      ?.x(dimensions.width / 2)
      .y(dimensions.height / 2);

    // Repulsion strength (negative is repulsion) - weaker repulsion means nodes stay closer
    fgRef.current.d3Force("charge")?.strength(-150);

    // Link distance - shorter distance brings linked nodes together
    fgRef.current.d3Force("link")?.distance(30);
  }, [graphData, dimensions]);

  const getNodeColor = useCallback((node: NodeObject<SybilNode>) => {
    return (
      (node.risk_label && LABEL_COLORS[node.risk_label]) || LABEL_COLORS.UNKNOWN
    );
  }, []);

  // Custom vẽ Node: bao gồm Avatar và chỉnh size to hơn
  const drawNode = useCallback(
    (node: NodeObject<SybilNode>, ctx: CanvasRenderingContext2D) => {
      // Logic xác định size và color dựa trên label
      const isMalicious = node.risk_label === "MALICIOUS";
      const size = isMalicious ? 8 : 5;
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const color = getNodeColor(node);

      // 1. Vẽ Glow (viền mờ) cho node malicious
      if (isMalicious) {
        ctx.beginPath();
        ctx.arc(x, y, size + 3, 0, 2 * Math.PI, false);
        ctx.fillStyle = "rgba(239, 68, 68, 0.3)";
        ctx.fill();
      }

      // 2. Logic load và vẽ Avatar
      const rawImgUrl = node.attributes?.picture_url;
      let img = null;
      if (rawImgUrl) {
        if (imgCache.current[rawImgUrl]) {
          img = imgCache.current[rawImgUrl];
        } else {
          const newImg = new Image();
          newImg.src = resolvePictureUrl(rawImgUrl);
          newImg.onload = () => {
            imgCache.current[rawImgUrl] = newImg;
            setImagesLoaded((prev) => prev + 1); // Trigger re-render
          };
          imgCache.current[rawImgUrl] = newImg;
        }
      }

      // Vẽ nền node (Clip thành hình tròn cho avatar)
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.clip();

      if (img && img.complete) {
        ctx.drawImage(img, x - size, y - size, size * 2, size * 2);
      } else {
        ctx.fillStyle = "#1e293b"; // Fallback color
        ctx.fill();
      }
      ctx.restore();

      // 3. Vẽ Viền (Stroke) thể hiện màu nhãn
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.strokeStyle = color;
      ctx.lineWidth = isMalicious ? 2 : 1;
      ctx.stroke();
    },
    [getNodeColor]
  );

  return (
    <div ref={containerRef} className="relative h-full w-full bg-black/40">
      <ForceGraph2D
        key={`fg-${imagesLoaded}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)"
        // --- PERFORMANCE OPTIMIZED LINKS ---
        linkColor={(link: SybilEdge) => {
          const color =
            (link.edge_type && RELATION_COLORS[link.edge_type]) ||
            RELATION_COLORS.UNKNOWN;
          return `${color}88`; // Higher transparency for dense clusters
        }}
        linkWidth={0.5}
        linkDirectionalParticles={0} // Disable for performance
        // --- SỬA NODE Ở ĐÂY ---
        nodeCanvasObjectMode={() => "replace"}
        nodeCanvasObject={drawNode}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        nodeLabel={(node: NodeObject<SybilNode>) => {
          const isHighRisk =
            node.risk_label === "MALICIOUS" || node.risk_label === "HIGH_RISK";
          return `
            <div class="bg-black/95 border border-slate-700 p-3 font-mono text-[10px] shadow-2xl min-w-[200px]">
              <div class="flex items-center justify-between mb-1">
                <div class="text-accent-cyan font-bold text-xs">${node.attributes?.handle || "Unknown Handle"}</div>
                <div class="text-[8px] font-bold text-slate-500 bg-slate-800/50 px-1 rounded uppercase">${node.risk_label}</div>
              </div>
              <div class="text-slate-500 mb-2 break-all">ID: ${node.id}</div>
              <div class="flex items-center justify-between mb-2">
                <span class="text-slate-400">RISK SCORE:</span>
                <span class="${isHighRisk ? "text-accent-red" : "text-green-500"} font-bold text-sm">
                  ${node.risk_score.toFixed(2)}
                </span>
              </div>
              <div class="text-slate-400 border-t border-slate-800 pt-2 italic leading-relaxed">
                ${node.attributes?.reason || "No reasoning provided."}
              </div>
            </div>
          `;
        }}
      />

      <GraphLegend />
    </div>
  );
};

export default ClusterMap2D;
