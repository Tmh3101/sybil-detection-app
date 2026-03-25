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

interface ExtendedNode extends SybilNode {
  __img?: HTMLImageElement;
}

interface EgoGraph2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
  targetId: string;
  risk_label?: RiskClassification;
}

const EgoGraph2D: React.FC<EgoGraph2DProps> = ({
  graphData,
  targetId,
  risk_label,
}) => {
  const fgRef = useRef<
    ForceGraphMethods<ExtendedNode, AggregatedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [imagesLoaded, setImagesLoaded] = useState(0); // Trigger re-render when images load
  const imgCache = useRef<Record<string, HTMLImageElement>>({});

  // Use the standardized graph processor hook
  const processedData = useGraphProcessor(graphData, {
    targetId,
    aggregateEdges: true,
  });

  // Update dimensions based on parent container
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

  // Physics Tuning
  useEffect(() => {
    if (fgRef.current) {
      fgRef.current.d3Force("radial", d3.forceRadial(150, 0, 0));
      const charge = fgRef.current.d3Force("charge");
      if (charge) {
        (charge as d3.ForceManyBody<ExtendedNode>).strength(-200);
      }
      // Re-heat simulation to apply changes
      fgRef.current.d3ReheatSimulation();
    }
  }, [processedData]);

  const getTargetColor = useCallback(() => {
    if (risk_label === "MALICIOUS" || risk_label === "HIGH_RISK")
      return LABEL_COLORS["MALICIOUS"];
    return LABEL_COLORS["BENIGN"];
  }, [risk_label]);

  const getNodeColor = useCallback((node: NodeObject<ExtendedNode>) => {
    return (
      (node.risk_label && LABEL_COLORS[node.risk_label]) || LABEL_COLORS.UNKNOWN
    );
  }, []);

  const drawNode = useCallback(
    (
      node: NodeObject<ExtendedNode>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const isTarget = node.id === targetId;
      const size = isTarget ? 10 : 6;
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const color = isTarget ? getTargetColor() : getNodeColor(node);

      // 1. Draw Glow (Target only in EgoGraph)
      if (isTarget) {
        ctx.beginPath();
        ctx.arc(x, y, size + 4, 0, 2 * Math.PI, false);
        ctx.fillStyle = `${color}33`; // 0.2 opacity
        ctx.fill();
      }

      // 2. Avatar Logic (Ported from ClusterMap2D)
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

      // Draw Node Base (Clip for Avatar)
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
        ctx.fillStyle = "#1e293b";
        ctx.fill();
      }
      ctx.restore();

      // 3. Draw Border (Stroke)
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.strokeStyle = color;
      ctx.lineWidth = isTarget ? 2 : 1;
      ctx.stroke();

      // 4. Label (Handle)
      const label = node.attributes?.handle || (node.id as string).slice(0, 8);
      if (globalScale > 2) {
        ctx.font = `${10 / globalScale}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(label, x, y + size + 2);
      }
    },
    [targetId, getNodeColor, getTargetColor]
  );

  return (
    <div ref={containerRef} className="relative h-full min-h-[400px] w-full">
      <ForceGraph2D
        key={`fg-${imagesLoaded}`}
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "always"}
        // Link Rendering
        linkColor={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          const relationType = link.edge_type;
          const baseColor =
            (relationType && RELATION_COLORS[relationType as string]) ||
            RELATION_COLORS.UNKNOWN;

          const weight = link.aggregated_weight || 1;
          const opacity = Math.min(0.4 + Math.log10(weight) * 0.2, 0.8);

          const r = parseInt(baseColor.slice(1, 3), 16);
          const g = parseInt(baseColor.slice(3, 5), 16);
          const b = parseInt(baseColor.slice(5, 7), 16);

          return `rgba(${r}, ${g}, ${b}, ${opacity})`;
        }}
        linkWidth={(link: LinkObject<ExtendedNode, AggregatedLink>) => {
          const weight = link.aggregated_weight || 1;
          return Math.max(MIN_LINK_WIDTH, Math.sqrt(weight));
        }}
        linkDirectionalParticles={(
          link: LinkObject<ExtendedNode, AggregatedLink>
        ) => {
          const weight = link.aggregated_weight || 1;
          return weight > 1
            ? Math.min(Math.floor(Math.log2(weight)) + 1, 5)
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
        // Interactive tooltips
        nodeLabel={(node: NodeObject<ExtendedNode>) => {
          const isTarget = node.id === targetId;
          const isHighRisk =
            node.risk_label === "MALICIOUS" || node.risk_label === "HIGH_RISK";
          return `
            <div class="bg-black/95 border border-slate-700 p-3 font-mono text-[10px] shadow-2xl min-w-[200px]">
              <div class="flex items-center justify-between mb-1">
                <div class="text-accent-cyan font-bold text-xs">
                  ${node.attributes?.handle || "Unknown Handle"}
                  ${isTarget ? '<span class="ml-2 text-[8px] px-1 bg-accent-cyan/20 border border-accent-cyan/50 animate-pulse text-accent-cyan">[TARGET_ENTITY]</span>' : ""}
                </div>
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
          <div className="mb-1 flex items-center gap-3">
            <div className="h-3 w-3 animate-pulse rounded-full bg-[#00f2ff] shadow-[0_0_8px_rgba(0,242,255,0.6)]" />
            <span className="text-accent-cyan font-mono text-[9px] font-bold uppercase italic">
              Target Node
            </span>
          </div>
        }
      />
    </div>
  );
};

export default EgoGraph2D;
