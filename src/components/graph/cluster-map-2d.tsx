"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge } from "@/types/api";

interface ClusterMap2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
}

// Neon color palette for clusters
const CLUSTER_COLORS = [
  "#00f2ff", // Cyan
  "#f43f5e", // Rose/Red
  "#8b5cf6", // Purple
  "#fb923c", // Orange
  "#4ade80", // Green
  "#e879f9", // Fuchsia
  "#22d3ee", // Sky
  "#facc15", // Yellow
];

const ClusterMap2D: React.FC<ClusterMap2DProps> = ({ graphData }) => {
  const fgRef = useRef<ForceGraphMethods<SybilNode, SybilEdge> | undefined>(
    undefined
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Update dimensions based on parent container
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

  const getNodeColor = useCallback((node: NodeObject<SybilNode>) => {
    if (node.is_high_risk) return "#ff1744"; // Vivid Red for high risk

    if (node.cluster_id !== undefined) {
      return CLUSTER_COLORS[node.cluster_id % CLUSTER_COLORS.length];
    }

    return "#475569"; // Slate-600 default
  }, []);

  return (
    <div ref={containerRef} className="relative h-full w-full bg-black/40">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)"
        nodeColor={getNodeColor}
        nodeVal={(node) => (node.is_high_risk ? 4 : 2)}
        linkColor={() => "rgba(71, 85, 105, 0.2)"} // Very subtle links
        linkWidth={0.5}
        // Performance optimizations for large graphs
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        // Hide labels for performance
        nodeLabel={(node: NodeObject<SybilNode>) => `
          <div class="bg-black/90 border border-slate-700 p-2 font-mono text-[10px] uppercase">
            <div class="text-accent-cyan font-bold mb-1">CLUSTER_${node.cluster_id}</div>
            <div class="text-slate-400">ID: ${node.id.slice(0, 8)}...</div>
            ${node.is_high_risk ? '<div class="text-accent-red font-bold mt-1">[HIGH_RISK_ENTITY]</div>' : ""}
          </div>
        `}
        // Draw glow for high risk nodes on canvas
        nodeCanvasObjectMode={() => "after"}
        nodeCanvasObject={(node, ctx, globalScale) => {
          if (node.is_high_risk) {
            const x = node.x ?? 0;
            const y = node.y ?? 0;
            ctx.beginPath();
            ctx.arc(x, y, 5 / globalScale, 0, 2 * Math.PI, false);
            ctx.fillStyle = "rgba(255, 23, 68, 0.3)";
            ctx.fill();
          }
        }}
      />

      {/* Mini Legend Overlay */}
      <div className="pointer-events-none absolute bottom-4 left-4 rounded-sm border border-slate-800 bg-black/60 p-2 backdrop-blur-md">
        <div className="mb-1 flex items-center gap-2">
          <div className="h-2 w-2 animate-pulse rounded-full bg-[#ff1744]"></div>
          <span className="font-mono text-[8px] tracking-widest text-slate-400 uppercase">
            High Risk Entity
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-[#00f2ff]"></div>
          <span className="font-mono text-[8px] tracking-widest text-slate-400 uppercase">
            Sybil Cluster
          </span>
        </div>
      </div>
    </div>
  );
};

export default ClusterMap2D;
