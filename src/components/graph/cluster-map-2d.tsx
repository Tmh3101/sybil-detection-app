"use client";

import React, { useCallback, useRef, useState, useEffect } from "react";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge } from "@/types/api";

interface ClusterMap2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  } | null;
}

const CLUSTER_COLORS = [
  "#00f2ff", // Cyan
  "#a855f7", // Purple
  "#f97316", // Orange
  "#22c55e", // Green
  "#eab308", // Yellow
  "#ec4899", // Pink
];

const ClusterMap2D: React.FC<ClusterMap2DProps> = ({ graphData }) => {
  const fgRef = useRef<ForceGraphMethods<SybilNode, SybilEdge> | undefined>(
    undefined
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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

  const getNodeColor = useCallback((node: NodeObject<SybilNode>) => {
    if (node.is_high_risk) return "#ff1744"; // Neon Red for high risk
    const colorIdx = (node.cluster_id || 0) % CLUSTER_COLORS.length;
    return CLUSTER_COLORS[colorIdx];
  }, []);

  if (!graphData) return null;

  return (
    <div ref={containerRef} className="h-full w-full">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)"
        nodeColor={getNodeColor}
        nodeLabel={undefined} // Disable labels for performance
        nodeRelSize={4}
        linkColor={() => "rgba(71, 85, 105, 0.2)"} // Subdued link color
        linkWidth={0.5}
        enableNodeDrag={true}
        cooldownTicks={100}
        onEngineStop={() => {
          if (fgRef.current) {
            fgRef.current.zoomToFit(400);
          }
        }}
        nodeCanvasObject={(
          node: NodeObject<SybilNode>,
          ctx: CanvasRenderingContext2D,
          globalScale: number
        ) => {
          const color = getNodeColor(node);
          const r = node.is_high_risk ? 4 : 2;

          const x = node.x ?? 0;
          const y = node.y ?? 0;

          ctx.beginPath();
          ctx.arc(x, y, r, 0, 2 * Math.PI, false);
          ctx.fillStyle = color;
          ctx.fill();

          if (node.is_high_risk) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 10;
            ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
            ctx.lineWidth = 1 / globalScale;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }
        }}
      />
    </div>
  );
};

export default ClusterMap2D;
