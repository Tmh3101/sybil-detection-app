"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge } from "@/types/api";

interface EgoGraph2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
  targetId: string;
  classification?: "BENIGN" | "WARNING" | "SYBIL";
}

const EgoGraph2D: React.FC<EgoGraph2DProps> = ({
  graphData,
  targetId,
  classification,
}) => {
  const fgRef = useRef<ForceGraphMethods<SybilNode, SybilEdge> | undefined>(
    undefined
  );
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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

  const getTargetColor = useCallback(() => {
    if (classification === "SYBIL" || classification === "WARNING") {
      return "#ff1744"; // Neon Red
    }
    return "#00f2ff"; // Cyan
  }, [classification]);

  const getNodeColor = useCallback(
    (node: NodeObject<SybilNode>) => {
      if (node.id === targetId) return getTargetColor();
      return node.is_sybil ? "#f44336" : "#64748b"; // Red for other sybils, slate-500 for normal
    },
    [targetId, getTargetColor]
  );

  const getNodeVal = useCallback(
    (node: NodeObject<SybilNode>) => {
      return node.id === targetId ? 8 : 2;
    },
    [targetId]
  );

  return (
    <div ref={containerRef} className="h-full min-h-[400px] w-full">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)" // Transparent
        nodeColor={getNodeColor}
        nodeLabel={(node: NodeObject<SybilNode>) => `
          <div class="bg-black/90 border border-slate-700 p-2 font-mono text-[10px] uppercase">
            <div class="text-accent-cyan font-bold mb-1">${node.label || node.id}</div>
            <div class="flex justify-between gap-4">
              <span class="text-slate-500">TRUST_SCORE</span>
              <span class="${(node.trust_score ?? 0) < 3 ? "text-accent-red" : "text-accent-green"}">${(node.trust_score ?? 0).toFixed(2)}</span>
            </div>
            ${node.is_sybil ? '<div class="text-accent-red font-bold mt-1">[SYBIL_DETECTED]</div>' : ""}
          </div>
        `}
        nodeVal={getNodeVal}
        linkColor={() => "#1e293b"}
        linkWidth={1}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleColor={() => "#00f2ff"}
        enableNodeDrag={true}
        onNodeClick={(node: NodeObject<SybilNode>) => {
          if (fgRef.current && node.x !== undefined && node.y !== undefined) {
            // Center and zoom into node
            fgRef.current.centerAt(node.x, node.y, 1000);
            fgRef.current.zoom(4, 1000);
          }
        }}
        nodeCanvasObject={(
          node: NodeObject<SybilNode>,
          ctx: CanvasRenderingContext2D,
          globalScale: number
        ) => {
          const label = node.label || (node.id as string);
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px JetBrains Mono`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(
            (n) => n + fontSize * 0.2
          ); // some padding

          const x = node.x ?? 0;
          const y = node.y ?? 0;

          // Draw node
          const color = getNodeColor(node);
          ctx.beginPath();
          ctx.arc(x, y, getNodeVal(node), 0, 2 * Math.PI, false);
          ctx.fillStyle = color;
          ctx.fill();

          // Add shadow/glow for target
          if (node.id === targetId) {
            ctx.shadowColor = color;
            ctx.shadowBlur = 15;
            ctx.stroke();
            ctx.shadowBlur = 0;
          }

          // Draw label if zoomed in
          if (globalScale > 3) {
            ctx.fillStyle = "rgba(15, 23, 42, 0.9)"; // Slate-900 with alpha
            ctx.fillRect(
              x - bckgDimensions[0] / 2,
              y - bckgDimensions[1] / 2 - 12,
              bckgDimensions[0],
              bckgDimensions[1]
            );

            // Add border to label background
            ctx.strokeStyle = color;
            ctx.lineWidth = 1 / globalScale;
            ctx.strokeRect(
              x - bckgDimensions[0] / 2,
              y - bckgDimensions[1] / 2 - 12,
              bckgDimensions[0],
              bckgDimensions[1]
            );

            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = color; // Use node color (Cyan or Red)
            ctx.fillText(label, x, y - 12);
          }
        }}
      />
    </div>
  );
};

export default EgoGraph2D;
