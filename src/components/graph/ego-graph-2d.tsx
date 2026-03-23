'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { SybilNode, SybilEdge } from '@/types/api';

interface EgoGraph2DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
  targetId: string;
  classification?: 'BENIGN' | 'WARNING' | 'SYBIL';
}

const EgoGraph2D: React.FC<EgoGraph2DProps> = ({
  graphData,
  targetId,
  classification,
}) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null);
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
    if (classification === 'SYBIL' || classification === 'WARNING') {
      return '#ff1744'; // Neon Red
    }
    return '#00f2ff'; // Cyan
  }, [classification]);

  const getNodeColor = useCallback(
    (node: SybilNode) => {
      if (node.id === targetId) return getTargetColor();
      return node.is_sybil ? '#f44336' : '#64748b'; // Red for other sybils, slate-500 for normal
    },
    [targetId, getTargetColor]
  );

  const getNodeVal = useCallback(
    (node: SybilNode) => {
      return node.id === targetId ? 8 : 2;
    },
    [targetId]
  );

  return (
    <div ref={containerRef} className="w-full h-full min-h-[400px]">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)" // Transparent
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeColor={getNodeColor as any}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeLabel={(node: any) => `
          <div class="bg-black/90 border border-slate-700 p-2 font-mono text-[10px] uppercase">
            <div class="text-accent-cyan font-bold mb-1">${(node as SybilNode).label || (node as SybilNode).id}</div>
            <div class="flex justify-between gap-4">
              <span class="text-slate-500">TRUST_SCORE</span>
              <span class="${(node as SybilNode).trust_score < 3 ? 'text-accent-red' : 'text-accent-green'}">${(node as SybilNode).trust_score.toFixed(2)}</span>
            </div>
            ${(node as SybilNode).is_sybil ? '<div class="text-accent-red font-bold mt-1">[SYBIL_DETECTED]</div>' : ''}
          </div>
        `}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeVal={getNodeVal as any}
        linkColor={() => '#1e293b'}
        linkWidth={1}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleColor={() => '#00f2ff'}
        enableNodeDrag={true}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onNodeClick={(node: any) => {
          if (fgRef.current) {
            // Center and zoom into node
            fgRef.current.centerAt(node.x, node.y, 1000);
            fgRef.current.zoom(4, 1000);
          }
        }}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
          const label = (node as SybilNode).label || (node as SybilNode).id;
          const fontSize = 12 / globalScale;
          ctx.font = `${fontSize}px JetBrains Mono`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

          // Draw node
          const color = getNodeColor(node as SybilNode);
          ctx.beginPath();
          ctx.arc(node.x, node.y, getNodeVal(node as SybilNode), 0, 2 * Math.PI, false);
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
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2 - 8, bckgDimensions[0] as any, bckgDimensions[1] as any);

            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#000';
            ctx.fillText(label, node.x, node.y - 8);
          }
        }}
      />
    </div>
  );
};

export default EgoGraph2D;
