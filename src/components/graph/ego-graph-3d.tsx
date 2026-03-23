'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import ForceGraph3D, { ForceGraphMethods } from 'react-force-graph-3d';
import { SybilNode, SybilEdge } from '@/types/api';

interface EgoGraph3DProps {
  graphData: {
    nodes: SybilNode[];
    links: SybilEdge[];
  };
  targetId: string;
  classification?: 'BENIGN' | 'WARNING' | 'SYBIL';
}

const EgoGraph3D: React.FC<EgoGraph3DProps> = ({
  graphData,
  targetId,
  classification,
}) => {
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
    (node: any) => {
      if (node.id === targetId) return getTargetColor();
      return node.is_sybil ? '#f44336' : '#64748b'; // Red for other sybils, slate-500 for normal
    },
    [targetId, getTargetColor]
  );

  const getNodeVal = useCallback(
    (node: any) => {
      return node.id === targetId ? 8 : 2;
    },
    [targetId]
  );

  return (
    <div ref={containerRef} className="w-full h-full min-h-[400px]">
      <ForceGraph3D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        backgroundColor="rgba(0,0,0,0)" // Transparent
        nodeColor={getNodeColor}
        nodeLabel={(node: any) => `
          <div class="bg-black/90 border border-slate-700 p-2 font-mono text-[10px] uppercase">
            <div class="text-accent-cyan font-bold mb-1">${node.label || node.id}</div>
            <div class="flex justify-between gap-4">
              <span class="text-slate-500">TRUST_SCORE</span>
              <span class="${node.trust_score < 3 ? 'text-accent-red' : 'text-accent-green'}">${node.trust_score.toFixed(2)}</span>
            </div>
            ${node.is_sybil ? '<div class="text-accent-red font-bold mt-1">[SYBIL_DETECTED]</div>' : ''}
          </div>
        `}
        nodeVal={getNodeVal}
        nodeResolution={24}
        linkColor={() => '#1e293b'}
        linkWidth={0.5}
        linkDirectionalParticles={2}
        linkDirectionalParticleWidth={1.5}
        linkDirectionalParticleSpeed={0.005}
        linkDirectionalParticleColor={() => '#00f2ff'}
        showNavInfo={false}
        enableNodeDrag={false}
        onNodeClick={(node: any) => {
          // Aim at node from outside it
          const distance = 40;
          const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);

          if (fgRef.current) {
            fgRef.current.cameraPosition(
              { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }, // new position
              node, // lookAt ({ x, y, z })
              3000 // ms transition duration
            );
          }
        }}
      />
    </div>
  );
};

export default EgoGraph3D;
