"use client";

import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
  useMemo,
} from "react";
import ForceGraph2D, {
  ForceGraphMethods,
  NodeObject,
  LinkObject,
} from "react-force-graph-2d";
import { SybilNode, SybilEdge } from "@/types/api";

// --- MULTIGRAPH SCHEMA ---
// Defines visual properties for each relationship type
const MULTIGRAPH_SCHEMA: Record<
  string,
  { color: string; dash?: number[]; directed: boolean; label: string }
> = {
  // Directed Layers (Social/Interaction)
  FOLLOW: { color: "#3b82f6", directed: true, label: "FOLLOW" },
  UPVOTE: { color: "#06b6d4", directed: true, label: "UPVOTE" },
  REACTION: { color: "#06b6d4", directed: true, label: "REACTION" },
  COMMENT: { color: "#10b981", directed: true, label: "COMMENT" },
  QUOTE: { color: "#8b5cf6", directed: true, label: "QUOTE" },
  MIRROR: { color: "#f59e0b", directed: true, label: "MIRROR" },
  COLLECT: { color: "#ec4899", directed: true, label: "COLLECT" },

  // Undirected Layers (Ownership/Similarity)
  "CO-OWNER": { color: "#ef4444", directed: false, label: "CO-OWNER" },
  SAME_AVATAR: {
    color: "#f43f5e",
    dash: [2, 2],
    directed: false,
    label: "SAME_AVATAR",
  },
  FUZZY_HANDLE: {
    color: "#fb923c",
    dash: [4, 2],
    directed: false,
    label: "FUZZY_HANDLE",
  },
  SIM_BIO: {
    color: "#fbbf24",
    dash: [1, 3],
    directed: false,
    label: "SIM_BIO",
  },
  CLOSE_CREATION_TIME: {
    color: "#a855f7",
    dash: [5, 5],
    directed: false,
    label: "CLOSE_CREATION_TIME",
  },

  UNKNOWN: { color: "#64748b", directed: false, label: "UNKNOWN" },
};

interface ExtendedNode extends SybilNode {
  __img?: HTMLImageElement;
}

interface ExtendedLink extends Omit<SybilEdge, "source" | "target"> {
  source: string | NodeObject<ExtendedNode>;
  target: string | NodeObject<ExtendedNode>;
  multiLinkIndex?: number;
  multiLinkCount?: number;
}

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
  const fgRef = useRef<
    ForceGraphMethods<ExtendedNode, ExtendedLink> | undefined
  >(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const imgCache = useRef<Record<string, HTMLImageElement>>({});

  // Process data for multi-link support
  const processedData = useMemo(() => {
    const nodes = graphData.nodes.map((n) => ({ ...n }) as ExtendedNode);
    const links = graphData.links.map((l) => ({ ...l }) as ExtendedLink);

    // Group links by pair
    const linkGroups: Record<string, ExtendedLink[]> = {};
    links.forEach((link) => {
      const id = [link.source, link.target].sort().join("-");
      if (!linkGroups[id]) linkGroups[id] = [];
      linkGroups[id].push(link);
    });

    // Assign indices for curvature
    Object.values(linkGroups).forEach((group) => {
      const count = group.length;
      group.forEach((link, i) => {
        link.multiLinkIndex = i;
        link.multiLinkCount = count;
      });
    });

    return { nodes, links };
  }, [graphData]);

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
    if (classification === "SYBIL" || classification === "WARNING")
      return "#ff1744";
    return "#00f2ff";
  }, [classification]);

  const getNodeColor = useCallback(
    (node: NodeObject<ExtendedNode>) => {
      if (node.id === targetId) return getTargetColor();
      return node.is_sybil ? "#f44336" : "#00f2ff";
    },
    [targetId, getTargetColor]
  );

  const drawNode = useCallback(
    (
      node: NodeObject<ExtendedNode>,
      ctx: CanvasRenderingContext2D,
      globalScale: number
    ) => {
      const size = node.id === targetId ? 10 : 6;
      const x = node.x ?? 0;
      const y = node.y ?? 0;
      const color = getNodeColor(node);

      // Draw Outer Glow for Target/Sybil
      if (node.id === targetId || node.is_sybil) {
        ctx.beginPath();
        ctx.arc(x, y, size + 2, 0, 2 * Math.PI, false);
        ctx.fillStyle = node.is_sybil
          ? "rgba(244, 67, 54, 0.2)"
          : "rgba(0, 242, 255, 0.2)";
        ctx.fill();
      }

      // Avatar Logic
      const imgUrl = node.attributes.picture_url;
      let img = null;
      if (imgUrl) {
        if (imgCache.current[imgUrl]) {
          img = imgCache.current[imgUrl];
        } else {
          const newImg = new Image();
          newImg.src = imgUrl;
          newImg.onload = () => {
            imgCache.current[imgUrl] = newImg;
            // Refresh graph if needed or just wait for next frame
          };
          imgCache.current[imgUrl] = newImg;
        }
      }

      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.clip();

      if (img && img.complete) {
        ctx.drawImage(img, x - size, y - size, size * 2, size * 2);
      } else {
        // Placeholder
        ctx.fillStyle = "#1e293b";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Industrial placeholder icon (stylized silhouette)
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y - size / 4, size / 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y + size, size, Math.PI, 2 * Math.PI);
        ctx.fill();
      }
      ctx.restore();

      // Node Border
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI, false);
      ctx.strokeStyle = color;
      ctx.lineWidth = node.id === targetId ? 2 : 1;
      ctx.stroke();

      // Label
      const label = node.label || (node.id as string);
      if (globalScale > 2) {
        ctx.font = `${10 / globalScale}px "JetBrains Mono", monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = color;
        ctx.fillText(label, x, y + size + 2);
      }
    },
    [targetId, getNodeColor]
  );

  return (
    <div ref={containerRef} className="relative h-full min-h-[400px] w-full">
      <ForceGraph2D
        ref={fgRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={processedData}
        backgroundColor="rgba(0,0,0,0)"
        nodeCanvasObject={drawNode}
        nodeCanvasObjectMode={() => "always"}
        // Link Rendering
        linkColor={(link: LinkObject<ExtendedNode, ExtendedLink>) =>
          MULTIGRAPH_SCHEMA[link.type]?.color ||
          MULTIGRAPH_SCHEMA["UNKNOWN"].color
        }
        linkWidth={(link: LinkObject<ExtendedNode, ExtendedLink>) =>
          link.multiLinkCount && link.multiLinkCount > 1 ? 1.5 : 1
        }
        linkCurvature={(link: LinkObject<ExtendedNode, ExtendedLink>) => {
          if (!link.multiLinkCount || link.multiLinkCount <= 1) return 0;
          const index = link.multiLinkIndex ?? 0;
          const count = link.multiLinkCount;
          return (index - (count - 1) / 2) * 0.15;
        }}
        // Directional Arrows
        linkDirectionalArrowLength={(
          link: LinkObject<ExtendedNode, ExtendedLink>
        ) => (MULTIGRAPH_SCHEMA[link.type]?.directed ? 3 : 0)}
        linkDirectionalArrowRelPos={0.5}
        // Particles for activity
        linkDirectionalParticles={(
          link: LinkObject<ExtendedNode, ExtendedLink>
        ) => (MULTIGRAPH_SCHEMA[link.type]?.directed ? 2 : 0)}
        linkDirectionalParticleWidth={1.5}
        linkDirectionalParticleSpeed={0.005}
        // Interactive tooltips
        nodeLabel={(node: NodeObject<ExtendedNode>) => `
          <div class="bg-slate-950 border border-slate-800 p-3 font-mono text-[10px] uppercase shadow-2xl">
            <div class="text-accent-cyan font-bold mb-2 flex items-center gap-2">
              <span class="w-2 h-2 rounded-full ${node.is_sybil ? "bg-accent-red" : "bg-accent-green"}"></span>
              ${node.label || node.id}
            </div>
            <div class="space-y-1">
              <div class="flex justify-between gap-8 text-slate-500">
                <span>TRUST_SCORE</span>
                <span class="${(node.trust_score ?? 0) < 3 ? "text-accent-red" : "text-accent-green"} font-bold">
                  ${(node.trust_score ?? 0).toFixed(2)}
                </span>
              </div>
              <div class="flex justify-between gap-8 text-slate-500">
                <span>REPOSTS</span>
                <span class="text-accent-cyan font-bold">${node.attributes.total_reposts ?? 0}</span>
              </div>
              ${node.is_sybil ? '<div class="text-accent-red font-black mt-2 border-t border-accent-red/20 pt-1 text-center animate-pulse">[SYBIL_WARNING]</div>' : ""}
            </div>
          </div>
        `}
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

      {/* Legend Overlay */}
      <div className="pointer-events-none absolute top-4 left-4 rounded-sm border border-slate-800 bg-black/60 p-3 backdrop-blur-md">
        <div className="mb-2 text-[8px] font-bold tracking-[0.2em] text-slate-500 uppercase">
          Relationship Layers
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(MULTIGRAPH_SCHEMA)
            .slice(0, 10)
            .map(([key, value]) => (
              <div key={key} className="flex items-center gap-2">
                <div
                  className="h-0.5 w-2"
                  style={{ backgroundColor: value.color }}
                ></div>
                <span className="font-mono text-[7px] tracking-widest text-slate-400">
                  {value.label}
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default EgoGraph2D;
