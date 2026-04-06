import React, { useMemo } from "react";
import { SybilNode, SybilEdge } from "@/types/api";
import {
  LABEL_COLORS,
  LABEL_GROUPS,
  EDGE_LAYERS,
  computeEdgeCounts,
} from "@/lib/graph-constants";
import { useTranslations } from "next-intl";

interface GraphLegendProps {
  showNodes?: boolean;
  showRelations?: boolean;
  extraItems?: React.ReactNode;
  graphData?: { nodes: SybilNode[]; links: SybilEdge[] };
  visibleLayers?: string[];
  onToggleLayer?: (layer: string) => void;
}

const GraphLegend: React.FC<GraphLegendProps> = ({
  showNodes = true,
  showRelations = true,
  extraItems,
  graphData,
  visibleLayers = [],
  onToggleLayer,
}) => {
  const t = useTranslations("GraphLegend");

  // Edge counts per layer
  const edgeCounts = useMemo(() => {
    if (!graphData?.links) return {};
    return computeEdgeCounts(graphData.links);
  }, [graphData]);

  // Node counts per risk label
  const nodeCounts = useMemo(() => {
    if (!graphData?.nodes) return {};
    const c: Record<string, number> = {};
    for (const n of graphData.nodes) {
      const k = n.risk_label || "UNKNOWN";
      c[k] = (c[k] || 0) + 1;
    }
    return c;
  }, [graphData]);

  const totalEdges = graphData?.links?.length ?? 0;
  const totalNodes = graphData?.nodes?.length ?? 0;

  return (
    <div className="absolute top-6 right-6 z-10 flex min-w-[200px] flex-col gap-3 border border-slate-700/70 bg-black/88 p-4 shadow-2xl backdrop-blur-md">
      {/* ── Node map ── */}
      {showNodes && (
        <div className="flex flex-col gap-1.5">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-mono text-[8px] font-bold tracking-[0.18em] text-slate-500 uppercase">
              {t("node_map")}
            </span>
            {totalNodes > 0 && (
              <span className="font-mono text-[8px] text-slate-600">
                {t("nodes_count", { count: totalNodes })}
              </span>
            )}
          </div>

          {extraItems}

          {LABEL_GROUPS.map(({ label, key }) => {
            const count = nodeCounts[key] ?? 0;
            return (
              <div
                key={key}
                className="flex items-center justify-between gap-2"
              >
                <div className="flex items-center gap-2">
                  <div
                    className={`h-2 w-2 flex-shrink-0 rounded-full ${key === "MALICIOUS" ? "animate-pulse" : ""}`}
                    style={{
                      backgroundColor: LABEL_COLORS[key],
                      boxShadow:
                        key === "MALICIOUS"
                          ? `0 0 6px ${LABEL_COLORS[key]}99`
                          : "none",
                    }}
                  />
                  <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                    {label}
                  </span>
                </div>
                {count > 0 && (
                  <span
                    className="font-mono text-[8px] font-bold tabular-nums"
                    style={{ color: LABEL_COLORS[key] + "aa" }}
                  >
                    {count}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── Relation layers ── */}
      {showRelations && (
        <div
          className={`flex flex-col gap-1.5 ${
            showNodes ? "border-t border-slate-800/80 pt-3" : ""
          }`}
        >
          <div className="mb-1 flex items-center justify-between">
            <span className="font-mono text-[8px] font-bold tracking-[0.18em] text-slate-500 uppercase">
              {t("relation_layers")}
            </span>
            {totalEdges > 0 && (
              <span className="font-mono text-[8px] text-slate-600">
                {t("edges_count", { count: totalEdges })}
              </span>
            )}
          </div>

          {EDGE_LAYERS.map((layer) => {
            const count = edgeCounts[layer.key] ?? 0;
            const isVisible =
              visibleLayers.length === 0 || visibleLayers.includes(layer.key);

            return (
              <button
                key={layer.key}
                onClick={() => onToggleLayer?.(layer.key)}
                className={`flex items-center justify-between gap-2 transition-all hover:translate-x-0.5 ${
                  !isVisible ? "opacity-30 grayscale-[0.5]" : "opacity-100"
                }`}
              >
                <div className="flex items-center gap-2 text-left">
                  {/* Line + direction indicator */}
                  <div className="flex flex-shrink-0 items-center gap-0.5">
                    <div
                      className="h-px w-4"
                      style={{ backgroundColor: layer.color }}
                    />
                    {layer.directed ? (
                      /* Arrow for directed layers */
                      <svg
                        width="5"
                        height="6"
                        viewBox="0 0 5 6"
                        style={{ color: layer.color }}
                      >
                        <polygon points="0,0 5,3 0,6" fill="currentColor" />
                      </svg>
                    ) : (
                      /* Diamond for undirected layers */
                      <svg
                        width="6"
                        height="6"
                        viewBox="0 0 6 6"
                        style={{ color: layer.color + "99" }}
                      >
                        <polygon
                          points="3,0 6,3 3,6 0,3"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1"
                        />
                      </svg>
                    )}
                  </div>
                  <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                    {layer.label}
                  </span>
                </div>

                {/* Edge count badge */}
                {count > 0 ? (
                  <span
                    className="rounded-sm px-1.5 py-0.5 font-mono text-[8px] font-bold tabular-nums"
                    style={{
                      backgroundColor: layer.color + "18",
                      color: layer.color + "cc",
                      border: `1px solid ${layer.color}33`,
                    }}
                  >
                    {count}
                  </span>
                ) : (
                  <span className="font-mono text-[8px] text-slate-700">—</span>
                )}
              </button>
            );
          })}

          {/* Direction key */}
          <div className="mt-1.5 flex flex-col gap-1 border-t border-slate-800/60 pt-2">
            <div className="flex items-center gap-2">
              <svg width="14" height="8" viewBox="0 0 14 8">
                <line
                  x1="0"
                  y1="4"
                  x2="8"
                  y2="4"
                  stroke="#334155"
                  strokeWidth="1"
                />
                <polygon points="8,1.5 13,4 8,6.5" fill="#334155" />
              </svg>
              <span className="font-mono text-[7px] text-slate-600">
                {t("directed")}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <svg width="14" height="8" viewBox="0 0 14 8">
                <line
                  x1="0"
                  y1="4"
                  x2="14"
                  y2="4"
                  stroke="#334155"
                  strokeWidth="1"
                />
                <polygon
                  points="7,1.5 12,4 7,6.5 2,4"
                  fill="none"
                  stroke="#334155"
                  strokeWidth="1"
                />
              </svg>
              <span className="font-mono text-[7px] text-slate-600">
                {t("undirected")}
              </span>
            </div>

            {/* AI Attention Focus Key */}
            <div className="mt-1 flex items-center gap-2">
              <div className="relative flex h-2 w-14 items-center justify-center overflow-hidden">
                <div className="absolute h-[1px] w-full bg-[#ef4444] opacity-40"></div>
                <div className="absolute h-[4px] w-[4px] animate-ping rounded-full bg-[#ef4444] shadow-[0_0_8px_#ef4444]"></div>
              </div>
              <span className="font-mono text-[7px] font-bold text-[#ef4444]">
                {t("ai_focus")}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphLegend;
