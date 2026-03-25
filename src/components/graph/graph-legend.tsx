import React from "react";
import {
  LABEL_COLORS,
  RELATION_COLORS,
  RELATION_GROUPS,
  LABEL_GROUPS,
} from "@/lib/graph-constants";

interface GraphLegendProps {
  showNodes?: boolean;
  showRelations?: boolean;
  extraItems?: React.ReactNode;
}

// // Arrow icon for directed edges
// const DirectedIcon = () => (
//   <span className="ml-1 text-[8px] text-slate-500" title="Directed (one-way)">
//     →
//   </span>
// );

// // Double-line icon for undirected
// const UndirectedIcon = () => (
//   <span className="ml-1 text-[8px] text-slate-500" title="Undirected (two-way)">
//     ↔
//   </span>
// );

// const DIRECTED_TYPES = new Set(["CO-OWNER", "FOLLOW", "INTERACT"]);
// // Note: CO-OWNER is actually undirected, FOLLOW/INTERACT are directed
// const UNDIRECTED_TYPES = new Set(["CO-OWNER", "SIMILARITY"]);

const GraphLegend: React.FC<GraphLegendProps> = ({
  showNodes = true,
  showRelations = true,
  extraItems,
}) => {
  return (
    <div className="absolute top-6 right-6 z-10 flex min-w-[190px] flex-col gap-4 border border-slate-700/80 bg-black/85 p-4 shadow-2xl backdrop-blur-md">
      {showNodes && (
        <div className="flex flex-col gap-2">
          <div className="mb-1 text-[8px] font-bold tracking-[0.2em] text-slate-500 uppercase">
            Node Map
          </div>
          {extraItems}
          {LABEL_GROUPS.map(({ label, key }) => (
            <div key={key} className="flex items-center gap-3">
              <div
                className={`h-2 w-2 rounded-full ${key === "MALICIOUS" ? "animate-pulse shadow-[0_0_8px_#ef4444]" : ""}`}
                style={{ backgroundColor: LABEL_COLORS[key] }}
              />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                {label}
              </span>
            </div>
          ))}
        </div>
      )}

      {showRelations && (
        <div
          className={`flex flex-col gap-2 ${showNodes ? "border-t border-slate-800 pt-3" : ""}`}
        >
          <div className="mb-1 text-[8px] font-bold tracking-[0.2em] text-slate-500 uppercase">
            Relation Layers
          </div>
          {RELATION_GROUPS.map(({ label, type }) => {
            const isDirected = type === "FOLLOW" || type === "INTERACT";
            const isUndirected = type === "CO-OWNER" || type === "SIMILARITY";
            return (
              <div key={type} className="flex items-center gap-2">
                <div
                  className="h-0.5 w-4 flex-shrink-0"
                  style={{
                    backgroundColor:
                      RELATION_COLORS[type] || RELATION_COLORS.UNKNOWN,
                  }}
                />
                {/* Arrow indicator for directed edges */}
                {isDirected && (
                  <div
                    className="flex-shrink-0"
                    style={{
                      color: RELATION_COLORS[type] || RELATION_COLORS.UNKNOWN,
                    }}
                  >
                    <svg width="8" height="8" viewBox="0 0 8 8">
                      <polygon points="0,0 8,4 0,8" fill="currentColor" />
                    </svg>
                  </div>
                )}
                {isUndirected && (
                  <span
                    className="flex-shrink-0 text-[8px]"
                    style={{
                      color: RELATION_COLORS[type] || RELATION_COLORS.UNKNOWN,
                    }}
                  >
                    ◇
                  </span>
                )}
                <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                  {label
                    .replace(" (directed)", "")
                    .replace(" (undirected)", "")}
                </span>
              </div>
            );
          })}
          {/* Direction key */}
          <div className="mt-1 flex flex-col gap-1 border-t border-slate-800/60 pt-2">
            <div className="flex items-center gap-2">
              <svg
                width="8"
                height="8"
                viewBox="0 0 8 8"
                className="flex-shrink-0"
              >
                <polygon points="0,0 8,4 0,8" fill="#64748b" />
              </svg>
              <span className="font-mono text-[8px] text-slate-600">
                directed
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex-shrink-0 text-[8px] text-slate-600">◇</span>
              <span className="font-mono text-[8px] text-slate-600">
                undirected
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphLegend;
