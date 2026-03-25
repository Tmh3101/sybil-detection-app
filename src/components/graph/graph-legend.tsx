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

const GraphLegend: React.FC<GraphLegendProps> = ({
  showNodes = true,
  showRelations = true,
  extraItems,
}) => {
  return (
    <div className="absolute top-6 right-6 z-10 flex min-w-[180px] flex-col gap-4 border border-slate-700 bg-black/80 p-4 shadow-2xl backdrop-blur-md">
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
          {RELATION_GROUPS.map(({ label, type }) => (
            <div key={type} className="flex items-center gap-3">
              <div
                className="h-0.5 w-3"
                style={{
                  backgroundColor:
                    RELATION_COLORS[type] || RELATION_COLORS.UNKNOWN,
                }}
              />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                {label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default GraphLegend;
