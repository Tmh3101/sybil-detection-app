"use client";

import React from "react";
import { AggregatedLink } from "@/hooks/use-graph-processor";
import { RELATION_COLORS } from "@/lib/graph-constants";
import { X, Link as LinkIcon, AlertCircle, Info } from "lucide-react";
import { NodeObject } from "react-force-graph-2d";
import { SybilNode } from "@/types/api";
import { useTranslations } from "next-intl";

interface EdgeDetailPanelProps {
  link: AggregatedLink;
  onClose: () => void;
}

const EdgeDetailPanel: React.FC<EdgeDetailPanelProps> = ({ link, onClose }) => {
  const t = useTranslations("DetailPanels.edge");
  const type = (link.edge_type as string) || "UNKNOWN";
  const color = RELATION_COLORS[type] || RELATION_COLORS.UNKNOWN;
  const weight = link.aggregated_weight || 1;
  const violations = link.violations || [];

  // Extract source and target IDs
  const sourceId =
    typeof link.source === "object"
      ? (link.source as NodeObject<SybilNode>).id
      : String(link.source);
  const targetId =
    typeof link.target === "object"
      ? (link.target as NodeObject<SybilNode>).id
      : String(link.target);

  return (
    <div className="flex h-full flex-col overflow-hidden border-l border-slate-800/80 bg-[#050810] font-mono shadow-2xl">
      {/* ── Header ── */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-slate-800/80 px-4 py-4">
        <div className="flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center rounded-sm border-2"
            style={{
              borderColor: color + "44",
              backgroundColor: color + "0d",
            }}
          >
            <LinkIcon size={20} style={{ color }} />
          </div>
          <div className="flex flex-col">
            <span
              className="text-sm font-black tracking-tighter uppercase italic"
              style={{ color }}
            >
              {type.replace("_", " ")}
            </span>
            <span className="text-[9px] text-slate-500 uppercase">
              {t("relationship_detail")}
            </span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="hover:border-accent-red/40 hover:text-accent-red flex h-8 w-8 items-center justify-center border border-slate-800 text-slate-500 transition-all active:scale-95"
        >
          <X size={14} />
        </button>
      </div>

      {/* ── Body ── */}
      <div className="scrollbar-thin flex-1 overflow-y-auto p-4">
        {/* Connection Info */}
        <div className="mb-6 flex flex-col gap-3">
          <div className="flex flex-col gap-1 border border-slate-800/50 bg-slate-900/20 p-3">
            <span className="text-[8px] font-bold tracking-widest text-slate-500 uppercase">
              {t("source_label")}
            </span>
            <span className="text-[10px] break-all text-slate-300">
              {sourceId}
            </span>
          </div>
          <div className="flex justify-center">
            <div className="h-4 w-[1px] bg-slate-800" />
          </div>
          <div className="flex flex-col gap-1 border border-slate-800/50 bg-slate-900/20 p-3">
            <span className="text-[8px] font-bold tracking-widest text-slate-500 uppercase">
              {t("target_label")}
            </span>
            <span className="text-[10px] break-all text-slate-300">
              {targetId}
            </span>
          </div>
        </div>

        {/* Weight Status */}
        <div
          className="mb-6 flex items-center justify-between border p-3"
          style={{
            borderColor: color + "33",
            backgroundColor: color + "08",
          }}
        >
          <div className="flex items-center gap-2">
            <Info size={14} style={{ color }} />
            <span
              className="text-xs font-bold tracking-widest uppercase"
              style={{ color }}
            >
              {t("intensity_label")}
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[8px] text-slate-500 uppercase">
              {t("weight_label")}
            </span>
            <span className="text-sm font-black tabular-nums" style={{ color }}>
              {weight.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Violations / Constraints (Ràng buộc 2/3) */}
        {type === "SIMILARITY" && (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <AlertCircle size={12} className="text-accent-red" />
              <span className="text-[9px] font-bold tracking-widest text-slate-400 uppercase">
                {t("heuristic_violations")}
              </span>
            </div>

            {violations.length > 0 ? (
              <div className="flex flex-col gap-1.5">
                {violations.map((v, i) => (
                  <div
                    key={i}
                    className="border-accent-red/20 bg-accent-red/5 flex items-center gap-2 border px-2 py-2"
                  >
                    <div className="bg-accent-red h-1 w-1 animate-pulse rounded-full" />
                    <span className="text-accent-red/90 text-[9px] font-bold tracking-tight uppercase">
                      {v.replace("_", " ")}
                    </span>
                  </div>
                ))}
                <div className="mt-2 border-t border-slate-800 pt-2">
                  <p className="text-[8px] leading-relaxed text-slate-500 uppercase italic">
                    {t("similarity_constraints_info")}
                  </p>
                </div>
              </div>
            ) : (
              <div className="border border-slate-800 bg-slate-900/20 px-3 py-4 text-center">
                <span className="text-[9px] text-slate-600 uppercase italic">
                  {t("no_violation_metadata")}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Other Relationship Info */}
        {type !== "SIMILARITY" && (
          <div className="border border-slate-800 bg-slate-900/20 p-4 text-center">
            <p className="text-[9px] leading-relaxed text-slate-500 uppercase">
              {t.rich("direct_relationship_info", {
                type: type,
                highlight: (chunks) => <span style={{ color }}>{chunks}</span>,
              })}
            </p>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div className="border-t border-slate-800/60 px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-[8px] font-bold tracking-widest text-slate-600 uppercase italic">
            {t("logic_info")}
          </span>
          <span className="text-[8px] text-slate-700 tabular-nums">
            {t("link_prefix", { type: type.slice(0, 3).toUpperCase() })}
          </span>
        </div>
      </div>
    </div>
  );
};

export default EdgeDetailPanel;
