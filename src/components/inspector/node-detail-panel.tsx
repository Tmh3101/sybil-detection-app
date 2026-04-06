"use client";

import React from "react";
import { SybilNode } from "@/types/api";
import { LABEL_COLORS } from "@/lib/graph-constants";
import {
  X,
  User,
  ShieldCheck,
  Shield,
  ShieldAlert,
  AlertTriangle,
} from "lucide-react";
import { resolvePictureUrl } from "@/lib/utils";
import Image from "next/image";
import { useTranslations } from "next-intl";

interface NodeDetailPanelProps {
  node: SybilNode;
  onClose: () => void;
}

const RISK_ICONS: Record<string, React.ReactNode> = {
  MALICIOUS: <AlertTriangle size={14} />,
  HIGH_RISK: <ShieldAlert size={14} />,
  LOW_RISK: <Shield size={14} />,
  BENIGN: <ShieldCheck size={14} />,
};

const NodeDetailPanel: React.FC<NodeDetailPanelProps> = ({ node, onClose }) => {
  const t = useTranslations("DetailPanels.node");
  const rl = node.risk_label || "UNKNOWN";
  const color = LABEL_COLORS[rl] || LABEL_COLORS.UNKNOWN;
  const pictureUrl = node.attributes?.picture_url
    ? resolvePictureUrl(String(node.attributes.picture_url))
    : "";
  const handle = String(
    node.attributes?.handle || node.id || t("unknown_handle")
  );

  return (
    <div className="flex h-full flex-col overflow-hidden border-l border-slate-800/80 bg-[#050810] font-mono shadow-2xl">
      {/* ── Header ── */}
      <div className="flex flex-shrink-0 items-center justify-between border-b border-slate-800/80 px-4 py-4">
        <div className="flex items-center gap-3">
          <div
            className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-sm border-2"
            style={{
              borderColor: color + "44",
              backgroundColor: color + "0d",
            }}
          >
            {pictureUrl ? (
              <Image
                src={pictureUrl}
                alt={handle}
                width={40}
                height={40}
                className="h-full w-full object-cover"
                unoptimized
              />
            ) : (
              <User size={20} style={{ color }} />
            )}
          </div>
          <div className="flex flex-col">
            <span
              className="text-sm font-black tracking-tighter uppercase italic"
              style={{ color }}
            >
              {handle}
            </span>
            <span className="text-[9px] text-slate-500 tabular-nums">
              {node.id}
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
        {/* Risk Status */}
        <div
          className="mb-6 flex items-center justify-between border p-3"
          style={{
            borderColor: color + "33",
            backgroundColor: color + "08",
          }}
        >
          <div className="flex items-center gap-2">
            <div style={{ color }}>{RISK_ICONS[rl]}</div>
            <span
              className="text-xs font-bold tracking-widest uppercase"
              style={{ color }}
            >
              {rl.replace("_", " ")}
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[8px] text-slate-500 uppercase">
              {t("risk_score_label")}
            </span>
            <span className="text-sm font-black tabular-nums" style={{ color }}>
              {(node.risk_score * 100).toFixed(0)}
            </span>
          </div>
        </div>

        {/* Stats Grid */}
        {/* <div className="mb-6 grid grid-cols-2 gap-3">
          <div className="border border-slate-800 bg-slate-900/20 p-2.5">
            <div className="mb-1 flex items-center gap-1.5 text-slate-500">
              <Users size={10} />
              <span className="text-[8px] font-bold uppercase">Followers</span>
            </div>
            <span className="text-xs font-bold text-slate-200 tabular-nums">
              {node.attributes?.follower_count?.toLocaleString() ?? "0"}
            </span>
          </div>
          <div className="border border-slate-800 bg-slate-900/20 p-2.5">
            <div className="mb-1 flex items-center gap-1.5 text-slate-500">
              <MessageSquare size={10} />
              <span className="text-[8px] font-bold uppercase">Posts</span>
            </div>
            <span className="text-xs font-bold text-slate-200 tabular-nums">
              {node.attributes?.post_count?.toLocaleString() ?? "0"}
            </span>
          </div>
          <div className="border border-slate-800 bg-slate-900/20 p-2.5">
            <div className="mb-1 flex items-center gap-1.5 text-slate-500">
              <Activity size={10} />
              <span className="text-[8px] font-bold uppercase">Trust</span>
            </div>
            <span className="text-xs font-bold text-slate-200 tabular-nums">
              {Number(node.attributes?.trust_score || 0).toFixed(1)}
            </span>
          </div>
          <div className="border border-slate-800 bg-slate-900/20 p-2.5">
            <div className="mb-1 flex items-center gap-1.5 text-slate-500">
              <Calendar size={10} />
              <span className="text-[8px] font-bold uppercase">Age</span>
            </div>
            <span className="truncate text-xs font-bold text-slate-200 uppercase">
              {String(node.attributes?.account_age || "N/A")}
            </span>
          </div>
        </div> */}

        {/* Detection Reasons */}
        {node.attributes?.reason && (
          <div className="mb-6 flex flex-col gap-2">
            <span className="text-[8px] font-bold tracking-widest text-slate-500 uppercase">
              {t("primary_reason")}
            </span>
            <div className="flex flex-col border border-slate-800 bg-slate-900/40 px-2 py-1.5">
              <span className="text-[10px] leading-tight text-slate-300">
                {String(node.attributes.reason)}
              </span>
            </div>
          </div>
        )}

        {((node.attributes?.reasons as unknown as string[]) || []).length >
          0 && (
          <div className="flex flex-col gap-2">
            <span className="text-[8px] font-bold tracking-widest text-slate-500 uppercase">
              {t("detection_flags")}
            </span>
            <div className="flex flex-col gap-1.5">
              {(node.attributes.reasons as unknown as string[]).map((r, i) => {
                const penaltyMatch = r.match(/\+(\d+)$/);
                const penalty = penaltyMatch ? penaltyMatch[0] : null;
                const baseText = penalty
                  ? r.slice(0, -penalty.length).trim()
                  : r;
                const parts = baseText.includes(":")
                  ? baseText.split(":")
                  : [baseText];
                const title = parts[0].trim();
                const description =
                  parts.length > 1 ? parts.slice(1).join(":").trim() : null;

                return (
                  <div
                    key={i}
                    className="flex flex-col border border-slate-800 bg-slate-900/40 px-2 py-1.5"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className="text-[9px] font-bold uppercase"
                        style={{ color: color + "cc" }}
                      >
                        {title}
                      </span>
                      {penalty && (
                        <span className="animate-pulse text-[10px] font-black text-red-500 tabular-nums">
                          {penalty}
                        </span>
                      )}
                    </div>
                    {description && (
                      <span className="mt-0.5 text-[8px] leading-tight text-slate-500">
                        {description}
                      </span>
                    )}
                    {!description && !penaltyMatch && (
                      <span className="mt-0.5 text-[8px] leading-tight text-slate-500">
                        {r}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* ── Footer ── */}
      <div className="border-t border-slate-800/60 px-4 py-3">
        <div className="flex items-center justify-between">
          <span className="text-[8px] font-bold tracking-widest text-slate-600 uppercase">
            {t("cluster_id", { id: node.cluster_id ?? "N/A" })}
          </span>
          <span className="text-[8px] text-slate-700 tabular-nums">
            {t("id_prefix", { id: (node.id || "000").slice(-4).toUpperCase() })}
          </span>
        </div>
      </div>
    </div>
  );
};

export default NodeDetailPanel;
