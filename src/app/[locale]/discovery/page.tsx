"use client";

import { useState, useMemo, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { TerminalLog } from "@/components/ui/terminal-log";
import {
  Play,
  Calendar,
  Filter,
  Database,
  Loader2,
  ChevronDown,
  ChevronUp,
  Settings,
  Layers,
  SlidersHorizontal,
} from "lucide-react";
import {
  useStartDiscovery,
  useDiscoveryStatus,
} from "@/hooks/use-sybil-discovery";
import { SybilNode, RiskClassification } from "@/types/api";
import { LABEL_COLORS, EDGE_LAYERS } from "@/lib/graph-constants";
import { useTranslations } from "next-intl";

const LoadingFallback = () => {
  const t = useTranslations("DiscoveryPage");
  return (
    <div className="flex flex-col items-center gap-4">
      <Loader2 className="text-accent-cyan animate-spin" size={40} />
      <span className="animate-pulse font-mono text-xs tracking-widest text-slate-500 uppercase italic">
        {t("engine_loading")}
      </span>
    </div>
  );
};

const UniversalGraph2D = dynamic(
  () => import("@/components/graph/universal-graph-2d"),
  {
    ssr: false,
    loading: () => <LoadingFallback />,
  }
);

import ClusterDetailPanel from "@/components/graph/cluster-detail-panel";
import EdgeDetailPanel from "@/components/inspector/edge-detail-panel";
import { AggregatedLink } from "@/hooks/use-graph-processor";
import { RiskDistributionChart } from "@/components/stats/risk-distribution-chart";
import { NetworkStructureChart } from "@/components/stats/network-structure-chart";
import { RiskDistributionItem, EdgeDistributionItem } from "@/types/api";

// ─── Filter state types ───
const ALL_LABELS: RiskClassification[] = [
  "BENIGN",
  "LOW_RISK",
  "HIGH_RISK",
  "MALICIOUS",
];

export default function DiscoveryPage() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("2026-01-01");
  const [endDate, setEndDate] = useState("2026-03-01");
  const [maxNodes, setMaxNodes] = useState(1000);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [maxEpochs, setMaxEpochs] = useState(400);
  const [patience, setPatience] = useState(30);
  const [learningRate, setLearningRate] = useState(0.005);

  const startPickerRef = useRef<HTMLInputElement>(null);
  const endPickerRef = useRef<HTMLInputElement>(null);

  const t = useTranslations("DiscoveryPage");

  // ─── Filter state ───
  const [activeLabels, setActiveLabels] = useState<Set<RiskClassification>>(
    new Set(ALL_LABELS)
  );
  const [filterClusterId, setFilterClusterId] = useState<string>("");

  // ─── Cluster drill-down state ───
  const [selectedCluster, setSelectedCluster] = useState<{
    clusterId: number;
    nodes: SybilNode[];
  } | null>(null);

  const [selectedLink, setSelectedLink] = useState<AggregatedLink | null>(null);

  const toDisplayDate = (dateStr: string) => {
    if (!dateStr) return "";
    const [y, m, d] = dateStr.split("-");
    return `${d}/${m}/${y}`;
  };

  const [startDisplay, setStartDisplay] = useState(toDisplayDate(startDate));
  const [endDisplay, setEndDisplay] = useState(toDisplayDate(endDate));
  const [dateError, setDateError] = useState<string | null>(null);

  const validateDateRange = (start: string, end: string): string | null => {
    const s = new Date(start),
      e = new Date(end);
    if (s > e) return t("error_start_after_end");
    const diffDays = Math.ceil(Math.abs(e.getTime() - s.getTime()) / 86400000);
    if (diffDays > 7) return t("error_range_exceeded");
    return null;
  };

  const onDateInput = (
    val: string,
    setDisplay: (v: string) => void,
    setActual: (v: string) => void
  ) => {
    const clean = val.replace(/[^\d]/g, "");
    let formatted = clean;
    if (clean.length > 2) formatted = clean.slice(0, 2) + "/" + clean.slice(2);
    if (clean.length > 4)
      formatted =
        clean.slice(0, 2) + "/" + clean.slice(2, 4) + "/" + clean.slice(4, 8);
    setDisplay(formatted);
    if (clean.length === 8) {
      const d = clean.slice(0, 2),
        m = clean.slice(2, 4),
        y = clean.slice(4, 8);
      setActual(`${y}-${m}-${d}`);
      setDateError(null);
    }
  };

  const onPickerChange = (
    val: string,
    setDisplay: (v: string) => void,
    setActual: (v: string) => void
  ) => {
    setActual(val);
    setDisplay(toDisplayDate(val));
    setDateError(null);
  };

  const startDiscovery = useStartDiscovery();
  const { data: statusData } = useDiscoveryStatus(taskId);

  const handleStart = async () => {
    const error = validateDateRange(startDate, endDate);
    if (error) {
      setDateError(error);
      return;
    }
    setDateError(null);
    setSelectedCluster(null);
    try {
      const response = await startDiscovery.mutateAsync({
        time_range: { start_date: startDate, end_date: endDate },
        max_nodes: maxNodes,
        hyperparameters: {
          max_epochs: maxEpochs,
          patience,
          learning_rate: learningRate,
        },
      });
      setTaskId(response.task_id);
    } catch (e) {
      console.error("Failed to start discovery:", e);
    }
  };

  // ─── Filtered graph data ───
  const filteredGraphData = useMemo(() => {
    if (!statusData?.graph_data) return null;
    const { nodes, links } = statusData.graph_data;

    const numericCluster =
      filterClusterId !== "" ? parseInt(filterClusterId) : null;

    const filteredNodes = nodes.filter((n) => {
      if (!activeLabels.has(n.risk_label as RiskClassification)) return false;
      if (
        numericCluster !== null &&
        !isNaN(numericCluster) &&
        n.cluster_id !== numericCluster
      )
        return false;
      return true;
    });

    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredLinks = links.filter((l) => {
      const s = String(
        typeof l.source === "object"
          ? (l.source as { id: string }).id
          : l.source
      );
      const t = String(
        typeof l.target === "object"
          ? (l.target as { id: string }).id
          : l.target
      );
      return nodeIds.has(s) && nodeIds.has(t);
    });

    return { nodes: filteredNodes, links: filteredLinks };
  }, [statusData?.graph_data, activeLabels, filterClusterId]);

  // ─── Chart Data ───
  const riskDistributionData = useMemo<RiskDistributionItem[]>(() => {
    if (!filteredGraphData?.nodes) return [];

    const counts: Record<RiskClassification, number> = {
      BENIGN: 0,
      LOW_RISK: 0,
      HIGH_RISK: 0,
      MALICIOUS: 0,
    };

    filteredGraphData.nodes.forEach((n) => {
      const label = n.risk_label as RiskClassification;
      if (counts[label] !== undefined) {
        counts[label]++;
      }
    });

    return Object.entries(counts).map(([label, count]) => ({
      label: label as RiskClassification,
      count,
    }));
  }, [filteredGraphData?.nodes]);

  const edgeDistributionData = useMemo<EdgeDistributionItem[]>(() => {
    if (!filteredGraphData?.links) return [];

    const counts: Record<string, number> = {
      FOLLOW: 0,
      INTERACT: 0,
      SIMILARITY: 0,
      "CO-OWNER": 0,
    };
    let totalEdges = 0;

    filteredGraphData.links.forEach((l) => {
      const type = l.edge_type || "UNKNOWN";
      if (type.endsWith("_REV")) return;

      let matchedLayer: string | null = null;
      for (const layer of EDGE_LAYERS) {
        if (layer.types.includes(type)) {
          matchedLayer = layer.key;
          break;
        }
      }

      if (matchedLayer && counts[matchedLayer] !== undefined) {
        counts[matchedLayer]++;
        totalEdges++;
      }
    });

    return Object.entries(counts)
      .map(([layer, count]) => ({
        layer,
        count,
        percentage:
          totalEdges > 0 ? Number(((count / totalEdges) * 100).toFixed(1)) : 0,
      }))
      .sort((a, b) => b.count - a.count);
  }, [filteredGraphData?.links]);

  // ─── Cluster IDs for filter dropdown ───
  const allClusterIds = useMemo(() => {
    if (!statusData?.graph_data?.nodes) return [];
    const ids = new Set(
      statusData.graph_data.nodes
        .map((n) => n.cluster_id)
        .filter((id): id is number => id !== undefined && id !== null)
    );
    return Array.from(ids).sort((a, b) => a - b);
  }, [statusData]);

  // ─── Cluster click handler ───
  const handleClusterNodeClick = useCallback(
    (clusterId: number, nodes: SybilNode[]) => {
      setSelectedCluster({ clusterId, nodes });
    },
    []
  );

  const toggleLabel = (label: RiskClassification) => {
    setActiveLabels((prev) => {
      const next = new Set(prev);
      if (next.has(label)) {
        if (next.size === 1) return prev; // keep at least one
        next.delete(label);
      } else {
        next.add(label);
      }
      return next;
    });
  };

  const logs = useMemo(() => {
    const base = [
      t("log_initialized"),
      t("log_config_time", { start: startDate, end: endDate }),
      t("log_config_nodes", { max: maxNodes }),
    ];
    if (startDiscovery.isPending) base.push(t("log_initiating"));
    if (taskId) base.push(t("log_task_assigned", { taskId }));
    if (statusData) {
      base.push(
        t("log_progress", {
          status: statusData.status,
          progress: statusData.progress,
          step: statusData.current_step,
        })
      );
      if (statusData.message)
        base.push(t("log_message", { message: statusData.message }));
      if (statusData.status === "COMPLETED") base.push(t("log_success"));
      else if (statusData.status === "FAILED") base.push(t("log_aborted"));
    }
    return base;
  }, [
    startDate,
    endDate,
    maxNodes,
    startDiscovery.isPending,
    taskId,
    statusData,
    t,
  ]);

  const isProcessing =
    startDiscovery.isPending ||
    (statusData &&
      (statusData.status === "PENDING" || statusData.status === "PROCESSING"));

  const isCompleted =
    statusData?.status === "COMPLETED" && !!statusData.graph_data;

  return (
    <div className="flex flex-col gap-6">
      {/* ── Header ── */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex flex-col">
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            {t("page_title")}{" "}
            <span className="text-accent-cyan">
              {t("page_title_highlight")}
            </span>
          </h2>
          <span className="text-subtle">{t("page_subtitle")}</span>
        </div>
        <div className="bg-surface border-border flex items-center gap-2 rounded-sm border px-4 py-2">
          <Database size={14} className="text-accent-cyan" />
          <span className="font-mono text-[10px] font-bold tracking-widest text-slate-500 uppercase italic">
            {t("dataset")}
          </span>
        </div>
      </div>

      {/* ── Discovery Parameters ── */}
      <IndustrialCard title={t("card_parameters")}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end gap-6">
            {/* Start Date */}
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                <Calendar size={12} /> {t("start_date")}
              </label>
              <div className="relative flex items-center">
                <input
                  type="text"
                  placeholder="DD/MM/YYYY"
                  className="bg-surface-secondary/50 border-border text-foreground focus:border-accent-cyan w-32 rounded-sm border p-2 pr-10 font-mono text-xs shadow-inner transition-all outline-none disabled:opacity-50"
                  value={startDisplay}
                  onChange={(e) =>
                    onDateInput(e.target.value, setStartDisplay, setStartDate)
                  }
                  disabled={!!isProcessing}
                />
                <button
                  type="button"
                  className="hover:text-accent-cyan absolute right-2 text-slate-500"
                  onClick={() => startPickerRef.current?.showPicker()}
                  disabled={!!isProcessing}
                >
                  <Calendar size={16} />
                </button>
                <input
                  ref={startPickerRef}
                  type="date"
                  className="pointer-events-none absolute inset-0 opacity-0"
                  onChange={(e) =>
                    onPickerChange(
                      e.target.value,
                      setStartDisplay,
                      setStartDate
                    )
                  }
                  value={startDate}
                />
              </div>
            </div>

            {/* End Date */}
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                <Calendar size={12} /> {t("end_date")}
              </label>
              <div className="relative flex items-center">
                <input
                  type="text"
                  placeholder="DD/MM/YYYY"
                  className="bg-surface-secondary/50 border-border text-foreground focus:border-accent-cyan w-32 rounded-sm border p-2 pr-10 font-mono text-xs shadow-inner transition-all outline-none disabled:opacity-50"
                  value={endDisplay}
                  onChange={(e) =>
                    onDateInput(e.target.value, setEndDisplay, setEndDate)
                  }
                  disabled={!!isProcessing}
                />
                <button
                  type="button"
                  className="hover:text-accent-cyan absolute right-2 text-slate-500"
                  onClick={() => endPickerRef.current?.showPicker()}
                  disabled={!!isProcessing}
                >
                  <Calendar size={16} />
                </button>
                <input
                  ref={endPickerRef}
                  type="date"
                  className="pointer-events-none absolute inset-0 opacity-0"
                  onChange={(e) =>
                    onPickerChange(e.target.value, setEndDisplay, setEndDate)
                  }
                  value={endDate}
                />
              </div>
            </div>

            {/* Max Nodes */}
            <div className="flex flex-col gap-2">
              <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                <Filter size={12} /> {t("max_nodes")}
              </label>
              <input
                type="number"
                className="bg-surface-secondary/50 border-border text-foreground focus:border-accent-cyan w-24 rounded-sm border p-2 font-mono text-xs shadow-inner transition-all outline-none disabled:opacity-50"
                value={maxNodes}
                onChange={(e) => setMaxNodes(Number(e.target.value))}
                disabled={!!isProcessing}
              />
            </div>

            {/* Start button */}
            <div className="flex flex-1 flex-col gap-2">
              <button
                className={`group active:shadow-neo-concave relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-sm py-2.5 font-black text-white shadow-lg transition-all active:translate-y-0.5 disabled:translate-y-0 dark:text-black ${isProcessing ? "cursor-not-allowed bg-slate-700 shadow-none grayscale" : "bg-accent-red hover:brightness-110"}`}
                onClick={handleStart}
                disabled={!!isProcessing}
              >
                <div className="absolute inset-0 translate-x-[-100%] bg-white/20 italic transition-transform duration-500 group-hover:translate-x-[100%]" />
                {isProcessing ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Play size={18} fill="currentColor" />
                )}
                <span className="tracking-[0.2em] uppercase italic">
                  {isProcessing ? t("btn_processing") : t("btn_start")}
                </span>
              </button>
              {dateError && (
                <span className="text-accent-red font-mono text-[10px] font-bold uppercase italic">
                  {t("error_prefix")} {dateError}
                </span>
              )}
            </div>
          </div>

          {/* Advanced Options */}
          <div className="border-t border-slate-800 pt-2">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase transition-colors hover:text-slate-300"
            >
              {showAdvanced ? (
                <ChevronUp size={12} />
              ) : (
                <ChevronDown size={12} />
              )}
              {t("advanced_options")}
            </button>
            {showAdvanced && (
              <div className="mt-4 grid grid-cols-3 gap-6">
                {[
                  {
                    label: t("max_epochs"),
                    val: maxEpochs,
                    set: setMaxEpochs,
                    step: undefined,
                  },
                  {
                    label: t("patience"),
                    val: patience,
                    set: setPatience,
                    step: undefined,
                  },
                  {
                    label: t("learning_rate"),
                    val: learningRate,
                    set: setLearningRate,
                    step: 0.0001,
                  },
                ].map(({ label, val, set, step }) => (
                  <div key={label} className="flex flex-col gap-2">
                    <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                      <Settings size={12} /> {label}
                    </label>
                    <input
                      type="number"
                      step={step}
                      className="bg-surface-secondary/50 border-border text-foreground focus:border-accent-cyan rounded-sm border p-2 font-mono text-xs shadow-inner outline-none disabled:opacity-50"
                      value={val}
                      onChange={(e) => set(Number(e.target.value) as never)}
                      disabled={!!isProcessing}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </IndustrialCard>

      {/* ── Graph Area (graph + optional drill-down panel) ── */}
      <div
        className={`relative flex overflow-hidden rounded-sm border border-slate-800/70 shadow-2xl`}
        style={{ height: "700px" }}
      >
        {/* ── Graph canvas ── */}
        <div
          className={`relative flex-1 bg-[#050608] transition-all duration-300 ${selectedCluster ? "w-[calc(100%-320px)]" : "w-full"}`}
        >
          {/* Background grid */}
          <div className="pointer-events-none absolute inset-0">
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b12_1px,transparent_1px),linear-gradient(to_bottom,#1e293b12_1px,transparent_1px)] bg-[size:40px_40px]" />
          </div>

          {/* ── Filter toolbar (shown when completed) ── */}
          {isCompleted && (
            <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
              {/* Filter toggle */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center gap-2 border px-3 py-1.5 font-mono text-[9px] font-bold uppercase backdrop-blur-sm transition-all ${showFilters ? "border-accent-cyan/40 bg-accent-cyan/10 text-accent-cyan" : "border-border bg-surface/70 text-subtle hover:text-foreground"}`}
              >
                <SlidersHorizontal size={11} />
              </button>

              {/* Filter panel */}
              {showFilters && (
                <div className="glass-industrial flex min-w-[200px] flex-col gap-3 p-3 shadow-2xl">
                  {/* Risk label filter */}
                  <div>
                    <div className="text-subtle mb-2 !text-[8px] font-bold">
                      {t("filter_risk_label")}
                    </div>
                    <div className="flex flex-col gap-1.5">
                      {ALL_LABELS.map((rl) => {
                        const active = activeLabels.has(
                          rl as RiskClassification
                        );
                        const color = LABEL_COLORS[rl];
                        const cnt =
                          statusData?.graph_data?.nodes.filter(
                            (n) => n.risk_label === (rl as RiskClassification)
                          ).length ?? 0;
                        return (
                          <button
                            key={rl}
                            onClick={() => toggleLabel(rl)}
                            className="flex items-center justify-between gap-2 px-2 py-1 transition-all"
                            style={{
                              background: active ? color + "12" : "transparent",
                              border: `1px solid ${active ? color + "44" : "var(--border)"}`,
                            }}
                          >
                            <div className="flex items-center gap-2">
                              <div
                                className="h-1.5 w-1.5 rounded-full"
                                style={{
                                  backgroundColor: active ? color : "gray",
                                }}
                              />
                              <span
                                className="font-mono text-[8px] font-bold uppercase"
                                style={{
                                  color: active ? color : "var(--foreground)",
                                  opacity: active ? 1 : 0.6,
                                }}
                              >
                                {rl.replace("_", " ")}
                              </span>
                            </div>
                            <span
                              className="font-mono text-[8px] tabular-nums"
                              style={{
                                color: active ? color : "var(--foreground)",
                                opacity: active ? 0.7 : 0.4,
                              }}
                            >
                              {cnt}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* Cluster filter */}
                  <div>
                    <div className="text-subtle mb-2 flex items-center gap-1 !text-[8px] font-bold">
                      <Layers size={9} /> {t("filter_cluster_id")}
                    </div>
                    <div className="flex gap-2">
                      <select
                        value={filterClusterId}
                        onChange={(e) => setFilterClusterId(e.target.value)}
                        className="bg-surface-secondary border-border text-foreground focus:border-accent-cyan/50 flex-1 border px-2 py-1 font-mono text-[9px] outline-none"
                      >
                        <option value="">{t("filter_all_clusters")}</option>
                        {allClusterIds.map((id) => (
                          <option key={id} value={String(id)}>
                            {t("filter_cluster_option", { id })}
                          </option>
                        ))}
                      </select>
                      {filterClusterId && (
                        <button
                          onClick={() => setFilterClusterId("")}
                          className="border-border text-subtle hover:text-foreground border px-2 text-[9px]"
                        >
                          ×
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Results count */}
                  <div className="border-border text-subtle border-t pt-2 !text-[8px]">
                    {t("showing_nodes", {
                      filtered: filteredGraphData?.nodes.length ?? 0,
                      total: statusData?.graph_data?.nodes.length ?? 0,
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Main content: graph or empty state ── */}
          {isCompleted && filteredGraphData ? (
            <UniversalGraph2D
              mode="CLUSTER"
              graphData={filteredGraphData}
              onClusterNodeClick={handleClusterNodeClick}
              onLinkClick={setSelectedLink}
              allNodes={statusData?.graph_data?.nodes}
            />
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-6">
              {!taskId && !isProcessing ? (
                <div className="flex flex-col items-center text-center">
                  <div className="bg-surface-secondary border-border text-subtle mb-6 flex h-24 w-24 items-center justify-center border">
                    <Database size={40} />
                  </div>
                  <h2 className="text-subtle mb-2 !text-xl font-black">
                    {t("no_scan_data")}
                  </h2>
                  <p className="text-subtle max-w-xs leading-relaxed !font-normal">
                    {t("no_scan_desc")}
                  </p>
                </div>
              ) : (
                <div className="relative flex flex-col items-center gap-6">
                  <div className="relative">
                    <div className="border-accent-cyan/20 absolute inset-0 animate-ping rounded-full border duration-[3s]" />
                    <div className="border-accent-cyan/10 absolute inset-[-20px] animate-ping rounded-full border duration-[5s]" />
                    <div className="border-accent-cyan/30 flex h-48 w-48 animate-[spin_20s_linear_infinite] items-center justify-center rounded-full border-2 border-dashed">
                      <div className="border-accent-cyan/50 border-t-accent-cyan flex h-32 w-32 animate-spin items-center justify-center rounded-full border">
                        <Database
                          size={32}
                          className="text-accent-cyan opacity-50"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <span className="text-accent-cyan font-mono text-sm font-bold tracking-[0.3em] uppercase italic">
                      {isProcessing ? t("scanning") : t("awaiting")}
                    </span>
                    {statusData && (
                      <span className="text-subtle">
                        {t("progress", {
                          progress: statusData.progress,
                          step: statusData.current_step,
                        })}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Scan statistics overlay ── */}
          {isCompleted && statusData.graph_data && (
            <div
              className="glass-industrial absolute top-4 z-10 flex flex-col gap-2 p-4 shadow-2xl"
              style={{
                left: showFilters ? "220px" : "64px",
                transition: "left 0.2s",
              }}
            >
              <div className="text-subtle mb-1 font-bold">
                {t("scan_stats")}
              </div>
              {[
                {
                  label: t("clusters_count"),
                  value: statusData.graph_data.cluster_count,
                  color: "text-accent-cyan",
                },
                {
                  label: t("nodes_found"),
                  value:
                    filteredGraphData?.nodes.length ??
                    statusData.graph_data.num_nodes,
                  color: "text-foreground",
                },
                {
                  label: t("edges_found"),
                  value:
                    filteredGraphData?.links.length ??
                    statusData.graph_data.num_edges,
                  color: "text-foreground",
                },
              ].map(({ label, value, color }) => (
                <div
                  key={label}
                  className="border-border flex items-center justify-between gap-8 border-b pb-1 last:border-0 last:pb-0"
                >
                  <span className="text-subtle !text-[10px] lowercase first-letter:uppercase">
                    {label}
                  </span>
                  <span className={`font-mono text-[11px] font-bold ${color}`}>
                    {value}
                  </span>
                </div>
              ))}
              {selectedCluster && (
                <div className="border-accent-cyan/20 mt-1 border-t pt-2">
                  <span className="text-accent-cyan font-mono text-[9px]">
                    {t("viewing_cluster", { id: selectedCluster.clusterId })}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Cluster drill-down panel ── */}
        {selectedCluster && (
          <div
            className="w-80 flex-shrink-0 border-l border-slate-800/80"
            style={{ height: "100%" }}
          >
            <ClusterDetailPanel
              clusterId={selectedCluster.clusterId}
              nodes={selectedCluster.nodes}
              onClose={() => setSelectedCluster(null)}
            />
          </div>
        )}

        {/* ── Edge detail panel ── */}
        {selectedLink && (
          <div
            className="w-80 flex-shrink-0 border-l border-slate-800/80"
            style={{ height: "100%" }}
          >
            <EdgeDetailPanel
              link={selectedLink}
              onClose={() => setSelectedLink(null)}
            />
          </div>
        )}
      </div>

      {/* ── Charts Area ── */}
      {isCompleted && filteredGraphData && (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <RiskDistributionChart data={riskDistributionData} />
          <NetworkStructureChart data={edgeDistributionData} />
        </div>
      )}

      {/* ── Terminal Log ── */}
      <TerminalLog className="border-border h-40 shadow-2xl" logs={logs} />

      <style jsx global>{`
        @keyframes scan {
          from {
            transform: translateY(-100%);
          }
          to {
            transform: translateY(500%);
          }
        }
      `}</style>
    </div>
  );
}
