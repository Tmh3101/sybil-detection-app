"use client";

import { useState, useMemo, useRef } from "react";
import dynamic from "next/dynamic";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { TerminalLog } from "@/components/ui/terminal-log";
import { Play, Calendar, Filter, Database, Loader2 } from "lucide-react";
import {
  useStartDiscovery,
  useDiscoveryStatus,
} from "@/hooks/use-sybil-discovery";

const ClusterMap2D = dynamic(
  () => import("@/components/graph/cluster-map-2d"),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center gap-4">
        <Loader2 className="text-accent-cyan animate-spin" size={40} />
        <span className="animate-pulse font-mono text-xs tracking-widest text-slate-500 uppercase italic">
          Loading Cluster Map Engine...
        </span>
      </div>
    ),
  }
);

export default function DiscoveryPage() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("2026-03-01");
  const [endDate, setEndDate] = useState("2026-03-23");
  const [maxNodes, setMaxNodes] = useState(1000);

  const startPickerRef = useRef<HTMLInputElement>(null);
  const endPickerRef = useRef<HTMLInputElement>(null);

  // Helper: YYYY-MM-DD -> DD/MM/YYYY
  const toDisplayDate = (dateStr: string) => {
    if (!dateStr) return "";
    const parts = dateStr.split("-");
    if (parts.length !== 3) return dateStr;
    const [y, m, d] = parts;
    return `${d}/${m}/${y}`;
  };

  // Improved approach: Separate display state for inputs
  const [startDisplay, setStartDisplay] = useState(toDisplayDate(startDate));
  const [endDisplay, setEndDisplay] = useState(toDisplayDate(endDate));

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
      const d = clean.slice(0, 2);
      const m = clean.slice(2, 4);
      const y = clean.slice(4, 8);
      setActual(`${y}-${m}-${d}`);
    }
  };

  const onPickerChange = (
    val: string,
    setDisplay: (v: string) => void,
    setActual: (v: string) => void
  ) => {
    setActual(val);
    setDisplay(toDisplayDate(val));
  };

  const startDiscovery = useStartDiscovery();
  const { data: statusData } = useDiscoveryStatus(taskId);

  const handleStart = async () => {
    try {
      const response = await startDiscovery.mutateAsync({
        time_range: {
          start_date: startDate,
          end_date: endDate,
        },
        max_nodes: maxNodes,
      });
      setTaskId(response.task_id);
    } catch (error) {
      console.error("Failed to start discovery:", error);
    }
  };

  const logs = useMemo(() => {
    const baseLogs = [
      "[SYSTEM] DISCOVERY LAB INITIALIZED",
      `[CONFIG] TIME_RANGE: ${startDate} TO ${endDate}`,
      `[CONFIG] MAX_NODES: ${maxNodes}`,
    ];

    if (startDiscovery.isPending) {
      baseLogs.push("[ACTION] INITIATING START DISCOVERY PROTOCOL...");
    }

    if (taskId) {
      baseLogs.push(`[ACTION] TASK_ID ASSIGNED: ${taskId}`);
    }

    if (statusData) {
      const statusLog = `[${statusData.status}] PROGRESS: ${statusData.progress}% - ${statusData.current_step}`;
      baseLogs.push(statusLog);

      if (statusData.message) {
        const messagePrefix =
          statusData.status === "FAILED" ? "[ERROR]" : "[MESSAGE]";
        baseLogs.push(`${messagePrefix} ${statusData.message}`);
      }

      if (statusData.status === "COMPLETED") {
        baseLogs.push(
          "[SUCCESS] DISCOVERY PROTOCOL COMPLETED. RENDERING CLUSTER MAP."
        );
      } else if (statusData.status === "FAILED") {
        baseLogs.push(
          "[CRITICAL] DISCOVERY PROTOCOL ABORTED. CHECK SYSTEM ARCHIVE."
        );
      }
    }

    return baseLogs;
  }, [
    startDate,
    endDate,
    maxNodes,
    startDiscovery.isPending,
    taskId,
    statusData,
  ]);

  const isProcessing =
    startDiscovery.isPending ||
    (statusData &&
      (statusData.status === "PENDING" || statusData.status === "PROCESSING"));

  return (
    <div className="flex h-full flex-col gap-6">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex flex-col">
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Discovery <span className="text-accent-cyan">Lab</span>
          </h2>
          <span className="text-subtle">
            Large Scale Cluster Detection & Network Analysis
          </span>
        </div>
        <div className="bg-surface border-border flex items-center gap-2 rounded-sm border px-4 py-2">
          <Database size={14} className="text-accent-cyan" />
          <span className="font-mono text-[10px] font-bold tracking-widest text-slate-500 uppercase italic">
            Dataset: lens-protocol-mainnet
          </span>
        </div>
      </div>

      {/* Top Controls */}
      <IndustrialCard title="DISCOVERY PARAMETERS">
        <div className="flex items-end gap-8">
          <div className="flex flex-col gap-2">
            <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
              <Calendar size={12} /> Start Date
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
                className="hover:text-accent-cyan absolute right-2 text-slate-500 transition-colors"
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
                  onPickerChange(e.target.value, setStartDisplay, setStartDate)
                }
                value={startDate}
              />
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
              <Calendar size={12} /> End Date
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
                className="hover:text-accent-cyan absolute right-2 text-slate-500 transition-colors"
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
          <div className="flex flex-col gap-2">
            <label className="flex items-center gap-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
              <Filter size={12} /> Max Nodes
            </label>
            <input
              type="number"
              className="bg-surface-secondary/50 border-border text-foreground focus:border-accent-cyan w-24 rounded-sm border p-2 font-mono text-xs shadow-inner transition-all outline-none disabled:opacity-50"
              value={maxNodes}
              onChange={(e) => setMaxNodes(Number(e.target.value))}
              disabled={!!isProcessing}
            />
          </div>
          <button
            className={`group active:shadow-neo-concave relative flex flex-1 items-center justify-center gap-3 overflow-hidden rounded-sm py-2.5 font-black text-white shadow-lg transition-all active:translate-y-0.5 disabled:translate-y-0 dark:text-black ${isProcessing ? "cursor-not-allowed bg-slate-700 shadow-none grayscale" : "bg-accent-red hover:brightness-110 active:shadow-inner"}`}
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
              {isProcessing ? "PROCESSING..." : "START DISCOVERY"}
            </span>
          </button>
        </div>
      </IndustrialCard>

      {/* Cluster Map - Dark Screen for clarity */}
      <div className="border-border relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-sm border bg-[#050608] shadow-2xl">
        {/* Background Grid */}
        <div className="pointer-events-none absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px]" />
          <div className="h-full w-full bg-[radial-gradient(circle_at_center,_#1e293b20_0%,_transparent_70%)]" />
        </div>

        {statusData?.status === "COMPLETED" && statusData.graph_data ? (
          <ClusterMap2D graphData={statusData.graph_data} />
        ) : (
          <div className="relative flex flex-col items-center gap-6">
            {!taskId && !isProcessing ? (
              <div className="flex flex-col items-center text-center">
                <div className="mb-6 flex h-24 w-24 items-center justify-center border border-slate-800 bg-slate-900/50 text-slate-700">
                  <Database size={40} />
                </div>
                <h2 className="mb-2 text-xl font-black tracking-tighter text-slate-500 uppercase italic">
                  [ NO SCAN DATA ]
                </h2>
                <p className="max-w-xs font-mono text-[10px] leading-relaxed tracking-widest text-slate-600 uppercase">
                  Select time range & max nodes to begin network scanning.
                </p>
              </div>
            ) : (
              <>
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
                    {isProcessing
                      ? "SCANNING NETWORK..."
                      : "AWAITING PROTOCOLS..."}
                  </span>
                  {statusData && (
                    <span className="font-mono text-[10px] text-slate-500 uppercase">
                      Progress: {statusData.progress}% | Step:{" "}
                      {statusData.current_step}
                    </span>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* Legend */}
        {statusData?.status === "COMPLETED" && (
          <div className="absolute top-6 right-6 z-10 flex flex-col gap-3 border border-slate-700 bg-black/80 p-4 shadow-2xl backdrop-blur-sm">
            <div className="mb-1 text-[8px] font-bold tracking-[0.2em] text-slate-500 uppercase">
              Legend
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-[#ff1744] shadow-[0_0_8px_#ff1744]" />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                High Risk Sybil
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-[#00f2ff]" />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                Cluster Engine A
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-[#8b5cf6]" />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                Cluster Engine B
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="h-2 w-2 rounded-full bg-[#f43f5e]" />
              <span className="font-mono text-[9px] font-bold text-slate-300 uppercase">
                Cluster Engine C
              </span>
            </div>
          </div>
        )}

        {/* Scan effect */}
        {isProcessing && (
          <div className="via-accent-cyan/10 pointer-events-none absolute inset-0 z-0 h-20 w-full animate-[scan_4s_linear_infinite] bg-gradient-to-b from-transparent to-transparent" />
        )}
      </div>

      {/* Bottom Terminal */}
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
