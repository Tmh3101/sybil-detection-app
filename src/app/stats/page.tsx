"use client";

import React from "react";
import { useStats } from "@/hooks/use-stats";
import { KPICards } from "@/components/stats/kpi-cards";
import { NetworkStructureChart } from "@/components/stats/network-structure-chart";
import { RiskDistributionChart } from "@/components/stats/risk-distribution-chart";
import { BootSequenceLoader } from "@/components/ui/boot-sequence-loader";
import { AlertCircle, RefreshCw } from "lucide-react";

export default function StatsPage() {
  const { data, isLoading, isError } = useStats();

  if (isLoading) {
    return (
      <div className="h-[calc(100vh-64px)] w-full">
        <BootSequenceLoader />
      </div>
    );
  }

  if (isError || !data.overview || !data.clusters) {
    return (
      <div className="flex h-[calc(100vh-64px)] w-full flex-col items-center justify-center gap-4 p-6 text-center">
        <div className="rounded-full bg-red-500/10 p-4 text-red-500">
          <AlertCircle size={48} />
        </div>
        <h2 className="text-foreground font-mono text-xl font-black tracking-widest uppercase italic">
          System Error: <span className="text-red-500">Analytics Offline</span>
        </h2>
        <p className="max-w-md font-mono text-xs leading-relaxed text-slate-500">
          Failed to establish a secure connection with the statistics engine.
          Please ensure the backend services are operational and try again.
        </p>
        <button
          onClick={() => window.location.reload()}
          className="bg-surface border-border hover:border-accent-cyan/50 mt-4 flex items-center gap-2 rounded-sm border px-6 py-2 font-mono text-[10px] font-bold tracking-[0.2em] uppercase italic transition-all"
        >
          <RefreshCw size={14} className="text-accent-cyan" />
          Re-initialize System
        </button>
      </div>
    );
  }

  const { overview, risk, clusters } = data;

  return (
    <div className="animate-in fade-in flex flex-col gap-8 p-8 duration-700">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <div className="bg-accent-cyan h-4 w-1 rounded-full shadow-[0_0_8px_rgba(var(--accent-cyan),0.6)]" />
          <h1 className="text-foreground font-mono text-2xl font-black tracking-tighter uppercase italic">
            Network <span className="text-accent-cyan">Statistics</span>
          </h1>
        </div>
        <p className="font-mono text-[10px] tracking-widest text-slate-500 uppercase">
          Structural analysis and heuristic risk distribution overview
        </p>
      </div>

      {/* Tier 1: KPI Cards */}
      <KPICards
        totalNodes={overview.total_nodes}
        totalEdges={overview.total_edges}
        totalClusters={clusters.total_clusters}
        avgClusterSize={clusters.avg_cluster_size}
      />

      {/* Tier 2: Distribution Charts */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {overview.edge_distribution && (
          <NetworkStructureChart data={overview.edge_distribution} />
        )}
        {risk?.distribution && (
          <RiskDistributionChart data={risk.distribution} />
        )}
      </div>
    </div>
  );
}
