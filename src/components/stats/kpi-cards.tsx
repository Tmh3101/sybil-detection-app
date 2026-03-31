"use client";

import React from "react";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { Users, Share2, Box, TrendingUp } from "lucide-react";

interface KPICardsProps {
  totalNodes: number;
  totalEdges: number;
  totalClusters: number;
  avgClusterSize: number;
}

export const KPICards: React.FC<KPICardsProps> = ({
  totalNodes = 0,
  totalEdges = 0,
  totalClusters = 0,
  avgClusterSize = 0,
}) => {
  const kpis = [
    {
      label: "TOTAL NODES",
      value: (totalNodes || 0).toLocaleString(),
      icon: Users,
      color: "text-accent-cyan",
    },
    {
      label: "TOTAL EDGES",
      value: (totalEdges || 0).toLocaleString(),
      icon: Share2,
      color: "text-blue-500",
    },
    {
      label: "TOTAL CLUSTERS",
      value: (totalClusters || 0).toLocaleString(),
      icon: Box,
      color: "text-purple-500",
    },
    {
      label: "AVG CLUSTER SIZE",
      value: (avgClusterSize || 0).toFixed(1),
      icon: TrendingUp,
      color: "text-accent-green",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi) => (
        <IndustrialCard key={kpi.label}>
          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-1">
              <span className="font-mono text-[8px] font-bold tracking-[0.2em] text-slate-500 uppercase">
                {kpi.label}
              </span>
              <span
                className={`text-2xl font-black tracking-tight ${kpi.color}`}
              >
                {kpi.value}
              </span>
            </div>
            <div className="bg-surface-secondary/50 border-border rounded-sm border p-2">
              <kpi.icon size={18} className={kpi.color} />
            </div>
          </div>
        </IndustrialCard>
      ))}
    </div>
  );
};
