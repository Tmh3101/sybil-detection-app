"use client";

import { useEffect, useState } from "react";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { TerminalLog } from "@/components/ui/terminal-log";
import { Play, Calendar, Filter, Database } from "lucide-react";

interface ClusterNode {
  cx: number;
  cy: number;
  r: number;
  delay: string;
}

interface ClusterData {
  alpha: ClusterNode[];
  sigma: ClusterNode[];
  threat: ClusterNode[];
}

export default function DiscoveryPage() {
  const [clusterData, setClusterData] = useState<ClusterData | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setClusterData({
      alpha: [...Array(30)].map(() => ({
        cx: 250 + Math.random() * 150,
        cy: 200 + Math.random() * 150,
        r: 2 + Math.random() * 4,
        delay: `${Math.random() * 2}s`,
      })),
      sigma: [...Array(45)].map(() => ({
        cx: 650 + Math.random() * 200,
        cy: 350 + Math.random() * 180,
        r: 1 + Math.random() * 3,
        delay: `${Math.random() * 2}s`,
      })),
      threat: [...Array(20)].map(() => ({
        cx: 500 + Math.random() * 100,
        cy: 250 + Math.random() * 100,
        r: 3 + Math.random() * 5,
        delay: `${Math.random() * 2}s`,
      })),
    });
  }, []);

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between mb-2">
        <div className="flex flex-col">
          <h2 className="text-3xl font-black tracking-tighter text-foreground italic uppercase">
            Discovery <span className="text-accent-cyan">Lab</span>
          </h2>
          <span className="text-subtle">
            Large Scale Cluster Detection & Network Analysis
          </span>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 bg-surface border border-border rounded-sm">
          <Database size={14} className="text-accent-cyan" />
          <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest italic font-bold">
            Dataset: BigQuery_Mainnet_Current
          </span>
        </div>
      </div>

      {/* Top Controls */}
      <IndustrialCard title="DISCOVERY_PARAMETERS">
        <div className="flex items-end gap-8">
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-mono text-slate-500 uppercase flex items-center gap-2 font-bold">
              <Calendar size={12} /> Start Date
            </label>
            <input
              type="date"
              className="bg-surface-secondary/50 border border-border text-foreground text-xs p-2 rounded-sm focus:border-accent-cyan outline-none font-mono shadow-inner"
              defaultValue="2026-03-01"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-mono text-slate-500 uppercase flex items-center gap-2 font-bold">
              <Calendar size={12} /> End Date
            </label>
            <input
              type="date"
              className="bg-surface-secondary/50 border border-border text-foreground text-xs p-2 rounded-sm focus:border-accent-cyan outline-none font-mono shadow-inner"
              defaultValue="2026-03-23"
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="text-[10px] font-mono text-slate-500 uppercase flex items-center gap-2 font-bold">
              <Filter size={12} /> Sensitivity
            </label>
            <select className="bg-surface-secondary/50 border border-border text-foreground text-xs p-2 rounded-sm focus:border-accent-cyan outline-none font-mono w-40 shadow-inner">
              <option>AGGRESSIVE</option>
              <option>BALANCED</option>
              <option>CONSERVATIVE</option>
            </select>
          </div>
          <button className="flex-1 bg-accent-red hover:brightness-110 text-white dark:text-black font-black py-2.5 rounded-sm flex items-center justify-center gap-3 transition-all group overflow-hidden relative shadow-lg">
            <div className="absolute inset-0 bg-white/20 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-500 italic" />
            <Play size={18} fill="currentColor" />
            <span className="tracking-[0.2em] italic uppercase">
              START DISCOVERY PROTOCOL
            </span>
          </button>
        </div>
      </IndustrialCard>

      {/* Cluster Map - Dark Screen for clarity */}
      <div className="flex-1 bg-[#050608] border border-border rounded-sm relative overflow-hidden flex items-center justify-center min-h-0 shadow-2xl">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_#1e293b20_0%,_transparent_70%)]" />

        {/* Mock Cluster Map SVG */}
        <svg className="w-full h-full max-h-[500px]" viewBox="0 0 1000 600">
          {clusterData && (
            <>
              {/* Cluster Clouds */}
              {/* Purple Cluster */}
              <g className="opacity-60">
                {clusterData.alpha.map((node, i) => (
                  <circle
                    key={`p-${i}`}
                    cx={node.cx}
                    cy={node.cy}
                    r={node.r}
                    fill="#a855f7"
                    className="animate-pulse"
                    style={{ animationDelay: node.delay }}
                  />
                ))}
                <text
                  x="250"
                  y="180"
                  className="fill-purple-400 font-mono text-[10px] font-bold uppercase tracking-widest"
                >
                  Cluster_Alpha_T7
                </text>
              </g>

              {/* Orange Cluster */}
              <g className="opacity-60">
                {clusterData.sigma.map((node, i) => (
                  <circle
                    key={`o-${i}`}
                    cx={node.cx}
                    cy={node.cy}
                    r={node.r}
                    fill="#f97316"
                    className="animate-pulse"
                    style={{ animationDelay: node.delay }}
                  />
                ))}
                <text
                  x="700"
                  y="330"
                  className="fill-orange-400 font-mono text-[10px] font-bold uppercase tracking-widest"
                >
                  Cluster_Sigma_X4
                </text>
              </g>

              {/* Red (Danger) Cluster */}
              <g className="opacity-80">
                {clusterData.threat.map((node, i) => (
                  <circle
                    key={`r-${i}`}
                    cx={node.cx}
                    cy={node.cy}
                    r={node.r}
                    fill="#ff1744"
                    className="animate-pulse"
                    style={{ animationDelay: node.delay }}
                  />
                ))}
                <text
                  x="500"
                  y="230"
                  className="fill-accent-red font-mono text-[10px] font-bold uppercase tracking-widest"
                >
                  SYBIL_THREAT_OMEGA
                </text>
              </g>
            </>
          )}

          {/* Connection Lines */}
          <line
            x1="300"
            y1="250"
            x2="520"
            y2="280"
            stroke="#475569"
            strokeWidth="0.5"
            strokeDasharray="4,2"
          />
          <line
            x1="680"
            y1="400"
            x2="550"
            y2="320"
            stroke="#475569"
            strokeWidth="0.5"
            strokeDasharray="4,2"
          />
        </svg>

        {/* Legend */}
        <div className="absolute top-6 right-6 flex flex-col gap-3 bg-black/80 p-4 border border-slate-700 backdrop-blur-sm shadow-2xl">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-accent-red" />
            <span className="text-[9px] font-mono text-slate-300 uppercase font-bold">
              Malicious Cluster
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-orange-500" />
            <span className="text-[9px] font-mono text-slate-300 uppercase font-bold">
              Suspicious Group
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-purple-500" />
            <span className="text-[9px] font-mono text-slate-300 uppercase font-bold">
              Heuristic Match
            </span>
          </div>
        </div>

        {/* Scan effect */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent-cyan/10 to-transparent h-20 w-full animate-[scan_4s_linear_infinite] pointer-events-none" />
      </div>

      {/* Bottom Terminal */}
      <TerminalLog
        className="h-40 border-border shadow-2xl"
        logs={[
          "[INIT] STARTING CLUSTER DISCOVERY...",
          "[QUERY] FETCHING 4.2M TRANSACTIONS FROM BigQuery...",
          "[PROCESS] RUNNING Louvain Community Detection...",
          "[PROCESS] EXTRACTING FEATURES FOR 12k CANDIDATES...",
          "[GAE] TRAINING GRAPH AUTOENCODER MODEL...",
          "[INFO] CLUSTER_OMEGA DETECTED: 442 NODES",
          "[INFO] HIGH SIMILARITY FOUND IN IP_HISTORY_LOG",
          "[STABLE] WAITING FOR USER INPUT...",
        ]}
      />

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
