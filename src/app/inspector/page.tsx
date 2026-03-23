"use client";

import { Suspense } from "react";
import dynamic from "next/dynamic";
import { useSearchParams } from "next/navigation";
import { useInspectProfile } from "@/hooks/use-sybil-inference";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { TerminalLog } from "@/components/ui/terminal-log";
import Image from "next/image";
import {
  Wallet,
  ShieldCheck,
  User,
  Activity,
  AlertTriangle,
  Loader2,
} from "lucide-react";

const EgoGraph3D = dynamic(() => import("@/components/graph/ego-graph-3d"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex flex-col items-center justify-center gap-4 bg-black/20">
      <Loader2 className="text-accent-cyan animate-spin" size={32} />
      <span className="text-[10px] font-mono text-accent-cyan animate-pulse uppercase tracking-[0.2em] font-bold">
        INITIALIZING 3D RENDER ENGINE...
      </span>
    </div>
  ),
});

function InspectorContent() {
  const searchParams = useSearchParams();
  const walletId = searchParams.get("wallet");
  const { data, isLoading, isError } = useInspectProfile(walletId);

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case "SYBIL":
        return "text-accent-red";
      case "WARNING":
        return "text-orange-500";
      case "BENIGN":
        return "text-accent-green";
      default:
        return "text-accent-cyan";
    }
  };

  if (!walletId) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <div className="p-8 border-2 border-dashed border-border rounded-lg bg-surface/50 max-w-lg text-center">
          <Activity className="mx-auto mb-4 text-slate-500 animate-pulse" size={48} />
          <h2 className="text-xl font-black italic tracking-tighter uppercase text-foreground mb-2">
            Awaiting Target Input...
          </h2>
          <p className="text-sm font-mono text-slate-500 uppercase tracking-widest leading-relaxed">
            Please enter a wallet address or handle in the search bar above to begin risk assessment.
          </p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-6">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="text-accent-cyan animate-spin" size={48} />
          <div className="flex flex-col items-center">
            <span className="text-sm font-mono text-accent-cyan animate-pulse uppercase tracking-[0.2em] font-bold">
              [SYS] Waking up AI Core...
            </span>
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest mt-1">
              Warming up tensors & loading local graph
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-4">
        <div className="p-8 border-2 border-accent-red/20 rounded-lg bg-accent-red/5 max-w-lg text-center">
          <AlertTriangle className="mx-auto mb-4 text-accent-red" size={48} />
          <h2 className="text-xl font-black italic tracking-tighter uppercase text-accent-red mb-2">
            [ERR] Failed to Fetch Target Data
          </h2>
          <p className="text-sm font-mono text-slate-500 uppercase tracking-widest leading-relaxed">
            The sybil engine encountered an error while analyzing the target. Please verify the ID and try again.
          </p>
        </div>
      </div>
    );
  }

  const analysis = data?.analysis;
  const profile = data?.profile_info;
  const prob = (analysis?.sybil_probability || 0) * 100;
  const colorClass = getClassificationColor(analysis?.classification || "");

  return (
    <div className="flex flex-col gap-6 h-full">
      <div className="flex items-center justify-between mb-2">
        <div className="flex flex-col">
          <h2 className="text-3xl font-black tracking-tighter text-foreground italic uppercase">
            Profile <span className="text-accent-cyan">Inspector</span>
          </h2>
          <span className="text-subtle">
            Target Identification & Risk Assessment Module
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="px-4 py-2 bg-surface border border-border rounded-sm text-xs font-mono">
            ID: <span className="text-accent-cyan font-bold uppercase">{walletId.slice(0, 6)}...{walletId.slice(-4)}</span>
          </div>
          <button className="px-6 py-2 bg-accent-red text-white dark:text-black font-black text-xs uppercase tracking-widest hover:brightness-110 transition-all rounded-sm italic shadow-lg">
            QUARANTINE
          </button>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
        {/* Left Column: Risk Score & Logs */}
        <div className="col-span-3 flex flex-col gap-6 overflow-hidden">
          <IndustrialCard
            title="RISK_SCORE_GAUGE"
            className="flex flex-col items-center"
          >
            <div className="relative w-48 h-48 flex items-center justify-center">
              {/* SVG Gauge */}
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="96"
                  cy="96"
                  r="80"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className="text-slate-200 dark:text-slate-800"
                />
                <circle
                  cx="96"
                  cy="96"
                  r="80"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="transparent"
                  strokeDasharray={502.4}
                  strokeDashoffset={502.4 * (1 - (prob / 100))}
                  className={`${colorClass} transition-all duration-1000 ease-out`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className={`text-5xl font-black italic tracking-tighter ${colorClass}`}>
                  {Math.round(prob)}%
                </span>
                <span className="text-subtle -mt-1 uppercase tracking-widest text-[10px] font-bold">
                  {analysis?.classification}
                </span>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-4 w-full">
              <div className="flex flex-col items-center p-2 bg-surface-secondary/30 border border-border shadow-inner">
                <span className="text-[8px] text-slate-500 uppercase font-bold">
                  Classification
                </span>
                <span className={`text-xs font-bold ${colorClass}`}>
                  {analysis?.classification}
                </span>
              </div>
              <div className="flex flex-col items-center p-2 bg-surface-secondary/30 border border-border shadow-inner">
                <span className="text-[8px] text-slate-500 uppercase font-bold">
                  Certainty
                </span>
                <span className="text-xs font-bold text-accent-cyan">
                  {prob > 50 ? Math.round(prob) : Math.round(100 - prob)}%
                </span>
              </div>
            </div>
          </IndustrialCard>

          <IndustrialCard
            title="LIVE_DETECTION_STREAM"
            className="flex-1 min-h-0 p-0"
          >
            <TerminalLog
              className="h-full border-0 bg-transparent"
              logs={analysis?.reasoning || []}
            />
          </IndustrialCard>
        </div>

        {/* Middle Column: Ego Graph */}
        <div className="col-span-6">
          <div className="h-full w-full bg-[#050608] border border-border rounded-sm relative overflow-hidden group shadow-2xl">
            {/* Background Grid - Visible on dark screen */}
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none" />
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] opacity-20 pointer-events-none" />

            {/* Live 3D Ego Graph */}
            <EgoGraph3D 
              graphData={data?.local_graph || { nodes: [], links: [] }}
              targetId={walletId || ""}
              classification={data?.analysis?.classification}
            />

            {/* Overlays */}
            <div className="absolute top-4 left-4 flex gap-2 pointer-events-none">
              <div className="px-2 py-1 bg-black/80 border border-slate-700 text-[9px] font-mono text-accent-cyan uppercase font-bold">
                EGO_GRAPH_v3_WEBGL
              </div>
              <div className="px-2 py-1 bg-black/80 border border-slate-700 text-[9px] font-mono text-slate-400 uppercase">
                REAL_TIME_SPATIAL
              </div>
            </div>

            <div className="absolute bottom-4 right-4 flex flex-col gap-1 items-end pointer-events-none">
              <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">
                Node Connections: {data?.local_graph?.nodes?.length || 0}
              </span>
              <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">
                Network Depth: 4
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Profile Data */}
        <div className="col-span-3 flex flex-col gap-6 overflow-hidden">
          <IndustrialCard title="ENTITY_IDENTITY">
            <div className="flex items-center gap-4 mb-6">
              <div className="w-16 h-16 bg-surface-secondary border-2 border-border rounded-sm flex items-center justify-center relative overflow-hidden group shadow-inner">
                {profile?.picture_url ? (
                  <Image src={profile.picture_url} alt={profile.handle} width={64} height={64} className="w-full h-full object-cover" />
                ) : (
                  <User
                    size={32}
                    className="text-slate-400 group-hover:text-accent-cyan transition-colors"
                  />
                )}
                <div className="absolute inset-0 bg-accent-cyan/10 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
              <div className="flex flex-col">
                <span className="text-subtle tracking-tighter">Alias</span>
                <span className="text-lg font-black text-foreground italic uppercase truncate max-w-[180px]">
                  {profile?.handle || "UNKNOWN_ENTITY"}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-surface-secondary/20 border border-border rounded-sm shadow-sm">
                <div className="flex items-center gap-2">
                  <Wallet size={14} className="text-slate-400" />
                  <span className="text-[10px] text-slate-500 uppercase font-mono font-bold">
                    Owner
                  </span>
                </div>
                <span className="text-xs font-mono text-accent-cyan font-bold">
                  {profile?.owned_by ? `${profile.owned_by.slice(0, 6)}...${profile.owned_by.slice(-4)}` : "N/A"}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-surface-secondary/20 border border-border rounded-sm shadow-sm">
                <div className="flex items-center gap-2">
                  <ShieldCheck size={14} className={prob > 70 ? "text-accent-red" : "text-accent-green"} />
                  <span className="text-[10px] text-slate-500 uppercase font-mono font-bold">
                    Risk Level
                  </span>
                </div>
                <span className={`text-xs font-mono font-bold ${colorClass}`}>
                  {analysis?.classification}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-surface-secondary/20 border border-border rounded-sm shadow-sm">
                <div className="flex items-center gap-2">
                  <Activity size={14} className="text-accent-cyan" />
                  <span className="text-[10px] text-slate-500 uppercase font-mono font-bold">
                    Status
                  </span>
                </div>
                <span className="text-xs font-mono text-accent-cyan font-bold uppercase">
                  {isLoading ? "Analyzing" : "Processed"}
                </span>
              </div>
            </div>
          </IndustrialCard>

          <IndustrialCard title="DETECTION_METRICS" className="flex-1">
            <div className="space-y-4">
              <div className="p-3 bg-surface-secondary/20 border border-border rounded-sm">
                <div className="flex justify-between mb-2">
                  <span className="text-[9px] font-mono text-slate-500 font-bold uppercase">Sybil Probability</span>
                  <span className={`text-[9px] font-bold font-mono ${colorClass}`}>{Math.round(prob)}%</span>
                </div>
                <div className="w-full h-1.5 bg-background overflow-hidden rounded-full border border-border">
                  <div
                    className={`h-full transition-all duration-1000 ease-out ${prob > 70 ? 'bg-accent-red' : prob > 30 ? 'bg-orange-500' : 'bg-accent-green'}`}
                    style={{ width: `${prob}%` }}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-[9px] font-mono text-slate-500 font-bold uppercase px-1">Reasoning Tokens:</span>
                <div className="flex flex-wrap gap-2">
                  {(analysis?.reasoning || []).slice(0, 4).map((r, i) => {
                    const tag = r.split(':')[0].replace('[', '').replace(']', '');
                    return (
                      <span key={i} className="px-2 py-0.5 bg-surface-secondary border border-border text-[8px] font-mono text-slate-400 uppercase">
                        {tag}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          </IndustrialCard>
        </div>
      </div>
    </div>
  );
}

export default function InspectorPage() {
  return (
    <Suspense fallback={
      <div className="h-full flex flex-col items-center justify-center gap-6">
        <Loader2 className="text-accent-cyan animate-spin" size={48} />
        <span className="text-sm font-mono text-accent-cyan animate-pulse uppercase tracking-[0.2em] font-bold">
          [SYS] INITIALIZING MODULE...
        </span>
      </div>
    }>
      <InspectorContent />
    </Suspense>
  );
}
