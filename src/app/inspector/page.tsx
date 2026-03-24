"use client";

import { Suspense, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams, useRouter } from "next/navigation";
import { useInspectProfile } from "@/hooks/use-sybil-inference";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { TerminalLog } from "@/components/ui/terminal-log";
import { BootSequenceLoader } from "@/components/ui/boot-sequence-loader";
import Image from "next/image";
import {
  Wallet,
  ShieldCheck,
  User,
  Activity,
  AlertTriangle,
  Loader2,
  Radar,
  Search,
} from "lucide-react";

const SearchForm = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [searchValue, setSearchValue] = useState(
    searchParams.get("wallet") || ""
  );

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchValue.trim()) {
      router.push(`/inspector?wallet=${searchValue.trim()}`);
    }
  };

  return (
    <form onSubmit={handleSearch} className="group relative w-full max-w-md">
      <Search
        className="group-focus-within:text-accent-cyan absolute top-1/2 left-3 -translate-y-1/2 text-slate-500 transition-colors"
        size={16}
      />
      <input
        type="text"
        placeholder="ENTER WALLET_ID OR HANDLE..."
        value={searchValue}
        onChange={(e) => setSearchValue(e.target.value)}
        className="bg-background border-border focus:border-accent-cyan focus:ring-accent-cyan/20 w-full rounded-sm border px-10 py-2 font-mono text-xs tracking-widest uppercase transition-all placeholder:text-slate-600 focus:ring-1 focus:outline-none"
      />
    </form>
  );
};

const EgoGraph2D = dynamic(() => import("@/components/graph/ego-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-black/20">
      <Loader2 className="text-accent-cyan animate-spin" size={32} />
      <span className="text-accent-cyan animate-pulse font-mono text-[10px] font-bold tracking-[0.2em] uppercase">
        INITIALIZING 2D RENDER ENGINE...
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
      <div className="flex h-full flex-col items-center justify-center">
        <div className="bg-surface/50 border-border relative flex h-[500px] w-full max-w-2xl flex-col items-center justify-center overflow-hidden rounded-sm border shadow-2xl transition-colors duration-300">
          <div className="absolute inset-0 opacity-10 dark:opacity-20">
            <div className="h-full w-full bg-[radial-gradient(circle_at_center,var(--accent-cyan)_0%,transparent_70%)]" />
            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(var(--border-rgb),0.5)_1px,transparent_1px),linear-gradient(to_bottom,rgba(var(--border-rgb),0.5)_1px,transparent_1px)] bg-[size:30px_30px] dark:bg-[size:40px_40px]" />
          </div>

          <div className="relative flex flex-col items-center gap-8">
            <div className="relative">
              <Radar
                className="text-slate-300 dark:text-slate-700"
                size={120}
                strokeWidth={1}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-accent-cyan/40 h-2 w-2 animate-ping rounded-full" />
              </div>
            </div>

            <div className="flex flex-col items-center text-center">
              <h2 className="text-foreground/40 mb-2 text-2xl font-black tracking-tighter uppercase italic dark:text-slate-400">
                [ SYSTEM STANDBY ]
              </h2>
              <div className="mb-6 flex items-center gap-3">
                <span className="bg-border h-[1px] w-8" />
                <p className="text-accent-cyan/60 font-mono text-xs tracking-[0.3em] uppercase">
                  Awaiting Target Input
                </p>
                <span className="bg-border h-[1px] w-8" />
              </div>

              <SearchForm />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return <BootSequenceLoader />;
  }

  if (isError) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-6">
        <div className="border-accent-red/20 bg-accent-red/5 max-w-lg rounded-lg border-2 p-8 text-center backdrop-blur-sm">
          <AlertTriangle className="text-accent-red mx-auto mb-4" size={48} />
          <h2 className="text-accent-red mb-2 text-xl font-black tracking-tighter uppercase italic">
            [ERR] Failed to Fetch Target Data
          </h2>
          <p className="text-subtle mb-6 font-mono text-sm leading-relaxed tracking-widest uppercase">
            The sybil engine encountered an error while analyzing the target.
            Please verify the ID and try again.
          </p>
          <div className="flex justify-center">
            <SearchForm />
          </div>
        </div>
      </div>
    );
  }

  const analysis = data?.analysis;
  const profile = data?.profile_info;
  const prob = (analysis?.sybil_probability || 0) * 100;
  const colorClass = getClassificationColor(analysis?.classification || "");

  return (
    <div className="flex h-full flex-col gap-6">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex flex-col">
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Profile <span className="text-accent-cyan">Inspector</span>
          </h2>
          <span className="text-subtle">
            Target Identification & Risk Assessment Module
          </span>
        </div>

        <div className="flex flex-1 justify-center px-12">
          <SearchForm />
        </div>

        <div className="flex items-center gap-4">
          <div className="bg-surface border-border rounded-sm border px-4 py-2 font-mono text-xs shadow-sm">
            ID:{" "}
            <span className="text-accent-cyan font-bold uppercase">
              {walletId.slice(0, 6)}...{walletId.slice(-4)}
            </span>
          </div>
          <button className="bg-accent-red active:shadow-neo-concave rounded-sm px-6 py-2 text-xs font-black tracking-widest text-white uppercase italic shadow-lg transition-all hover:brightness-110 active:translate-y-0.5 dark:text-black">
            QUARANTINE
          </button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-12 gap-6">
        {/* Left Column: Risk Score & Logs */}
        <div className="col-span-3 flex flex-col gap-6 overflow-hidden">
          <IndustrialCard
            title="RISK_SCORE_GAUGE"
            className="flex flex-col items-center"
          >
            <div className="relative flex h-48 w-48 items-center justify-center">
              {/* SVG Gauge */}
              <svg className="h-full w-full -rotate-90 transform">
                <circle
                  cx="96"
                  cy="96"
                  r="80"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className="text-slate-100 dark:text-slate-800"
                />
                <circle
                  cx="96"
                  cy="96"
                  r="80"
                  stroke="currentColor"
                  strokeWidth="12"
                  fill="transparent"
                  strokeDasharray={502.4}
                  strokeDashoffset={502.4 * (1 - prob / 100)}
                  className={`${colorClass} transition-all duration-1000 ease-out`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span
                  className={`text-5xl font-black tracking-tighter italic ${colorClass}`}
                >
                  {Math.round(prob)}%
                </span>
                <span className="text-subtle -mt-1 text-[10px] font-bold tracking-widest uppercase">
                  {analysis?.classification}
                </span>
              </div>
            </div>
            <div className="mt-4 grid w-full grid-cols-2 gap-4">
              <div className="bg-surface-secondary/50 border-border flex flex-col items-center border p-2 shadow-inner">
                <span className="text-[8px] font-bold text-slate-500 uppercase">
                  Classification
                </span>
                <span className={`text-xs font-bold ${colorClass}`}>
                  {analysis?.classification}
                </span>
              </div>
              <div className="bg-surface-secondary/50 border-border flex flex-col items-center border p-2 shadow-inner">
                <span className="text-[8px] font-bold text-slate-500 uppercase">
                  Certainty
                </span>
                <span className="text-accent-cyan text-xs font-bold">
                  {prob > 50 ? Math.round(prob) : Math.round(100 - prob)}%
                </span>
              </div>
            </div>
          </IndustrialCard>

          <IndustrialCard
            title="LIVE_DETECTION_STREAM"
            className="min-h-0 flex-1 p-0"
          >
            <TerminalLog
              className="h-full border-0 bg-transparent"
              logs={analysis?.reasoning || []}
            />
          </IndustrialCard>
        </div>

        {/* Middle Column: Ego Graph */}
        <div className="col-span-6">
          <div className="border-border bg-background group relative h-full w-full overflow-hidden rounded-sm border shadow-2xl transition-colors duration-300">
            {/* Background Grid */}
            <div className="pointer-events-none absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] dark:opacity-20" />
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(var(--border-rgb),0.3)_1px,transparent_1px),linear-gradient(to_bottom,rgba(var(--border-rgb),0.3)_1px,transparent_1px)] bg-[size:40px_40px] opacity-50 dark:opacity-20" />

            {/* Live 2D Ego Graph */}
            <EgoGraph2D
              graphData={data?.local_graph || { nodes: [], links: [] }}
              targetId={walletId || ""}
              classification={data?.analysis?.classification}
            />

            {/* Overlays */}
            <div className="pointer-events-none absolute top-4 left-4 flex gap-2">
              <div className="text-accent-cyan bg-surface/80 border-border border px-2 py-1 font-mono text-[9px] font-bold uppercase backdrop-blur-sm">
                EGO_GRAPH_v3_CANVAS
              </div>
              <div className="bg-surface/80 border-border text-subtle border px-2 py-1 font-mono text-[9px] uppercase backdrop-blur-sm">
                REAL_TIME_2D_VIEW
              </div>
            </div>

            <div className="pointer-events-none absolute right-4 bottom-4 flex flex-col items-end gap-1">
              <span className="text-subtle font-mono text-[8px] font-bold uppercase">
                Node Connections: {data?.local_graph?.nodes?.length || 0}
              </span>
              <span className="text-subtle font-mono text-[8px] font-bold uppercase">
                Network Depth: 4
              </span>
            </div>
          </div>
        </div>

        {/* Right Column: Profile Data */}
        <div className="col-span-3 flex flex-col gap-6 overflow-hidden">
          <IndustrialCard title="ENTITY_IDENTITY">
            <div className="mb-6 flex items-center gap-4">
              <div className="bg-surface-secondary border-border group relative flex h-16 w-16 items-center justify-center overflow-hidden rounded-sm border-2 shadow-inner transition-colors duration-300">
                {profile?.picture_url ? (
                  <Image
                    src={profile.picture_url}
                    alt={profile.handle}
                    width={64}
                    height={64}
                    className="h-full w-full object-cover"
                  />
                ) : (
                  <User
                    size={32}
                    className="group-hover:text-accent-cyan text-slate-400 transition-colors"
                  />
                )}
                <div className="bg-accent-cyan/10 absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100" />
              </div>
              <div className="flex flex-col">
                <span className="text-subtle tracking-tighter">Alias</span>
                <span className="text-foreground max-w-[180px] truncate text-lg font-black uppercase italic">
                  {profile?.handle || "UNKNOWN_ENTITY"}
                </span>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-surface-secondary/40 border-border flex items-center justify-between rounded-sm border p-3 shadow-sm transition-colors duration-300">
                <div className="flex items-center gap-2">
                  <Wallet size={14} className="text-slate-400" />
                  <span className="text-subtle font-mono text-[10px] font-bold uppercase">
                    Owner
                  </span>
                </div>
                <span className="text-accent-cyan font-mono text-xs font-bold">
                  {profile?.owned_by
                    ? `${profile.owned_by.slice(0, 6)}...${profile.owned_by.slice(-4)}`
                    : "N/A"}
                </span>
              </div>
              <div className="bg-surface-secondary/40 border-border flex items-center justify-between rounded-sm border p-3 shadow-sm transition-colors duration-300">
                <div className="flex items-center gap-2">
                  <ShieldCheck
                    size={14}
                    className={
                      prob > 70 ? "text-accent-red" : "text-accent-green"
                    }
                  />
                  <span className="text-subtle font-mono text-[10px] font-bold uppercase">
                    Risk Level
                  </span>
                </div>
                <span className={`font-mono text-xs font-bold ${colorClass}`}>
                  {analysis?.classification}
                </span>
              </div>
              <div className="bg-surface-secondary/40 border-border flex items-center justify-between rounded-sm border p-3 shadow-sm transition-colors duration-300">
                <div className="flex items-center gap-2">
                  <Activity size={14} className="text-accent-cyan" />
                  <span className="text-subtle font-mono text-[10px] font-bold uppercase">
                    Status
                  </span>
                </div>
                <span className="text-accent-cyan font-mono text-xs font-bold uppercase">
                  {isLoading ? "Analyzing" : "Processed"}
                </span>
              </div>
            </div>
          </IndustrialCard>

          <IndustrialCard title="DETECTION_METRICS" className="flex-1">
            <div className="space-y-4">
              <div className="bg-surface-secondary/40 border-border rounded-sm border p-3 shadow-sm transition-colors duration-300">
                <div className="mb-2 flex justify-between">
                  <span className="text-subtle font-mono text-[9px] font-bold uppercase">
                    Sybil Probability
                  </span>
                  <span
                    className={`font-mono text-[9px] font-bold ${colorClass}`}
                  >
                    {Math.round(prob)}%
                  </span>
                </div>
                <div className="bg-background border-border h-1.5 w-full overflow-hidden rounded-full border">
                  <div
                    className={`h-full transition-all duration-1000 ease-out ${prob > 70 ? "bg-accent-red" : prob > 30 ? "bg-orange-500" : "bg-accent-green"}`}
                    style={{ width: `${prob}%` }}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <span className="text-subtle px-1 font-mono text-[9px] font-bold uppercase">
                  Reasoning Tokens:
                </span>
                <div className="flex flex-wrap gap-2">
                  {(analysis?.reasoning || []).slice(0, 4).map((r, i) => {
                    const tag = r
                      .split(":")[0]
                      .replace("[", "")
                      .replace("]", "");
                    return (
                      <span
                        key={i}
                        className="bg-surface-secondary border-border border px-2 py-0.5 font-mono text-[8px] text-slate-500 uppercase transition-colors duration-300 dark:text-slate-400"
                      >
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
    <Suspense
      fallback={
        <div className="flex h-full flex-col items-center justify-center gap-6">
          <Loader2 className="text-accent-cyan animate-spin" size={48} />
          <span className="text-accent-cyan animate-pulse font-mono text-sm font-bold tracking-[0.2em] uppercase">
            [SYS] INITIALIZING MODULE...
          </span>
        </div>
      }
    >
      <InspectorContent />
    </Suspense>
  );
}
