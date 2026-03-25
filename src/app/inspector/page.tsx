"use client";

import { Suspense, useState } from "react";
import dynamic from "next/dynamic";
import { useSearchParams, useRouter } from "next/navigation";
import { useInspectProfile } from "@/hooks/use-sybil-inference";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { BootSequenceLoader } from "@/components/ui/boot-sequence-loader";
import { resolvePictureUrl } from "@/lib/utils";
import { ProbabilityEqualizer } from "@/components/inspector/probability-equalizer";
import Image from "next/image";
import { User, AlertTriangle, Loader2, Radar, Search } from "lucide-react";

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
        placeholder="ENTER PROFILE_ID..."
        value={searchValue}
        onChange={(e) => setSearchValue(e.target.value)}
        className="bg-background border-border focus:border-accent-cyan focus:ring-accent-cyan/20 w-full rounded-sm border px-10 py-2 font-mono text-xs tracking-widest uppercase transition-all placeholder:text-slate-600 focus:ring-1 focus:outline-none"
      />
    </form>
  );
};

const UniversalGraph2D = dynamic(
  () => import("@/components/graph/universal-graph-2d"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full flex-col items-center justify-center gap-4 bg-[#050608]">
        <Loader2 className="text-accent-cyan animate-spin" size={32} />
        <span className="text-accent-cyan animate-pulse font-mono text-[10px] font-bold tracking-[0.2em] uppercase">
          INITIALIZING 2D RENDER ENGINE...
        </span>
      </div>
    ),
  }
);

// FIX 4 helper: risk label → color for UI badges
const RISK_LABEL_STYLES: Record<
  string,
  { text: string; bg: string; border: string }
> = {
  MALICIOUS: {
    text: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
  },
  HIGH_RISK: {
    text: "text-orange-400",
    bg: "bg-orange-500/10",
    border: "border-orange-500/30",
  },
  LOW_RISK: {
    text: "text-yellow-400",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
  },
  BENIGN: {
    text: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/30",
  },
};

function InspectorContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const walletId = searchParams.get("wallet");
  const { data, isLoading, isError } = useInspectProfile(walletId);

  const handleReset = () => {
    router.push("/inspector");
  };

  const getClassificationColor = (predict_label: string) => {
    switch (predict_label) {
      case "MALICIOUS":
        return "text-accent-red";
      case "HIGH_RISK":
        return "text-rose-500";
      case "LOW_RISK":
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

  const colorClass = getClassificationColor(analysis?.predict_label || "");
  const riskStyle =
    RISK_LABEL_STYLES[analysis?.predict_label || ""] ||
    RISK_LABEL_STYLES.BENIGN;

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
        <div className="flex items-center gap-4">
          <button
            onClick={handleReset}
            className="text-accent-cyan rounded-sm border border-slate-700 bg-slate-800 px-6 py-2 text-xs font-black tracking-widest uppercase italic shadow-lg transition-all hover:bg-slate-700 active:translate-y-0.5"
          >
            RESET
          </button>
        </div>
      </div>

      {/* Top Info Row */}
      <div className="grid grid-cols-12 gap-6">
        {/* Profile & Risk Merged Card */}
        <div className="col-span-4">
          <IndustrialCard title="ANALYSIS OVERVIEW" className="h-full">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div className="bg-surface-secondary border-border group relative flex h-16 w-16 items-center justify-center overflow-hidden rounded-sm border-2 shadow-inner transition-colors duration-300">
                  {profile?.picture_url ? (
                    <Image
                      src={resolvePictureUrl(profile.picture_url)}
                      alt={profile.handle || "profile"}
                      width={64}
                      height={64}
                      className="h-full w-full object-cover"
                      unoptimized
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
                  <span className="text-subtle text-[10px] tracking-tighter uppercase">
                    Handle
                  </span>
                  <span className="text-foreground max-w-[180px] truncate text-lg font-black uppercase italic">
                    {profile?.handle || "UNKNOWN_ENTITY"}
                  </span>
                </div>
              </div>

              <div className="font-mono text-xs">
                <span className="text-slate-500">PROFILE ID:</span>
                <br />
                <span className="text-accent-cyan text-[10px] font-bold break-all uppercase">
                  {walletId}
                </span>
              </div>

              {/* FIX 2: Risk label badge with correct color */}
              <div className="flex items-center justify-between border-t border-slate-800/60 pt-3">
                <div className="flex flex-col gap-1">
                  <span className="text-subtle font-mono text-[9px] font-bold uppercase">
                    Predict Label
                  </span>
                  <div
                    className={`inline-flex items-center gap-2 rounded-sm border px-3 py-1 ${riskStyle.bg} ${riskStyle.border}`}
                  >
                    <div
                      className={`h-2 w-2 rounded-full ${riskStyle.text.replace("text-", "bg-").replace("-400", "-500")} animate-pulse`}
                    />
                    <span
                      className={`text-base font-black italic ${colorClass}`}
                    >
                      {analysis?.predict_label}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </IndustrialCard>
        </div>

        {/* Probability Equalizer */}
        <div className="col-span-4">
          <ProbabilityEqualizer
            probabilities={analysis?.predict_proba || {}}
            className="h-full"
          />
        </div>

        {/* Detection Metrics */}
        <div className="col-span-4">
          <IndustrialCard title="DETECTION METRICS" className="h-full">
            <div className="flex flex-col gap-2">
              <span className="text-subtle px-1 font-mono text-[9px] font-bold uppercase">
                Reasoning:
              </span>
              <div className="flex flex-wrap gap-1">
                {(analysis?.reasoning || []).slice(0, 10).map((r, i) => (
                  <span
                    key={i}
                    className="bg-surface-secondary border-border border px-1.5 py-0.5 font-mono text-xs uppercase duration-300 dark:text-slate-400"
                  >
                    {r}
                  </span>
                ))}
              </div>
            </div>
          </IndustrialCard>
        </div>
      </div>

      {/* FIX 4: Main Graph Section - Synchronized dark background matching Discovery page */}
      <div className="min-h-[800px] flex-1">
        <div className="relative h-full w-full overflow-hidden rounded-sm border border-slate-800 bg-[#050608] shadow-2xl">
          {/* Subtle grid overlay matching discovery page aesthetic */}
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#1e293b1a_1px,transparent_1px),linear-gradient(to_bottom,#1e293b1a_1px,transparent_1px)] bg-[size:40px_40px] opacity-40" />
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_60%,#050608_100%)]" />

          {/* Live 2D Ego Graph */}
          <UniversalGraph2D
            mode="EGO"
            graphData={data?.local_graph || { nodes: [], links: [] }}
            targetId={walletId || ""}
            risk_label={data?.analysis?.predict_label}
          />

          {/* Overlays */}
          <div className="pointer-events-none absolute right-4 bottom-4 flex flex-col items-end gap-1">
            <span className="text-accent-cyan/80 animate-pulse font-mono text-[8px] font-bold uppercase">
              [ 2D RENDER ENGINE ACTIVE ]
            </span>
            <span className="font-mono text-[8px] font-bold text-slate-600 uppercase">
              Node Connections: {data?.local_graph?.nodes?.length || 0}
            </span>
            <span className="font-mono text-[8px] font-bold text-slate-600 uppercase">
              Network Depth: 2
            </span>
          </div>

          {/* Target node indicator top-left */}
          {walletId && (
            <div className="border-accent-cyan/20 pointer-events-none absolute top-4 left-4 flex items-center gap-2 border bg-black/60 px-3 py-1.5 backdrop-blur-sm">
              <div className="bg-accent-cyan h-2 w-2 animate-pulse rounded-full shadow-[0_0_6px_rgba(0,242,255,0.8)]" />
              <span className="text-accent-cyan font-mono text-[8px] font-bold tracking-widest uppercase">
                TARGET: {walletId}
              </span>
            </div>
          )}
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
