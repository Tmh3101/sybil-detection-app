"use client";

import { Suspense, useState, useMemo } from "react";
import dynamic from "next/dynamic";
import { useSearchParams, useRouter } from "next/navigation";
import { useInspectProfile } from "@/hooks/use-sybil-inference";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { BootSequenceLoader } from "@/components/ui/boot-sequence-loader";
import { resolvePictureUrl } from "@/lib/utils";
import { ProbabilityEqualizer } from "@/components/inspector/probability-equalizer";
import { LABEL_COLORS } from "@/lib/graph-constants";
import { SybilNode } from "@/types/api";
import Image from "next/image";
import {
  User,
  AlertTriangle,
  Loader2,
  Radar,
  Search,
  GitBranch,
} from "lucide-react";

const SearchForm = ({ defaultValue = "" }: { defaultValue?: string }) => {
  const router = useRouter();
  const [searchValue, setSearchValue] = useState(defaultValue);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchValue.trim())
      router.push(`/inspector?wallet=${searchValue.trim()}`);
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

function InspectorContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const walletId = searchParams.get("wallet");
  const { data, isLoading, isError } = useInspectProfile(walletId);

  // ─── Depth toggle: 1 = direct neighbors only, 2 = full ego-graph ───
  const [graphDepth, setGraphDepth] = useState<1 | 2>(1);

  const handleReset = () => {
    // Clear all search params by navigating back to base /inspector
    router.replace("/inspector");
  };

  // ─── Compute filtered local_graph based on depth (frontend filtering) ───
  const displayGraphData = useMemo(() => {
    if (!data?.local_graph || !walletId) return { nodes: [], links: [] };

    const targetId = (data.profile_info?.id || walletId).toLowerCase().trim();
    const nodesWithTargetPicture = data.local_graph.nodes.map(
      (node): SybilNode => {
        if (String(node.id).toLowerCase() !== targetId) return node;

        const existingPicture =
          typeof node.attributes?.picture_url === "string"
            ? node.attributes.picture_url
            : "";
        if (existingPicture || !data.profile_info?.picture_url) return node;

        return {
          ...node,
          attributes: {
            ...node.attributes,
            picture_url: data.profile_info.picture_url,
            handle: node.attributes?.handle || data.profile_info.handle,
          } as SybilNode["attributes"],
        };
      }
    );

    const graphData = {
      ...data.local_graph,
      nodes: nodesWithTargetPicture,
    };

    if (graphDepth === 2) return graphData;

    // Depth 1: only nodes with a direct edge to/from the target
    const tid = targetId;
    const directIds = new Set<string>([tid]);

    graphData.links.forEach((l) => {
      const s = String(
        typeof l.source === "object"
          ? (l.source as { id: string }).id
          : l.source
      ).toLowerCase();
      const t = String(
        typeof l.target === "object"
          ? (l.target as { id: string }).id
          : l.target
      ).toLowerCase();
      if (s === tid) directIds.add(t);
      if (t === tid) directIds.add(s);
    });

    const nodes = graphData.nodes.filter((n) =>
      directIds.has(String(n.id).toLowerCase())
    );
    const links = graphData.links.filter((l) => {
      const s = String(
        typeof l.source === "object"
          ? (l.source as { id: string }).id
          : l.source
      ).toLowerCase();
      const t = String(
        typeof l.target === "object"
          ? (l.target as { id: string }).id
          : l.target
      ).toLowerCase();
      return directIds.has(s) && directIds.has(t);
    });
    return { nodes, links };
  }, [data, graphDepth, walletId]);

  const analysis = data?.analysis;
  const profile = data?.profile_info;
  const riskLabel = analysis?.predict_label || "UNKNOWN";
  const riskColor = LABEL_COLORS[riskLabel] || LABEL_COLORS.UNKNOWN;

  if (!walletId) {
    return (
      <div className="flex h-full flex-col items-center justify-center">
        <div className="bg-surface/50 border-border relative flex h-[500px] w-full max-w-2xl flex-col items-center justify-center overflow-hidden rounded-sm border shadow-2xl">
          <div className="absolute inset-0 opacity-10">
            <div className="h-full w-full bg-[radial-gradient(circle_at_center,var(--accent-cyan)_0%,transparent_70%)]" />
          </div>
          <div className="relative flex flex-col items-center gap-8">
            <div className="relative">
              <Radar className="text-slate-700" size={120} strokeWidth={1} />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="bg-accent-cyan/40 h-2 w-2 animate-ping rounded-full" />
              </div>
            </div>
            <div className="flex flex-col items-center text-center">
              <h2 className="text-foreground/40 mb-2 text-2xl font-black tracking-tighter uppercase italic">
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

  if (isLoading) return <BootSequenceLoader />;

  if (isError) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-6">
        <div className="border-accent-red/20 bg-accent-red/5 max-w-lg rounded-lg border-2 p-8 text-center backdrop-blur-sm">
          <AlertTriangle className="text-accent-red mx-auto mb-4" size={48} />
          <h2 className="text-accent-red mb-2 text-xl font-black tracking-tighter uppercase italic">
            [ERR] Failed to Fetch Target Data
          </h2>
          <p className="text-subtle mb-6 font-mono text-sm leading-relaxed tracking-widest uppercase">
            Verify the ID and try again.
          </p>
          <SearchForm defaultValue={walletId || ""} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-6">
      {/* ── Header ── */}
      <div className="mb-2 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col">
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Profile <span className="text-accent-cyan">Inspector</span>
          </h2>
          <span className="text-subtle">
            Target Identification & Risk Assessment Module
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleReset}
            className="text-accent-cyan rounded-sm border border-slate-700 bg-slate-800 px-5 py-2 text-xs font-black tracking-widest whitespace-nowrap uppercase italic shadow-lg transition-all hover:bg-slate-700 active:translate-y-0.5"
          >
            RESET
          </button>
        </div>
      </div>

      {/* ── Info Cards Row ── */}
      <div className="grid grid-cols-12 gap-5">
        {/* Analysis Overview */}
        <div className="col-span-4">
          <IndustrialCard title="ANALYSIS OVERVIEW" className="h-full">
            <div className="flex flex-col gap-4">
              <div className="flex items-center gap-4">
                <div
                  className="relative flex h-16 w-16 flex-shrink-0 items-center justify-center overflow-hidden rounded-sm border-2"
                  style={{
                    borderColor: riskColor + "55",
                    backgroundColor: riskColor + "0a",
                  }}
                >
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
                    <User size={28} style={{ color: riskColor + "88" }} />
                  )}
                </div>
                <div className="flex min-w-0 flex-col">
                  <span className="text-subtle text-[9px] tracking-tighter uppercase">
                    Handle
                  </span>
                  <span
                    className="max-w-[180px] truncate text-lg font-black uppercase italic"
                    style={{ color: riskColor }}
                  >
                    {profile?.handle || "UNKNOWN"}
                  </span>
                </div>
              </div>

              <div className="font-mono text-[9px]">
                <span className="text-slate-500">PROFILE ID:</span>
                <br />
                <span
                  className="font-bold break-all"
                  style={{ color: riskColor + "cc" }}
                >
                  {walletId}
                </span>
              </div>

              <div
                className="flex items-center gap-2 rounded-sm border px-3 py-2"
                style={{
                  borderColor: riskColor + "44",
                  backgroundColor: riskColor + "08",
                }}
              >
                <div
                  className="h-2 w-2 flex-shrink-0 animate-pulse rounded-full"
                  style={{
                    backgroundColor: riskColor,
                    boxShadow: `0 0 6px ${riskColor}88`,
                  }}
                />
                <span
                  className="text-base font-black italic"
                  style={{ color: riskColor }}
                >
                  {riskLabel}
                </span>
              </div>
            </div>
          </IndustrialCard>
        </div>

        {/* Probability Distribution */}
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
              <span className="text-subtle px-1 text-[9px] font-bold uppercase">
                Reasoning:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {(analysis?.reasoning || []).map((r, i) => (
                  <span
                    key={i}
                    className="border px-1.5 py-0.5 font-mono text-[9px] uppercase"
                    style={{
                      borderColor: riskColor + "33",
                      color: riskColor + "aa",
                      backgroundColor: riskColor + "08",
                    }}
                  >
                    {r}
                  </span>
                ))}
              </div>
            </div>
          </IndustrialCard>
        </div>
      </div>

      {/* ── Graph Area ── */}
      <div className="min-h-[800px] flex-1">
        {/* ── FIX: Dark background matching Discovery page ── */}
        <div className="relative h-full w-full overflow-hidden rounded-sm border border-slate-800/70 bg-[#050608] shadow-2xl">
          {/* Subtle grid */}
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,#1e293b12_1px,transparent_1px),linear-gradient(to_bottom,#1e293b12_1px,transparent_1px)] bg-[size:40px_40px]" />

          {/* ── Depth toggle overlay ── */}
          <div className="absolute top-4 left-4 z-20 flex items-center gap-2">
            <div className="flex items-center gap-1.5 border border-slate-700/80 bg-black/80 px-1.5 py-1 backdrop-blur-sm">
              <GitBranch size={10} className="text-slate-500" />
              <span className="font-mono text-[8px] font-bold text-slate-500 uppercase">
                Depth
              </span>
              {([1, 2] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setGraphDepth(d)}
                  className="px-2 py-0.5 font-mono text-[9px] font-bold transition-all"
                  style={{
                    color: graphDepth === d ? "#00f2ff" : "#334155",
                    background:
                      graphDepth === d ? "rgba(0,242,255,0.1)" : "transparent",
                    border: `1px solid ${graphDepth === d ? "rgba(0,242,255,0.3)" : "transparent"}`,
                  }}
                >
                  {d}
                </button>
              ))}
            </div>
            <div className="border border-slate-700/70 bg-black/70 px-2.5 py-1 backdrop-blur-sm">
              <span className="font-mono text-[8px] text-slate-600">
                {displayGraphData.nodes.length} nodes ·{" "}
                {displayGraphData.links.length} edges
              </span>
            </div>
          </div>

          <UniversalGraph2D
            mode="EGO"
            graphData={displayGraphData}
            targetId={profile?.id || walletId || ""}
            risk_label={riskLabel as import("@/types/api").RiskClassification}
            depthFilter={graphDepth}
          />

          {/* Overlays bottom-right (handled by graph component's zoom controls) */}
          <div className="pointer-events-none absolute right-16 bottom-6 flex flex-col items-end gap-1">
            <span className="text-accent-cyan/60 font-mono text-[8px] font-bold uppercase">
              [ 2D RENDER ENGINE ACTIVE ]
            </span>
            <span className="font-mono text-[8px] font-bold text-slate-700 uppercase">
              Depth: {graphDepth} · {displayGraphData.nodes.length} nodes
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function InspectorWrapper() {
  const searchParams = useSearchParams();
  const walletId = searchParams.get("wallet") || "idle";
  return <InspectorContent key={walletId} />;
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
      <InspectorWrapper />
    </Suspense>
  );
}
