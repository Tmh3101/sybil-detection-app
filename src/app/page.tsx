"use client";

// import { IndustrialCard } from "@/components/ui/industrial-card";
import {
  Radar,
  FlaskConical,
  // ShieldAlert,
  // Activity,
  // Cpu,
  // HardDrive,
} from "lucide-react";
import Link from "next/link";

export default function HomePage() {
  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-2">
        <h2 className="text-foreground text-5xl font-black tracking-tighter uppercase italic">
          SYSTEM <span className="text-accent-cyan">OVERVIEW</span>
        </h2>
        <p className="max-w-2xl font-mono text-sm leading-relaxed font-bold tracking-wider text-slate-500 uppercase dark:text-slate-400">
          Sybil Engine is a high-performance detection and network discovery
          platform designed for large-scale sybil cluster identification and
          individual profile risk assessment.
        </p>
      </div>

      {/* <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <IndustrialCard
          title="THREAT_LEVEL"
          className="border-l-accent-red border-l-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-accent-red text-4xl font-black tracking-tighter italic">
                CRITICAL
              </span>
              <span className="text-subtle mt-1 font-bold">
                Active clusters detected
              </span>
            </div>
            <ShieldAlert size={48} className="text-accent-red opacity-20" />
          </div>
        </IndustrialCard>

        <IndustrialCard
          title="CORE_UTILIZATION"
          className="border-l-accent-cyan border-l-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-foreground text-4xl font-black tracking-tighter italic">
                42.8%
              </span>
              <span className="text-subtle mt-1 font-bold">
                GAE Training in progress
              </span>
            </div>
            <Cpu size={48} className="text-accent-cyan opacity-20" />
          </div>
        </IndustrialCard>

        <IndustrialCard
          title="NODE_CAPACITY"
          className="border-l-accent-green border-l-4"
        >
          <div className="flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-foreground text-4xl font-black tracking-tighter italic">
                1.4M
              </span>
              <span className="text-subtle mt-1 font-bold">
                Indexed addresses
              </span>
            </div>
            <HardDrive size={48} className="text-accent-green opacity-20" />
          </div>
        </IndustrialCard>
      </div> */}

      <div className="mt-4 grid gap-8">
        <div className="flex flex-col gap-4">
          <h3 className="text-subtle border-border border-b pb-2 font-bold">
            Available Modules
          </h3>
          <div className="grid grid-cols-1 gap-4">
            <Link href="/inspector">
              <div className="group bg-surface border-border hover:border-accent-cyan relative cursor-pointer overflow-hidden rounded-sm border p-6 transition-all">
                <div className="group-hover:text-accent-cyan absolute top-0 right-0 p-2 opacity-10 transition-all group-hover:opacity-100">
                  <Radar size={32} />
                </div>
                <h4 className="text-foreground group-hover:text-accent-cyan text-xl font-bold tracking-widest uppercase italic transition-colors">
                  Profile Inspector
                </h4>
                <p className="mt-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                  Analyze individual wallets for sybil behavior and risk
                  scoring.
                </p>
              </div>
            </Link>
            <Link href="/discovery">
              <div className="group bg-surface border-border hover:border-accent-cyan relative cursor-pointer overflow-hidden rounded-sm border p-6 transition-all">
                <div className="group-hover:text-accent-cyan absolute top-0 right-0 p-2 opacity-10 transition-all group-hover:opacity-100">
                  <FlaskConical size={32} />
                </div>
                <h4 className="text-foreground group-hover:text-accent-cyan text-xl font-bold tracking-widest uppercase italic transition-colors">
                  Discovery Lab
                </h4>
                <p className="mt-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                  Identify large scale community clusters and fraudulent
                  networks.
                </p>
              </div>
            </Link>
          </div>
        </div>

        {/* <div className="flex flex-col gap-4">
          <h3 className="text-subtle border-border border-b pb-2 font-bold">
            System Diagnostics
          </h3>
          <div className="bg-surface border-border flex flex-1 flex-col gap-6 rounded-sm border p-6">
            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-[10px] font-bold uppercase">
                <span className="text-slate-500">Memory Integrity</span>
                <span className="text-accent-cyan">98.2%</span>
              </div>
              <div className="bg-background border-border h-2 w-full overflow-hidden rounded-full border shadow-inner">
                <div className="bg-accent-cyan h-full w-[98.2%] shadow-[0_0_8px_rgba(var(--accent-cyan),0.6)]" />
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-[10px] font-bold uppercase">
                <span className="text-slate-500">Heuristic Accuracy</span>
                <span className="text-accent-green">94.5%</span>
              </div>
              <div className="bg-background border-border h-2 w-full overflow-hidden rounded-full border shadow-inner">
                <div className="bg-accent-green h-full w-[94.5%] shadow-[0_0_8px_rgba(var(--accent-green),0.6)]" />
              </div>
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex justify-between font-mono text-[10px] font-bold uppercase">
                <span className="text-slate-500">Database Latency</span>
                <span className="text-accent-cyan">12ms</span>
              </div>
              <div className="bg-background border-border h-2 w-full overflow-hidden rounded-full border shadow-inner">
                <div className="bg-accent-cyan h-full w-[12%] shadow-[0_0_8px_rgba(var(--accent-cyan),0.6)]" />
              </div>
            </div>

            <div className="border-border mt-auto flex items-center gap-2 border-t pt-4 font-mono text-[10px] font-bold text-slate-500 uppercase">
              <Activity size={14} className="text-accent-green animate-pulse" />
              <span>Diagnostic Sync: Nominal [Last check: 0.02s ago]</span>
            </div>
          </div>
        </div> */}
      </div>
    </div>
  );
}
