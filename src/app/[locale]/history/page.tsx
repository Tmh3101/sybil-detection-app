"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { InspectorHistoryTable } from "@/components/history/inspector-history-table";
import { DiscoveryHistoryTable } from "@/components/history/discovery-history-table";
import { History, Search, FlaskConical } from "lucide-react";

export default function HistoryPage() {
  const t = useTranslations("HistoryPage");
  const [activeTab, setActiveTab] = useState<"inspector" | "discovery">("inspector");

  return (
    <div className="flex h-full flex-col gap-6 p-6">
      {/* Header */}
      <div className="mb-2 flex flex-col">
        <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic flex items-center gap-3">
          <History className="text-accent-cyan" size={28} />
          {t("page_title")}{" "}
          <span className="text-accent-cyan">
            {t("page_title_highlight")}
          </span>
        </h2>
        <span className="text-subtle mt-1 font-mono text-xs tracking-widest">
          {t("page_subtitle")}
        </span>
      </div>

      {/* Tabs Control */}
      <div className="flex border-b border-border">
        <button
          onClick={() => setActiveTab("inspector")}
          className={`flex items-center gap-2 px-6 py-3 font-mono text-[11px] font-bold tracking-widest uppercase transition-all ${
            activeTab === "inspector"
              ? "border-b-2 border-accent-cyan text-accent-cyan bg-surface-secondary/50"
              : "text-slate-500 hover:bg-surface-secondary/30 hover:text-slate-300"
          }`}
        >
          <Search size={14} />
          {t("tab_inspector")}
        </button>
        <button
          onClick={() => setActiveTab("discovery")}
          className={`flex items-center gap-2 px-6 py-3 font-mono text-[11px] font-bold tracking-widest uppercase transition-all ${
            activeTab === "discovery"
              ? "border-b-2 border-accent-cyan text-accent-cyan bg-surface-secondary/50"
              : "text-slate-500 hover:bg-surface-secondary/30 hover:text-slate-300"
          }`}
        >
          <FlaskConical size={14} />
          {t("tab_discovery")}
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1">
        {activeTab === "inspector" ? (
          <div className="animate-in fade-in duration-300">
            <InspectorHistoryTable />
          </div>
        ) : (
          <div className="animate-in fade-in duration-300">
            <DiscoveryHistoryTable />
          </div>
        )}
      </div>
    </div>
  );
}
