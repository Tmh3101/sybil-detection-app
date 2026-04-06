"use client";

import { useInspectorHistory } from "@/hooks/use-sybil-inference";
import { useRouter } from "@/i18n/routing";
import { useTranslations, useLocale } from "next-intl";
import { LABEL_COLORS } from "@/lib/graph-constants";
import { Search } from "lucide-react";

export function InspectorHistoryTable() {
  const t = useTranslations("HistoryPage");
  const tRisk = useTranslations("RiskLabels");
  const locale = useLocale();
  const router = useRouter();
  const { data, isLoading, isError } = useInspectorHistory();

  const formatDate = (dateString: string) => {
    return new Intl.DateTimeFormat(locale, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(new Date(dateString));
  };


  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-accent-cyan animate-pulse font-mono text-sm tracking-widest uppercase">
          LOADING...
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex h-64 items-center justify-center text-red-500 font-mono text-sm tracking-widest">
        [ERR] FAILED TO FETCH INSPECTOR HISTORY
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-slate-500 font-mono text-sm tracking-widest uppercase">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="border-border bg-surface/50 w-full overflow-x-auto rounded-sm border">
      <table className="w-full text-left font-mono text-xs">
        <thead className="bg-surface-secondary border-border border-b">
          <tr>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_timestamp")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_target_wallet")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_prediction")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_confidence")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400 text-right">
              {t("th_action")}
            </th>
          </tr>
        </thead>
        <tbody className="divide-border divide-y">
          {data.map((row) => {
            const riskColor = LABEL_COLORS[row.predict_label] || LABEL_COLORS.UNKNOWN;
            
            return (
              <tr
                key={row.id}
                className="hover:bg-surface-secondary/50 transition-colors"
              >
                <td className="px-4 py-3 text-slate-300">
                  {formatDate(row.timestamp)}
                </td>
                <td className="px-4 py-3 text-slate-300 font-bold truncate max-w-[200px]" title={row.target_wallet}>
                  {row.target_wallet}
                </td>
                <td className="px-4 py-3">
                  <div
                    className="inline-flex items-center gap-2 rounded-sm border px-2 py-1"
                    style={{
                      borderColor: riskColor + "44",
                      backgroundColor: riskColor + "08",
                    }}
                  >
                    <div
                      className="h-1.5 w-1.5 flex-shrink-0 rounded-full"
                      style={{
                        backgroundColor: riskColor,
                        boxShadow: `0 0 6px ${riskColor}88`,
                      }}
                    />
                    <span
                      className="text-[10px] font-bold uppercase italic"
                      style={{ color: riskColor }}
                    >
                      {tRisk(row.predict_label as keyof typeof tRisk.raw)}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {(row.confidence * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => router.push(`/inspector?wallet=${row.target_wallet}`)}
                    className="inline-flex items-center gap-2 rounded-sm border border-slate-700 bg-slate-800 px-3 py-1.5 text-[10px] font-black tracking-widest uppercase text-accent-cyan shadow-sm transition-all hover:bg-slate-700 active:translate-y-[1px]"
                  >
                    <Search size={12} />
                    {t("btn_investigate")}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
