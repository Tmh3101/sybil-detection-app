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
    const date = new Date(dateString);
    if (isNaN(date.getTime())) {
      return dateString || "---";
    }

    return new Intl.DateTimeFormat(locale, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);
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
      <div className="flex h-64 items-center justify-center font-mono text-sm tracking-widest text-red-500">
        [ERR] FAILED TO FETCH INSPECTOR HISTORY
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center font-mono text-sm tracking-widest text-slate-500 uppercase">
        {t("no_data")}
      </div>
    );
  }

  return (
    <div className="border-border bg-surface/50 w-full overflow-x-auto rounded-sm border">
      <table className="w-full text-left font-mono text-xs">
        <thead className="bg-surface-secondary border-border border-b">
          <tr>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_timestamp")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_target_wallet")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_prediction")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_confidence")}
            </th>
            <th className="px-4 py-3 text-right font-bold tracking-widest text-slate-400 uppercase">
              {t("th_action")}
            </th>
          </tr>
        </thead>
        <tbody className="divide-border divide-y">
          {data.map((row) => {
            const riskColor =
              LABEL_COLORS[row.predict_label] || LABEL_COLORS.UNKNOWN;

            return (
              <tr
                key={row.id}
                className="hover:bg-surface-secondary/50 transition-colors"
              >
                <td className="px-4 py-3 text-slate-300">
                  {formatDate(row.timestamp)}
                </td>
                <td
                  className="max-w-[200px] truncate px-4 py-3 font-bold text-slate-300"
                  title={row.target_address}
                >
                  {row.target_address}
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
                  {(row.confidence_score * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() =>
                      router.push(`/inspector?wallet=${row.target_address}`)
                    }
                    className="text-accent-cyan inline-flex items-center gap-2 rounded-sm border border-slate-700 bg-slate-800 px-3 py-1.5 text-[10px] font-black tracking-widest uppercase shadow-sm transition-all hover:bg-slate-700 active:translate-y-[1px]"
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
