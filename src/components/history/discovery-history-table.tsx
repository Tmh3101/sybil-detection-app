"use client";

import { useDiscoveryHistory } from "@/hooks/use-sybil-discovery";
import { useTranslations, useLocale } from "next-intl";

export function DiscoveryHistoryTable() {
  const t = useTranslations("HistoryPage");
  const locale = useLocale();
  const { data, isLoading, isError } = useDiscoveryHistory();

  const formatDate = (dateString: string) => {
    return new Intl.DateTimeFormat(locale, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
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
        [ERR] FAILED TO FETCH DISCOVERY HISTORY
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
              {t("th_run_time")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_analyzed_period")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_clusters")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_nodes")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400">
              {t("th_edges")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest uppercase text-slate-400 text-right">
              Status
            </th>
          </tr>
        </thead>
        <tbody className="divide-border divide-y">
          {data.map((row) => {
            return (
              <tr
                key={row.id}
                className="hover:bg-surface-secondary/50 transition-colors"
              >
                <td className="px-4 py-3 text-slate-300">
                  {formatDate(row.run_time)}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.analyzed_period.start_date} → {row.analyzed_period.end_date}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.clusters_found}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.total_nodes}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.total_edges}
                </td>
                <td className="px-4 py-3 text-right">
                  <span className={`px-2 py-1 text-[10px] font-bold tracking-widest uppercase rounded-sm border ${
                    row.status === 'COMPLETED' 
                      ? 'border-green-500/40 bg-green-500/10 text-green-400' 
                      : row.status === 'FAILED'
                      ? 'border-red-500/40 bg-red-500/10 text-red-400'
                      : 'border-yellow-500/40 bg-yellow-500/10 text-yellow-400'
                  }`}>
                    {row.status}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
