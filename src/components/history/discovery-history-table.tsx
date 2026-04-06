"use client";

import { useDiscoveryHistory } from "@/hooks/use-sybil-discovery";
import { useTranslations, useLocale } from "next-intl";

export function DiscoveryHistoryTable() {
  const t = useTranslations("HistoryPage");
  const locale = useLocale();
  const { data, isLoading, isError } = useDiscoveryHistory();

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
        [ERR] FAILED TO FETCH DISCOVERY HISTORY
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
              {t("th_run_time")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_analyzed_period")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_clusters")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_nodes")}
            </th>
            <th className="px-4 py-3 font-bold tracking-widest text-slate-400 uppercase">
              {t("th_edges")}
            </th>
            <th className="px-4 py-3 text-right font-bold tracking-widest text-slate-400 uppercase">
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
                  {formatDate(row.timestamp)}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.start_date && row.end_date ? (
                    <>
                      {formatDate(row.start_date)} →{" "}
                      {formatDate(row.end_date)}
                    </>
                  ) : (
                    "---"
                  )}
                </td>
                <td className="px-4 py-3 text-slate-300">
                  {row.cluster_count}
                </td>
                <td className="px-4 py-3 text-slate-300">{row.node_count}</td>
                <td className="px-4 py-3 text-slate-300">{row.edge_count}</td>
                <td className="px-4 py-3 text-right">
                  <span
                    className={`rounded-sm border px-2 py-1 text-[10px] font-bold tracking-widest uppercase ${
                      row.status === "COMPLETED"
                        ? "border-green-500/40 bg-green-500/10 text-green-400"
                        : row.status === "FAILED"
                          ? "border-red-500/40 bg-red-500/10 text-red-400"
                          : "border-yellow-500/40 bg-yellow-500/10 text-yellow-400"
                    }`}
                  >
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
