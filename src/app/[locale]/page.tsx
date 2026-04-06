"use client";

import { Radar, FlaskConical, BarChart3 } from "lucide-react";
import Link from "next/link";
import { useTranslations } from "next-intl";

export default function HomePage() {
  const t = useTranslations("HomePage");

  return (
    <div className="flex flex-col gap-10 pt-8">
      <div className="flex flex-col gap-2">
        <h2 className="text-foreground text-5xl font-black tracking-tighter uppercase italic">
          {t("title_system")}{" "}
          <span className="text-accent-cyan">{t("title_overview")}</span>
        </h2>
        <p className="max-w-2xl font-mono text-sm leading-relaxed font-bold tracking-wider text-slate-500 uppercase dark:text-slate-400">
          {t("subtitle")}
        </p>
      </div>

      <div className="mt-4 grid gap-8">
        <div className="flex flex-col gap-4">
          <h3 className="text-subtle border-border border-b pb-2 font-bold">
            {t("available_modules")}
          </h3>
          <div className="grid grid-cols-1 gap-4">
            <Link href="/inspector">
              <div className="group bg-surface border-border hover:border-accent-cyan relative cursor-pointer overflow-hidden rounded-sm border p-6 transition-all">
                <div className="group-hover:text-accent-cyan absolute top-0 right-0 p-2 opacity-10 transition-all group-hover:opacity-100">
                  <Radar size={32} />
                </div>
                <h4 className="text-foreground group-hover:text-accent-cyan text-xl font-bold tracking-widest uppercase italic transition-colors">
                  {t("inspector_title")}
                </h4>
                <p className="mt-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                  {t("inspector_desc")}
                </p>
              </div>
            </Link>
            <Link href="/discovery">
              <div className="group bg-surface border-border hover:border-accent-cyan relative cursor-pointer overflow-hidden rounded-sm border p-6 transition-all">
                <div className="group-hover:text-accent-cyan absolute top-0 right-0 p-2 opacity-10 transition-all group-hover:opacity-100">
                  <FlaskConical size={32} />
                </div>
                <h4 className="text-foreground group-hover:text-accent-cyan text-xl font-bold tracking-widest uppercase italic transition-colors">
                  {t("discovery_title")}
                </h4>
                <p className="mt-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                  {t("discovery_desc")}
                </p>
              </div>
            </Link>
            <Link href="/stats">
              <div className="group bg-surface border-border hover:border-accent-cyan relative cursor-pointer overflow-hidden rounded-sm border p-6 transition-all">
                <div className="group-hover:text-accent-cyan absolute top-0 right-0 p-2 opacity-10 transition-all group-hover:opacity-100">
                  <BarChart3 size={32} />
                </div>
                <h4 className="text-foreground group-hover:text-accent-cyan text-xl font-bold tracking-widest uppercase italic transition-colors">
                  {t("stats_title")}
                </h4>
                <p className="mt-2 font-mono text-[10px] font-bold text-slate-500 uppercase">
                  {t("stats_desc")}
                </p>
              </div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
