"use client";

import { Link, usePathname } from "@/i18n/routing";
import { useTranslations } from "next-intl";
import {
  LayoutDashboard,
  Search,
  FlaskConical,
  Terminal,
  Moon,
  Sun,
  BarChart3,
  History,
} from "lucide-react";
import { useThemeStore } from "@/store/theme-store";
import { useEffect, useState } from "react";
import { LanguageSwitcher } from "./language-switcher";

const navItems = [
  {
    key: "dashboard",
    href: "/",
    icon: LayoutDashboard,
  },
  {
    key: "inspector",
    href: "/inspector",
    icon: Search,
  },
  {
    key: "discovery",
    href: "/discovery",
    icon: FlaskConical,
  },
  {
    key: "stats",
    href: "/stats",
    icon: BarChart3,
  },
  {
    key: "history",
    href: "/history",
    icon: History,
  },
];


export const Sidebar = () => {
  const pathname = usePathname();
  const { theme, toggleTheme } = useThemeStore();
  const [mounted, setMounted] = useState(false);
  const t = useTranslations("Navigation");
  const tSidebar = useTranslations("Sidebar");

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  return (
    <aside className="bg-surface border-border flex w-64 flex-col border-r transition-colors duration-300">
      {/* Sidebar Header with Logo */}
      <div className="border-border border-b p-6">
        <div className="relative flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <div className="border-border bg-background rounded-sm border p-1.5 shadow-inner">
              <Terminal
                className="text-accent-cyan drop-shadow-[0_0_5px_rgba(var(--accent-cyan),0.4)]"
                size={18}
              />
            </div>
            <h1 className="text-foreground text-xl leading-none font-black tracking-[0.2em] uppercase italic">
              <span className="text-accent-cyan">SYBIL</span>SIGNAL
            </h1>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-8">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.key}>
                <Link
                  href={
                    item.href as "/" | "/inspector" | "/discovery" | "/stats" | "/history"
                  }
                  className={`flex items-center gap-4 rounded-sm px-5 py-3 font-mono text-[9px] font-bold tracking-[0.2em] transition-all ${

                    isActive
                      ? "bg-surface-secondary text-accent-cyan border-accent-cyan border-l-2 shadow-inner"
                      : "hover:text-foreground hover:bg-surface-secondary/50 text-slate-500"
                  } `}
                >
                  <item.icon size={14} />
                  {t(item.key)}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Sidebar Footer with Theme Toggle and Info */}
      <div className="border-border flex flex-col gap-6 border-t p-6">
        <div className="flex items-center justify-between">
          <div className="flex flex-col gap-1">
            <span className="font-mono text-[7px] font-bold tracking-widest text-slate-500 uppercase">
              {tSidebar("system_interface")}
            </span>
            <span className="text-accent-cyan font-mono text-[8px] font-bold uppercase italic">
              {mounted
                ? theme === "light"
                  ? tSidebar("light_mode")
                  : tSidebar("dark_mode")
                : "..."}
            </span>
          </div>

          {mounted && (
            <div className="flex gap-2">
              <LanguageSwitcher />
              <button
                onClick={toggleTheme}
                className="border-border bg-background hover:bg-surface-secondary text-foreground rounded-sm border p-2 shadow-sm transition-all"
                title={tSidebar("switch_theme", {
                  theme:
                    theme === "light"
                      ? tSidebar("dark_mode").toLowerCase()
                      : tSidebar("light_mode").toLowerCase(),
                })}
              >
                {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
              </button>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
