"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Search,
  FlaskConical,
  Terminal,
  Moon,
  Sun,
} from "lucide-react";
import { useThemeStore } from "@/store/theme-store";
import { useEffect, useState } from "react";

const navItems = [
  {
    name: "DASHBOARD",
    href: "/",
    icon: LayoutDashboard,
  },
  {
    name: "DISCOVERY LAB",
    href: "/discovery",
    icon: FlaskConical,
  },
  {
    name: "INSPECTOR",
    href: "/inspector",
    icon: Search,
  },
];

export const Sidebar = () => {
  const pathname = usePathname();
  const { theme, toggleTheme } = useThemeStore();
  const [mounted, setMounted] = useState(false);

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
            <h1 className="text-foreground text-sm leading-none font-black tracking-[0.2em] uppercase italic">
              SYBIL <span className="text-accent-cyan">ENGINE</span>
            </h1>
          </div>

          {/* <div className="bg-surface-secondary/50 flex items-center gap-2 rounded-sm border border-dashed border-slate-700/30 px-3 py-1.5">
            <div className="bg-accent-green h-1 w-1 animate-pulse rounded-full shadow-[0_0_3px_var(--accent-green)]" />
            <span className="font-mono text-[7px] tracking-[0.2em] text-slate-500 uppercase">
              Global Discovery Active
            </span>
          </div> */}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-8">
        <ul className="space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`flex items-center gap-4 rounded-sm px-5 py-3 font-mono text-[9px] font-bold tracking-[0.2em] transition-all ${
                    isActive
                      ? "bg-surface-secondary text-accent-cyan border-accent-cyan border-l-2 shadow-inner"
                      : "hover:text-foreground hover:bg-surface-secondary/50 text-slate-500"
                  } `}
                >
                  <item.icon size={14} />
                  {item.name}
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
              System Interface
            </span>
            <span className="text-accent-cyan font-mono text-[8px] font-bold uppercase italic">
              {theme === "light" ? "LIGHT MODE" : "DARK MODE"}
            </span>
          </div>

          {mounted && (
            <button
              onClick={toggleTheme}
              className="border-border bg-background hover:bg-surface-secondary text-foreground rounded-sm border p-2 shadow-sm transition-all"
              title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
            >
              {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          )}
        </div>

        {/* <div className="flex flex-col gap-2">
          <div className="flex justify-between">
            <span className="font-mono text-[7px] text-slate-500 uppercase">
              v2.4.1-STABLE
            </span>
            <span className="text-accent-cyan font-mono text-[7px]">68%</span>
          </div>
          <div className="bg-background border-border h-1 w-full overflow-hidden rounded-full border">
            <div className="bg-accent-cyan h-full w-2/3 animate-pulse opacity-50" />
          </div>
        </div> */}
      </div>
    </aside>
  );
};
