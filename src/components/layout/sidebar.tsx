"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Search, FlaskConical } from "lucide-react";

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

  return (
    <aside className="bg-surface border-border flex w-64 flex-col border-r transition-colors duration-300">
      <div className="border-border border-b p-8">
        <div className="flex items-center gap-3">
          <div className="bg-accent-cyan h-4 w-4 shadow-[0_0_8px_var(--accent-cyan)]" />
          <span className="text-foreground text-xs font-black tracking-[0.4em] uppercase italic">
            SYBIL ENGINE
          </span>
        </div>
      </div>

      <nav className="flex-1 px-4 py-10">
        <ul className="space-y-4">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`flex items-center gap-4 rounded-sm px-6 py-3 font-mono text-[10px] font-bold tracking-[0.2em] transition-all ${
                    isActive
                      ? "bg-surface-secondary text-accent-cyan border-accent-cyan border-l-2 shadow-inner"
                      : "hover:text-foreground hover:bg-surface-secondary/50 text-slate-500"
                  } `}
                >
                  <item.icon size={16} />
                  {item.name}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="border-border flex flex-col gap-4 border-t p-8">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[8px] font-bold tracking-widest text-slate-500 uppercase">
            Engine Version
          </span>
          <span className="text-accent-cyan font-mono text-[8px] font-bold italic">
            v2.4.1-STABLE
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <div className="bg-surface-secondary h-1 w-full overflow-hidden rounded-full">
            <div className="bg-accent-cyan h-full w-2/3 animate-pulse" />
          </div>
          <span className="text-right font-mono text-[7px] text-slate-600 uppercase">
            System Load: 68%
          </span>
        </div>
      </div>
    </aside>
  );
};
