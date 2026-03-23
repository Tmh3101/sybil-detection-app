"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Search,
  FlaskConical,
  Settings,
  Database,
} from "lucide-react";

const navItems = [
  {
    name: "DASHBOARD",
    href: "/",
    icon: LayoutDashboard,
  },
  {
    name: "DISCOVERY_LAB",
    href: "/discovery",
    icon: FlaskConical,
  },
  {
    name: "INSPECTOR",
    href: "/inspector",
    icon: Search,
  },
  {
    name: "DATA_STREAMS",
    href: "#",
    icon: Database,
  },
  {
    name: "SETTINGS",
    href: "#",
    icon: Settings,
  },
];

export const Sidebar = () => {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-surface border-r border-border flex flex-col transition-colors duration-300">
      <div className="p-8 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-4 h-4 bg-accent-cyan shadow-[0_0_8px_var(--accent-cyan)]" />
          <span className="font-black tracking-[0.4em] text-xs text-foreground uppercase italic">
            SYBIL_ENGINE
          </span>
        </div>
      </div>

      <nav className="flex-1 py-10 px-4">
        <ul className="space-y-4">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={`
                    flex items-center gap-4 px-6 py-3 rounded-sm text-[10px] font-mono font-bold tracking-[0.2em] transition-all
                    ${
                      isActive
                        ? "bg-surface-secondary text-accent-cyan border-l-2 border-accent-cyan shadow-inner"
                        : "text-slate-500 hover:text-foreground hover:bg-surface-secondary/50"
                    }
                  `}
                >
                  <item.icon size={16} />
                  {item.name}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="p-8 border-t border-border flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <span className="text-[8px] font-mono text-slate-500 uppercase tracking-widest font-bold">
            Engine Version
          </span>
          <span className="text-[8px] font-mono text-accent-cyan font-bold italic">
            v2.4.1-STABLE
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <div className="h-1 w-full bg-surface-secondary rounded-full overflow-hidden">
            <div className="h-full w-2/3 bg-accent-cyan animate-pulse" />
          </div>
          <span className="text-[7px] font-mono text-slate-600 uppercase text-right">
            System Load: 68%
          </span>
        </div>
      </div>
    </aside>
  );
};
