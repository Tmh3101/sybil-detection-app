"use client";

import { Terminal, Sun, Moon, Search } from "lucide-react";
import { useThemeStore } from "@/store/theme-store";
import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";

const SearchForm = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [searchValue, setSearchValue] = useState(searchParams.get("wallet") || "");

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchValue.trim()) {
      router.push(`/inspector?wallet=${searchValue.trim()}`);
    }
  };

  return (
    <form onSubmit={handleSearch} className="flex-1 max-w-md mx-8 relative group">
      <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-accent-cyan transition-colors" size={16} />
      <input
        type="text"
        placeholder="ENTER WALLET_ID OR HANDLE..."
        value={searchValue}
        onChange={(e) => setSearchValue(e.target.value)}
        className="w-full bg-background border border-border px-10 py-2 text-xs font-mono uppercase tracking-widest focus:outline-none focus:border-accent-cyan focus:ring-1 focus:ring-accent-cyan/20 transition-all placeholder:text-slate-600 rounded-sm"
      />
    </form>
  );
};

export const TopHeader = () => {
  const { theme, toggleTheme } = useThemeStore();
  const [mounted, setMounted] = useState(false);

  // Prevent hydration mismatch by only rendering theme toggle after mount
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  return (
    <header className="h-16 bg-surface border-b border-border flex items-center justify-between px-10 shadow-sm z-10 relative overflow-hidden transition-colors duration-300">
      {/* Subtle scanline effect overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.05)_50%),linear-gradient(90deg,rgba(255,0,0,0.01),rgba(0,255,0,0.005),rgba(0,0,255,0.01))] bg-[length:100%_2px,3px_100%] pointer-events-none opacity-20" />

      <div className="flex items-center gap-6 relative">
        <div className="p-2 border border-border bg-background shadow-inner rounded-sm">
          <Terminal
            className="text-accent-cyan drop-shadow-[0_0_5px_rgba(var(--accent-cyan),0.4)]"
            size={20}
          />
        </div>
        <div className="flex flex-col">
          <h1 className="text-xl font-bold tracking-[0.3em] uppercase text-foreground italic leading-tight">
            SYBIL <span className="text-accent-cyan">OVERWATCH</span>
          </h1>
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-accent-green shadow-[0_0_5px_var(--accent-green)] animate-pulse" />
            <span className="text-[8px] font-mono text-slate-500 uppercase tracking-widest">
              Global Discovery Active
            </span>
          </div>
        </div>
      </div>

      <Suspense fallback={<div className="flex-1 max-w-md mx-8 h-10 bg-surface-secondary/50 animate-pulse rounded-sm" />}>
        <SearchForm />
      </Suspense>

      <div className="flex items-center gap-8">
        {mounted && (
          <button
            onClick={toggleTheme}
            className="p-2 border border-border bg-background hover:bg-surface-secondary transition-all rounded-sm text-foreground"
            title={`Switch to ${theme === "light" ? "dark" : "light"} mode`}
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
          </button>
        )}
      </div>
    </header>
  );
};
