"use client";

import React from "react";

interface IndustrialCardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

export const IndustrialCard: React.FC<IndustrialCardProps> = ({
  children,
  title,
  className = "",
}) => {
  return (
    <div
      className={`bg-surface hover:border-accent-cyan/50 relative overflow-hidden rounded-sm border border-slate-200/60 p-6 transition-all duration-300 ease-out hover:-translate-y-0.5 dark:border-slate-800/60 ${className} `}
    >
      {/* 
         Mechanical screw points at corners 
         Simplified for a cleaner "Flat-Industrial" look 
      */}
      <div className="absolute top-2.5 left-2.5 h-1 w-1 rounded-full border border-slate-400/20 bg-slate-300 dark:border-slate-700/50 dark:bg-slate-800" />
      <div className="absolute top-2.5 right-2.5 h-1 w-1 rounded-full border border-slate-400/20 bg-slate-300 dark:border-slate-700/50 dark:bg-slate-800" />
      <div className="absolute bottom-2.5 left-2.5 h-1 w-1 rounded-full border border-slate-400/20 bg-slate-300 dark:border-slate-700/50 dark:bg-slate-800" />
      <div className="absolute right-2.5 bottom-2.5 h-1 w-1 rounded-full border border-slate-400/20 bg-slate-300 dark:border-slate-700/50 dark:bg-slate-800" />

      {/* Very subtle technical grid overlay */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(var(--border)_1px,_transparent_0)] bg-[length:20px_20px] opacity-[0.02] dark:opacity-[0.05]" />

      {title && (
        <div className="border-border/60 mb-4 border-b pb-2">
          <h3 className="text-accent-cyan font-mono text-[10px] font-bold tracking-[0.2em] uppercase italic">
            {title}
          </h3>
        </div>
      )}

      <div className="relative z-10">{children}</div>
    </div>
  );
};
