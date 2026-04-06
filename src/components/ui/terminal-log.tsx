"use client";

import React, { useEffect, useRef } from "react";
import { useTranslations } from "next-intl";

interface TerminalLogProps {
  logs?: string[];
  className?: string;
  autoScroll?: boolean;
}

export const TerminalLog: React.FC<TerminalLogProps> = ({
  logs,
  className = "",
  autoScroll = true,
}) => {
  const terminalRef = useRef<HTMLDivElement>(null);
  const t = useTranslations("Common.terminal");

  const defaultLogs = [
    t("init"),
    t("loading_adapter"),
    t("auth_handshake"),
    t("network_ready"),
    t("ml_init"),
  ];

  const displayLogs = logs || defaultLogs;

  useEffect(() => {
    if (autoScroll && terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [displayLogs, autoScroll]);

  return (
    <div
      ref={terminalRef}
      className={`border-border scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent h-48 overflow-y-auto rounded-sm border bg-[#050608] p-4 font-mono text-[12px] ${className}`}
    >
      <div className="flex flex-col gap-1">
        {displayLogs.map((log, index) => (
          <div key={index} className="flex gap-3">
            <span className="text-slate-600 select-none">
              [{index.toString().padStart(2, "0")}]
            </span>
            <span className="leading-tight text-[#00e676] drop-shadow-[0_0_2px_rgba(0,230,118,0.4)]">
              {log}
              {index === displayLogs.length - 1 && (
                <span className="ml-2 inline-block h-4 w-2 animate-pulse bg-[#00e676] align-middle shadow-[0_0_5px_#00e676]" />
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
