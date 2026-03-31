"use client";

import React from "react";
import { Loader2 } from "lucide-react";

export const BootSequenceLoader = () => {
  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center bg-[#050608]/80 backdrop-blur-sm">
      <div className="absolute inset-0 bg-[radial-gradient(var(--border)_1px,_transparent_0)] [mask-image:radial-gradient(ellipse_at_center,black,transparent)] bg-[length:24px_24px] opacity-10" />
      <div className="relative z-10 flex flex-col items-center gap-4">
        <Loader2 className="text-accent-cyan animate-spin" size={48} />
        <span className="text-accent-cyan/80 font-mono text-xs font-bold tracking-[0.2em] uppercase italic">
          Loading<span className="animate-ellipsis inline-block w-4"></span>
        </span>
      </div>

      <style jsx>{`
        @keyframes ellipsis {
          0% {
            content: ".";
          }
          33% {
            content: "..";
          }
          66% {
            content: "...";
          }
          100% {
            content: ".";
          }
        }
        .animate-ellipsis::after {
          content: ".";
          animation: ellipsis 1.5s infinite steps(1, end);
        }
      `}</style>
    </div>
  );
};
