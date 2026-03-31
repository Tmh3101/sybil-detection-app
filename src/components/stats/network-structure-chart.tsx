"use client";

import React, { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { EdgeDistributionItem } from "@/types/api";

interface NetworkStructureChartProps {
  data: EdgeDistributionItem[];
}

const COLORS = ["#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899", "#f97316"];

export const NetworkStructureChart: React.FC<NetworkStructureChartProps> = ({
  data,
}) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Avoid synchronous state update in effect to prevent cascading renders
    const timer = setTimeout(() => {
      setMounted(true);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  return (
    <IndustrialCard title="NETWORK STRUCTURE (EDGES)" className="h-[400px]">
      <div className="h-full w-full flex-1">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%" minHeight={300}>
            <PieChart>
              <Pie
                data={data}
                cx="50%"
                cy="45%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={5}
                dataKey="count"
                nameKey="layer"
                label={({ payload }) => `${payload.percentage}%`}
                labelLine={false}
              >
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name, props) => {
                  const percentage =
                    (props?.payload as EdgeDistributionItem | undefined)
                      ?.percentage ?? 0;
                  const numValue = typeof value === "number" ? value : 0;
                  return [
                    `${numValue.toLocaleString()} (${percentage}%)`,
                    String(name ?? "unknown").toUpperCase(),
                  ];
                }}
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "4px",
                  color: "#f8fafc",
                  fontSize: "12px",
                  fontFamily: "monospace",
                }}
                itemStyle={{ color: "#06b6d4" }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                formatter={(value) => {
                  return (
                    <span className="font-mono text-[9px] font-bold tracking-[0.1em] text-slate-400 uppercase">
                      {value}
                    </span>
                  );
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </IndustrialCard>
  );
};
