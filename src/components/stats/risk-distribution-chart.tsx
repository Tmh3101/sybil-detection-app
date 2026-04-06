"use client";

import React, { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { IndustrialCard } from "@/components/ui/industrial-card";
import { RiskDistributionItem } from "@/types/api";
import { useTranslations } from "next-intl";

interface RiskDistributionChartProps {
  data: RiskDistributionItem[];
}

const RISK_COLORS: Record<string, string> = {
  BENIGN: "#06b6d4", // cyan-500
  LOW_RISK: "#10b981", // emerald-500
  HIGH_RISK: "#f59e0b", // amber-500
  MALICIOUS: "#ef4444", // red-500
};

export const RiskDistributionChart: React.FC<RiskDistributionChartProps> = ({
  data,
}) => {
  const t = useTranslations("StatsPage");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Avoid synchronous state update in effect to prevent cascading renders
    const timer = setTimeout(() => {
      setMounted(true);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  return (
    <IndustrialCard title={t("risk_distribution_title")} className="h-[400px]">
      <div className="h-full w-full flex-1">
        {mounted && (
          <ResponsiveContainer width="100%" height="100%" minHeight={300}>
            <BarChart
              data={data}
              margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#334155"
                vertical={false}
                opacity={0.3}
              />
              <XAxis
                dataKey="label"
                stroke="#64748b"
                fontSize={10}
                fontFamily="monospace"
                tickFormatter={(value) => String(value || "").replace("_", " ")}
                angle={-25}
                textAnchor="end"
                interval={0}
              />
              <YAxis
                stroke="#64748b"
                fontSize={10}
                fontFamily="monospace"
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "rgba(255, 255, 255, 0.05)" }}
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "4px",
                  color: "#f8fafc",
                  fontSize: "12px",
                  fontFamily: "monospace",
                }}
                itemStyle={{ fontFamily: "monospace" }}
                formatter={(value, _name, props) => {
                  const label = (
                    props?.payload as RiskDistributionItem | undefined
                  )?.label;
                  const numericValue = typeof value === "number" ? value : 0;
                  const displayLabel = String(label || "UNKNOWN").replace(
                    "_",
                    " "
                  );

                  return [
                    <span
                      key={label ?? "unknown"}
                      style={{ color: label ? RISK_COLORS[label] : "#64748b" }}
                    >
                      {numericValue.toLocaleString()}
                    </span>,
                    displayLabel.toUpperCase(),
                  ];
                }}
              />
              <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={RISK_COLORS[entry.label] || "#64748b"}
                    stroke={RISK_COLORS[entry.label] || "#64748b"}
                    strokeWidth={1}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </IndustrialCard>
  );
};
