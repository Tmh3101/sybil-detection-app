"use client";

import { useQueries } from "@tanstack/react-query";
import apiClient from "@/lib/api";
import { NetworkOverview, RiskDistribution, ClusterStats } from "@/types/api";

/**
 * Unified hook to fetch all network statistics from the backend.
 * Uses useQueries for parallel fetching and graceful degradation.
 */
export const useStats = () => {
  const results = useQueries({
    queries: [
      {
        queryKey: ["stats", "overview"],
        queryFn: async () => {
          try {
            return await apiClient.get<NetworkOverview>(
              "/api/v1/stats/overview"
            );
          } catch (error) {
            console.error("Failed to fetch stats overview:", error);
            return null;
          }
        },
      },
      {
        queryKey: ["stats", "risk"],
        queryFn: async () => {
          try {
            return await apiClient.get<RiskDistribution>(
              "/api/v1/stats/risk-distribution"
            );
          } catch (error) {
            console.error("Failed to fetch risk distribution:", error);
            return null;
          }
        },
      },
      {
        queryKey: ["stats", "clusters"],
        queryFn: async () => {
          try {
            return await apiClient.get<ClusterStats>("/api/v1/stats/clusters");
          } catch (error) {
            console.error("Failed to fetch cluster stats:", error);
            return null;
          }
        },
      },
    ],
  });

  const isLoading = results.some((r) => r.isLoading);
  // We consider it an error only if ALL queries failed or if critical ones failed.
  // For now, let's say it's an error if we have no data at all when not loading.
  const isError = !isLoading && results.every((r) => !r.data && r.isError);

  return {
    data: {
      overview: results[0].data as NetworkOverview | null,
      risk: results[1].data as RiskDistribution | null,
      clusters: results[2].data as ClusterStats | null,
    },
    isLoading,
    isError,
  };
};
