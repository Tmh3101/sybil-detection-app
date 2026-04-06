import { useMutation, useQuery } from "@tanstack/react-query";
import apiClient from "@/lib/api";
import {
  DiscoveryStartRequest,
  DiscoveryStartResponse,
  DiscoveryStatusResponse,
  DiscoveryHistoryResponse,
} from "@/types/api";

/**
 * Hook to trigger a new Sybil Discovery Job (Module 1).
 */
export const useStartDiscovery = () => {
  return useMutation<DiscoveryStartResponse, Error, DiscoveryStartRequest>({
    mutationFn: async (data: DiscoveryStartRequest) => {
      return apiClient.post("/api/v1/sybil/discovery/start", data);
    },
  });
};

/**
 * Hook to poll the status of a Discovery Job.
 * Automatically polls every 3 seconds until status is COMPLETED or FAILED.
 */
export const useDiscoveryStatus = (taskId: string | null) => {
  return useQuery<DiscoveryStatusResponse, Error>({
    queryKey: ["discovery", taskId],
    queryFn: async () => {
      if (!taskId) throw new Error("Task ID is required");
      return apiClient.get(`/api/v1/sybil/discovery/status/${taskId}`);
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      // Stop polling if completed or failed
      if (status === "COMPLETED" || status === "FAILED") {
        return false;
      }
      return 3000; // Poll every 3 seconds
    },
    // Ensure we keep the previous data while fetching new status
    placeholderData: (previousData) => previousData,
  });
};

/**
 * Hook to fetch the history of Discovery jobs.
 */
export const useDiscoveryHistory = () => {
  return useQuery<DiscoveryHistoryResponse, Error>({
    queryKey: ["discovery", "history"],
    queryFn: async () => {
      return apiClient.get(`/api/v1/sybil/discovery/history`);
    },
  });
};

