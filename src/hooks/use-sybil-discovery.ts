import { useMutation, useQuery } from "@tanstack/react-query";
import apiClient from "@/lib/api";
import {
  DiscoveryStartRequest,
  DiscoveryStartResponse,
  DiscoveryStatusResponse,
} from "@/types/api";

export const useStartDiscovery = () => {
  return useMutation<DiscoveryStartResponse, Error, DiscoveryStartRequest>({
    mutationFn: async (body) => {
      return apiClient.post("/api/v1/sybil/discovery/start", body);
    },
  });
};

export const useDiscoveryStatus = (taskId: string | null) => {
  return useQuery<DiscoveryStatusResponse, Error>({
    queryKey: ["discoveryStatus", taskId],
    queryFn: async () => {
      if (!taskId) throw new Error("No task ID provided");
      return apiClient.get(`/api/v1/sybil/discovery/status/${taskId}`);
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "COMPLETED" || status === "FAILED" ? false : 3000;
    },
  });
};
