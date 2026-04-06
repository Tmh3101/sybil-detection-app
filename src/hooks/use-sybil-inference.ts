import { useQuery } from "@tanstack/react-query";
import apiClient from "@/lib/api";
import { InspectorResponse, InspectorHistoryResponse } from "@/types/api";

export const useInspectProfile = (profileId: string | null) => {
  return useQuery<InspectorResponse | null>({
    queryKey: ["inspector", profileId],
    queryFn: async () => {
      if (!profileId) return null;
      return apiClient.get(`/api/v1/inspector/profile/${profileId}`);
    },
    enabled: !!profileId,
  });
};

export const useInspectorHistory = () => {
  return useQuery<InspectorHistoryResponse, Error>({
    queryKey: ["inspector", "history"],
    queryFn: async () => {
      return apiClient.get(`/api/v1/inspector/history`);
    },
  });
};
