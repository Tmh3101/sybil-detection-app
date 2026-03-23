import axios from "axios";
import { toast } from "sonner";

const apiClient = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

apiClient.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    const errorMessage =
      error.response?.data?.error || error.message || "UNKNOWN_SYSTEM_ERROR";

    toast.error(`[SYS_ERR] ${errorMessage.toUpperCase()}`, {
      description: `TIME: ${new Date().toISOString()}`,
    });

    console.error("API Error:", error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export default apiClient;
