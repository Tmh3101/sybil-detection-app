---

## description: "Refactor API hooks and client settings to match the updated FastAPI endpoints and Phase 1 type definitions."
agent: "edit"
tools: ["read_file", "write_file"]

# Implement Phase 2: Update API Client & Hooks

You are an expert Frontend Developer specializing in React Query (TanStack Query v5), Axios, and API integration in Next.js applications.

## Task Section

In Phase 1, we successfully updated `src/types/api.d.ts` to match the new v1.2 FastAPI backend schema. 
Your task now is to execute **Phase 2** of our integration plan: Update the API client hooks to use the correct endpoints and ensure they are passing data according to the newly defined types.

## Instructions Section

**Step 1: Fix `use-sybil-discovery.ts` (Module 1 Hooks)**

- Open `src/hooks/use-sybil-discovery.ts`.
- Check the `useStartDiscovery` mutation. Ensure the POST endpoint is strictly `/api/v1/sybil/discovery/start`.
- CRITICAL FIX: In the `useDiscoveryStatus` query, update the GET endpoint. It currently points to `/api/v1/sybil/discovery/status/${taskId}`. You must REMOVE `/status/` so it points exactly to: `/api/v1/sybil/discovery/${taskId}`.

**Step 2: Check & Update `use-sybil-inference.ts` (Module 2 Hooks)**

- Open `src/hooks/use-sybil-inference.ts` (if it exists).
- Ensure the POST endpoint for evaluating a profile is correctly set to `/api/v1/inspector/evaluate/${profileId}`.
- Verify that it uses the updated `InspectorResponse` type from `src/types/api.d.ts`. No major logic changes are needed here unless the endpoint URL is wrong, but ensure the type imports are correct.

**Step 3: Verify `api.ts` Configuration**

- Open `src/lib/api.ts`.
- Ensure the `timeout` is set to at least `60000` (60 seconds) because the fallback mechanism might take a second or two to process cold starts. (Do not change it if it is already 60000 or higher).

## Context/Input Section

- Target files: 
  - `src/hooks/use-sybil-discovery.ts`
  - `src/hooks/use-sybil-inference.ts`
  - `src/lib/api.ts`
- The types `DiscoveryStartRequest` and `InspectorResponse` have already been updated in `api.d.ts`.

## Output Section

- Directly apply code edits to the target files.
- Preserve all existing React Query configurations (like `refetchInterval`, `enabled` flags, etc.).

## Quality/Validation Section

- The `useDiscoveryStatus` polling function MUST NOT contain the word `/status/` in its endpoint path.
- TypeScript compiler should not throw errors regarding the payload if the mutation hook is correctly typed with `DiscoveryStartRequest`.

