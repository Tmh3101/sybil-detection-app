# Implementation Plan: Frontend-Backend API Integration (Sybil Engine)

## Overview

This plan outlines the steps to integrate the Next.js (App Router) frontend with the Modal FastAPI backend for the Sybil Discovery Engine. The engineer will establish a robust data pipeline using Axios and React Query, bind real data to the Inspector and Discovery modules, and replace static mockups with functional, data-driven components.

**Core Principles:** DRY, YAGNI, frequent commits after each passing test/task. Assume zero prior context of the specific UI components but follow the established "Hardcore Industrial" design tokens.

## File Structure Mapping

The following files will be created or modified during this implementation:

- `.env.local`: Add `NEXT_PUBLIC_API_URL`.
- `src/lib/api.ts` (New): Axios instance configuration with base URL and error interceptors.
- `src/providers/query-provider.tsx` (New): TanStack Query setup.
- `src/app/layout.tsx`: Wrap the application with `QueryProvider`.
- `src/hooks/use-sybil-inference.ts` (New): React Query hook for Module 2.
- `src/hooks/use-sybil-discovery.ts` (New): React Query hook for Module 1 (including polling logic).
- `src/app/inspector/page.tsx`: Bind data to Gauge Chart, Terminal Log, and Metadata cards.
- `src/app/discovery/page.tsx`: Implement form submission, polling terminal logs, and cluster rendering.
- `src/components/graph/ego-graph-3d.tsx` (New): Implement `react-force-graph-3d` to replace the SVG mock.

---

## Task Decomposition

### Task 1: API Foundation & State Providers

**Objective:** Establish the communication layer between the Frontend and Modal Backend.

1. Create `.env.local` and define `NEXT_PUBLIC_API_URL`.
2. Install dependencies: `npm install axios @tanstack/react-query @tanstack/react-query-devtools`.
3. Create `src/lib/api.ts` exporting a configured Axios instance.
4. Create `src/providers/query-provider.tsx` to initialize `QueryClient`.
5. Update `src/app/layout.tsx` to wrap the `children` with `QueryProvider`.
   _Testing:_ Verify the app compiles and the React Query Devtools appear (if enabled in dev mode).
   _Commit:_ `feat: setup axios and react-query providers`

### Task 2: Profile Inspector (Module 2) Data Binding

**Objective:** Fetch and display real inference data for a target wallet.

1. Create `src/hooks/use-sybil-inference.ts` utilizing `useQuery` to call `GET /api/v1/inspector/profile/{walletId}`.
2. Update `src/components/layout/top-header.tsx` to include a functional search input that updates a URL search parameter (e.g., `?wallet=0x...`).
3. Update `src/app/inspector/page.tsx` to read the `wallet` search param and call the hook.
4. Bind the `final_probability` to the Gauge Chart, map the `reasoning` array to the `<TerminalLog>`, and bind `trust_score`/`followers` to the metadata cards. Ensure loading states (Cold Start) and error states are handled gracefully.
   _Testing:_ Search for a valid mock wallet address and verify UI components update with fetched data.
   _Commit:_ `feat: integrate module 2 data binding`

### Task 3: Interactive 3D Graph Integration

**Objective:** Replace the static SVG mockup with a functional 3D force-directed graph.

1. Install graph dependencies: `npm install react-force-graph-3d three`.
2. Create `src/components/graph/ego-graph-3d.tsx`.
3. Write a parser inside the component to transform the API's `local_graph` object into the `{ nodes, links }` format required by the library.
4. Style the graph dynamically: Red color/glow for `is_sybil=true` nodes, cyan for safe nodes.
5. Dynamically import this component in `src/app/inspector/page.tsx` (with `ssr: false`) to pass the data.
   _Testing:_ Verify the 3D graph renders, nodes are colored correctly, and it handles orbit controls.
   _Commit:_ `feat: implement interactive 3D ego-graph`

### Task 4: Discovery Lab (Module 1) Batch Processing

**Objective:** Trigger GAE training and handle long-polling for task status.

1. Create `src/hooks/use-sybil-discovery.ts` with two functions: a mutation to start the job (`POST /api/v1/sybil/discovery/start`) and a query to poll status (`GET /api/v1/sybil/discovery/status/{taskId}`) using React Query's `refetchInterval`.
2. Update `src/app/discovery/page.tsx` to capture start/end dates and trigger the mutation.
3. Bind the polling logs to the `<TerminalLog>` component at the bottom of the page.
   _Testing:_ Trigger a discovery job and verify the terminal logs update dynamically every N seconds until completion.
   _Commit:_ `feat: integrate module 1 discovery polling`

### Task 5: UX Polish & Cold Start Handling

**Objective:** Ensure the application feels robust, especially during Modal's serverless cold starts.

1. Implement a specialized "Waking up AI Engine..." loading skeleton in `src/app/inspector/page.tsx` if the API request takes longer than 3 seconds.
2. Add toast notifications (e.g., using `sonner` or `react-hot-toast`) for API errors or invalid wallet formats.
   _Testing:_ Throttle network speed in DevTools to simulate cold start and verify loading states.
   _Commit:_ `feat: add cold start handling and toast notifications`
