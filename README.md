# SYBIL ENGINE | INDUSTRIAL DASHBOARD

<div align="center">
  <img src="src/app/icon.jpg" alt="Sybil Engine Logo" width="120" height="120" style="border-radius: 8px;" />
  <p><strong>Autonomous Sybil Detection & Network Discovery Platform</strong></p>
  <p>A high-performance system designed for large-scale sybil cluster identification and individual profile risk assessment with an industrial-grade interface.</p>
</div>

---

> [!IMPORTANT]
> **SYSTEM STATUS: OPERATIONAL**
> This platform integrates advanced Graph Autoencoders (GAE) and heuristic reasoning to map fraudulent networks and assess wallet integrity in real-time.

## 🛠️ CORE MODULES

### 🔬 [Discovery Lab (Module 1)](/discovery)
**Identify large-scale community clusters and fraudulent networks.**
- **Batch Processing:** Trigger GAE training for massive dataset analysis.
- **Task Polling:** Real-time job status tracking and diagnostic streaming.
- **Cluster Visualization:** Mapping interconnected sybil groups at scale using 2D force-directed graphs.
- **Dynamic Filtering:** Isolate clusters by risk level, size, or connectivity patterns.

### 🕵️ [Profile Inspector (Module 2)](/inspector)
**Analyze individual wallets for sybil behavior and risk scoring.**
- **Interactive 3D Ego-Graphs:** Visualize wallet networks with orbital controls using `react-force-graph-3d`.
- **Heuristic Reasoning:** Detailed breakdown of suspicious activities through live terminal logs.
- **Risk Assessment:** Dynamic "final_probability" scoring with cold-start handling for new profiles.
- **Data Aggregation:** Intelligent edge bundling and multi-link visualization for complex transactions.

## 🚀 KEY FEATURES

- **Industrial UI/UX:** High-contrast design, CRT scanline effects, and retro-future aesthetics powered by Tailwind CSS 4.
- **Real-time Diagnostics:** Live system monitoring, memory integrity checks, and database latency metrics.
- **Autonomous Boot Sequence:** Specialized loading skeletons and animations for seamless system initialization.
- **High-Performance Visualization:** Optimized 2D/3D graph rendering for dense network layouts using D3-force and Three.js.
- **Reliable Data Fetching:** Robust API pipeline with TanStack Query v5 and Axios, featuring built-in retry logic and cold-start handling.

## ⚡ TECH STACK

- **Frontend Framework:** [Next.js 15+](https://nextjs.org/) (App Router)
- **Language:** [TypeScript](https://www.typescriptlang.org/)
- **Styling:** [Tailwind CSS 4](https://tailwindcss.com/) with [Tailwind Merge](https://github.com/dcastil/tailwind-merge) & [Clsx](https://github.com/lukeed/clsx)
- **State Management:** [Zustand](https://zustand-demo.pmnd.rs/)
- **Data Fetching:** [TanStack React Query v5](https://tanstack.com/query/latest) & [Axios](https://axios-http.com/)
- **Visualization:** [react-force-graph](https://github.com/vasturiano/react-force-graph), [Three.js](https://threejs.org/), [D3.js](https://d3js.org/)
- **Notifications:** [Sonner](https://sonner.emilkowal.ski/) (Industrial Toast System)
- **Icons:** [Lucide React](https://lucide.dev/)

## ⚙️ GETTING STARTED

### Prerequisites

- **Node.js**: 20.x or higher
- **Package Manager**: npm or yarn

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/sybil-detection-app.git
   cd sybil-detection-app
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure environment:**
   Create a `.env.local` file in the root directory:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend-api-url.com
   ```

4. **Launch development server:**
   ```bash
   npm run dev
   ```

The dashboard will be available at `http://localhost:3000`.

## 📊 ARCHITECTURE OVERVIEW

Sybil Engine follows a distributed architecture with a Next.js frontend and a Modal-hosted FastAPI backend.

- **API Layer (`src/lib/api.ts`):** Centralized Axios client with response interceptors for global error handling.
- **Graph Processing (`src/hooks/use-graph-processor.ts`):** Client-side logic for edge aggregation, multi-link curvature calculation, and visual encoding.
- **Custom Hooks:** Specialized hooks for discovery polling (`useDiscoveryStatus`) and profile inspection (`useInspectProfile`).
- **Design System:** Custom industrial design tokens integrated via Tailwind CSS and `globals.css`.

> [!NOTE]
> The application uses "Hardcore Industrial" design tokens, prioritizing raw data visibility and system-level transparency.
