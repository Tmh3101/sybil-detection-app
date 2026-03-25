# 🚀 Plan: Comprehensive Graph System Standardization

## 🎯 Objectives

1.  **Data Logic:** Standardize data processing (Edge Aggregation for Ego-Graph, Performance Pruning for Cluster Map).
2.  **Shared UI:** Implement a unified color system and a shared Heads-Up Display (HUD) Legend.
3.  **Performance:** Decouple render configurations to ensure the Discovery page remains smooth even with large datasets.

---

## 🛠️ Phase 1: Foundation (Constants & UI)

_Goal: Establish global visual rules._

- [x] **Task 1: Centralized Constants (`src/lib/graph-constants.ts`)**
  - [x] Move `LABEL_COLORS` and `RELATION_COLORS` from individual graph files into this library.
  - [x] Define shared line-width constants (e.g., `MIN_LINK_WIDTH = 1`, `MAX_LINK_WIDTH = 5`).
- [x] **Task 2: Shared Legend Component (`src/components/graph/graph-legend.tsx`)**
  - [x] Build a standalone `GraphLegend` component that consumes constants from Task 1.
  - [x] Maintain the "Hardcore Industrial" aesthetic (slate-800 borders, monospace fonts, tiny 9px text).

---

## 🧠 Phase 2: Core Logic Hook (The "Brain")

_Goal: Separate data processing from the UI for reusability._

- [x] **Task 3: Create `useGraphProcessor` Hook (`src/hooks/use-graph-processor.ts`)**
  - [x] Implement **Edge Aggregation Logic**:
    - [x] Group edges by `source`, `target`, and `edge_type`.
    - [x] Sum up weights or count occurrences into an `aggregated_weight` property.
  - [x] Implement **Data Cleaning**: Handle cases where library mutation changes ID strings into objects (Source/Target ID extraction).
  - [x] Ensure the hook returns `processedData` memoized via `useMemo`.

---

## 🎨 Phase 3: Component Refactoring (The "Body")

_Goal: Apply logic and standardize specialized rendering._

- [x] **Task 4: Refactor `EgoGraph2D` (`src/components/graph/ego-graph-2d.tsx`)**
  - [x] Integrate `useGraphProcessor` to aggregate edges.
  - [x] **Detail-Oriented Configuration:**
    - [x] Set `linkWidth` proportional to `Math.sqrt(link.aggregated_weight)`.
    - [x] Enable `linkDirectionalParticles` scaled by weight for a "data flow" effect.
  - [x] Replace local Legend JSX with the new `<GraphLegend />`.
- [x] **Task 5: Refactor `ClusterMap2D` (`src/components/graph/cluster-map-2d.tsx`)**
  - [x] Use `useGraphProcessor` (apply light aggregation or raw data pruning).
  - [x] **Performance-Oriented Configuration:**
    - [x] Disable `linkDirectionalParticles` to prevent lag on large graphs.
    - [x] Conditional Rendering: Disable Node Text/Labels if node count > 500.
    - [x] Set a static, ultra-thin `linkWidth` (e.g., 0.5) to highlight cluster structures.
  - [x] Replace local Legend JSX with `<GraphLegend />`.

---

## 🧪 Phase 4: Integration & Verification

- [x] **Task 6: Sync Page Implementation**
  - [x] Update `/inspector` and `/discovery` pages to use the refactored components.
  - [x] Verify that the `1.0 - Benign` risk calculation for the Gauge Chart is consistent with the graph's visual state.
- [x] **Task 7: Visual Audit**
  - [x] Ensure colors are identical across both modules.
  - [x] Confirm that high-interaction edges in the Inspector are visually thicker/brighter than single-interaction edges.

---
