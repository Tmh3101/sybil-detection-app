---
description: "Fix Avatar rendering, risk_label colors, and target node highlighting in UniversalGraph2D"
agent: "edit"
tools: ["editFiles", "codebase"]
---

# Fix UniversalGraph2D Rendering & Logic Bugs

You are an expert React, Canvas API, and Next.js Developer. Your task is to fix three critical rendering and logic bugs related to the `UniversalGraph2D` component and its data pipeline.

## Task Overview

1. **Avatar Loading:** Canvas is not re-rendering when images finish loading.
2. **Color Mapping:** All nodes appear as BENIGN because `risk_label` strings are not normalized, or data is lost in the processing hook.
3. **Target Node Highlight:** The target node in EGO mode is not glowing due to case-sensitive ID mismatches or missing props.

## Step-by-Step Instructions

### Step 1: Fix `src/components/graph/universal-graph-2d.tsx`

Open this file and make the following precise changes:

1. **State Trigger:** Add a new state at the top of the component to force re-renders: `const [avatarTrigger, setAvatarTrigger] = useState(0);`.
2. **Avatar OnLoad:** Inside the `getOrLoadImage` callback, locate `img.onload`. Add `setAvatarTrigger((prev) => prev + 1);` just before the `d3ReheatSimulation()` call.
3. **Case-Insensitive Target Color:** In the `targetNodeColor` useMemo hook, modify the `.find()` condition to be case-insensitive: `String(n.id).toLowerCase() === String(targetId).toLowerCase()`. Then, normalize the risk label: `const rl = String(found?.risk_label || "UNKNOWN").toUpperCase().trim();` before passing it to `LABEL_COLORS`.
4. **Normalize Node Drawing:** Inside the `drawNode` callback:
   - Normalize the label: `const rl = String(ext.risk_label || "UNKNOWN").toUpperCase().trim();`.
   - Update `isTarget` to be safely case-insensitive: `const isTarget = mode === "EGO" && !!targetId && String(node.id).toLowerCase() === String(targetId).toLowerCase();`.
5. **Normalize Tooltip:** Inside the `nodeLabel` callback, apply the same label normalization: `const rl = String(ext.risk_label || "UNKNOWN").toUpperCase().trim();`.

### Step 2: Fix `src/app/inspector/page.tsx`

Open the inspector page and check the EGO graph instantiation.

1. Locate the `<UniversalGraph2D>` component invocation.
2. Ensure that the `targetId` prop is explicitly and correctly passed using the profile ID from the API response (e.g., `targetId={analysisData.profile_info.id}`). Do not pass `undefined`.

### Step 3: Fix `src/hooks/use-graph-processor.ts`

Open the graph processor hook. Data loss might be occurring here.

1. Locate where the hook processes, maps, or clones the `nodes` array.
2. Ensure that when constructing the new node objects, you use the spread operator (`...n`) to preserve all original fields from the backend, specifically ensuring `risk_label` and `attributes` are NOT stripped out.

## Context & Constraints

- Do not remove the `d3ReheatSimulation()` calls. The `avatarTrigger` state is an _addition_ to force React to redraw the canvas.
- Web3 Wallet IDs must ALWAYS be compared using `.toLowerCase()` because checksum casing varies.
- Ensure TypeScript compiles successfully after your changes.

## Quality & Validation

- **Success Criteria 1:** Avatars pop into the canvas as soon as they finish loading over the network.
- **Success Criteria 2:** Nodes correctly display Red (MALICIOUS), Orange (HIGH_RISK), etc., mapping perfectly to `LABEL_COLORS`.
- **Success Criteria 3:** In Inspector mode, the central node has the dashed orbit and glowing aura.
