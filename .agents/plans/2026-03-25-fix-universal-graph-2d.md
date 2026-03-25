# Implementation Plan: Fix UniversalGraph2D Rendering and Data Sync

## Overview

This plan outlines the steps to fix four critical rendering and data synchronization issues in the `UniversalGraph2D` component. The fixes ensure that the graph properly handles the new backend API schema (`risk_label`), keeps disconnected clusters grouped together, renders avatars securely, and maintains a consistent background aesthetic across different pages.

**Core Objectives:**

1. Fix component distance by adjusting D3 forces.
2. Sync node colors by migrating from `label` to `risk_label`.
3. Standardize the canvas background to be transparent.
4. Safely render node avatars using `picture_url` with proper fallbacks.

## AI Agent Execution Directives

**CRITICAL:** You are instructed to execute these tasks sequentially. **After completing each task, you MUST open this plan document and update the checklist below by changing `[ ]` to `[x]` before proceeding to the next task.**

## Scope & File Structure

- **Target File:** `src/components/graph/universal-graph-2d.tsx`

## Step-by-Step Tasks & Checklist

### [x] Task 1: Update Visual Constants & Data Access

- **Action:** Open `src/components/graph/universal-graph-2d.tsx`.
- **Constants Update:** Locate the `LABEL_COLORS` object. Update its keys to use the clean strings provided by the new backend schema: `"BENIGN"`, `"LOW_RISK"`, `"HIGH_RISK"`, and `"MALICIOUS"`. Remove the numeric prefixes (e.g., `"0_BENIGN"` -> `"BENIGN"`).
- **Commit:** "refactor(graph): update LABEL_COLORS to use prefix-less keys"

### [x] Task 2: Sync Node Colors and Tooltips with `risk_label`

- **Action:** Inside `universal-graph-2d.tsx`, locate the `drawNode` callback and the `nodeLabel` (tooltip) HTML generator.
- **Node Update:** Replace any property access of `node.label` with `node.risk_label` for determining colors. Ensure the stroke/border color uses `LABEL_COLORS[node.risk_label || "UNKNOWN"]`.
- **Tooltip Update:** Ensure the tooltip HTML template string displays `${node.risk_label}` instead of `${node.label}`. Remove any local string manipulation (like `.split("_")`) since the backend now provides clean strings.
- **Commit:** "fix(graph): sync node rendering and tooltips with risk_label schema"

### [x] Task 3: Fix Graph Physics (D3 Forces)

- **Action:** Locate the `useEffect` block that configures the D3 forces (using `fgRef.current.d3Force`).
- **Center Force:** Add or update the center force to pull disconnected clusters together: `fgRef.current.d3Force('center', d3.forceCenter(0, 0).strength(0.15));`.
- **Charge Force:** Adjust the charge (many-body) force to prevent nodes from flying too far apart. Ensure it is set to something manageable, e.g., `fgRef.current.d3Force('charge', d3.forceManyBody().strength(-100));`.
- **Commit:** "fix(graph): adjust D3 center and charge forces to group disconnected clusters"

### [x] Task 4: Fix Avatar Rendering & Background Consistency

- **Action:** Locate the avatar drawing logic inside the `drawNode` callback.
- **Avatar Fix:** Before setting the `Image.src`, pass the URL through the existing helper: `const safeUrl = resolvePictureUrl(node.attributes?.picture_url);`.
- **Fallback Logic:** If `safeUrl` is falsy or the image fails to load, do NOT attempt to draw the image. Instead, fill the circular path with a default dark color (e.g., `#1e293b`). Keep the circular clipping logic intact.
- **Background Fix:** In the `<ForceGraph2D>` component props, explicitly set `backgroundColor="rgba(0,0,0,0)"` so it becomes transparent. Ensure the wrapper `<div>` has a consistent tailwind background class (e.g., `bg-slate-950/40` or matches the layout).
- **Commit:** "fix(graph): resolve picture URLs safely and set transparent canvas background"

## Plan Review Loop

After writing the complete plan:

1. Dispatch a single plan-document-reviewer subagent with precisely crafted review context.
2. If ❌ Issues Found: fix the issues, re-dispatch reviewer for the whole plan.
3. If ✅ Approved: proceed to execution handoff.
