---
description: "Phase 2: Visualize GAT attention weights on UniversalGraph2D for Explainable AI"
agent: "edit"
tools: ["editFiles"]
---

# Phase 2: Frontend Visualization of GAT Attention (XAI)

You are an expert React, Canvas, and Data Visualization Developer. The backend team has successfully exposed a new field `gat_attention` (float) on the graph edges. Your task is to visualize this data on the `UniversalGraph2D` component to create a glowing, "Explainable AI" effect for high-risk connections.

## Step-by-Step Instructions

### Step 1: Update TypeScript Definitions (`src/types/api.d.ts`)
1. Open `src/types/api.d.ts`.
2. Locate the `SybilEdge` interface.
3. Add the new optional property: `gat_attention?: number;`.

### Step 2: Preserve Attention in Graph Processor (`src/hooks/use-graph-processor.ts`)
When multiple edges between the same two nodes are aggregated, we must preserve the highest attention score.
1. Open `src/hooks/use-graph-processor.ts`.
2. Locate the `AggregatedLink` interface and add `gat_attention?: number;`.
3. Locate the aggregation logic (where links are grouped and mapped to `AggregatedLink`).
4. Add a calculation for `gat_attention` by taking the maximum value among the grouped links.
   *Example snippet inside the map function:*
   ```typescript
   gat_attention: Math.max(...group.map(l => l.gat_attention || 0)),
   ```

### Step 3: Enhance Graph Visuals (`src/components/graph/universal-graph-2d.tsx`)
We will use ForceGraph's native props to visualize the attention.
1. Open `src/components/graph/universal-graph-2d.tsx`.
2. Locate the `<ForceGraph2D>` component invocation.
3. Add or update the `linkWidth` prop to increase thickness based on attention:
   ```tsx
   linkWidth={(link) => {
     const l = link as AggregatedLink;
     const baseWidth = l.weight ? Math.min(l.weight, 5) : 1.8; // Use existing constants if available
     const attentionBoost = (l.gat_attention || 0) * 8; // Boost thickness
     return baseWidth + attentionBoost;
   }}
   ```
4. Add particle effects for edges the AI heavily focused on (attention > 0.1):
   ```tsx
   linkDirectionalParticles={(link) => {
     const l = link as AggregatedLink;
     // Only show particles if attention is significant
     return (l.gat_attention || 0) > 0.1 ? 3 : 0;
   }}
   linkDirectionalParticleWidth={(link) => {
     const l = link as AggregatedLink;
     return 2 + (l.gat_attention || 0) * 4;
   }}
   linkDirectionalParticleSpeed={(link) => {
     const l = link as AggregatedLink;
     return 0.01 + (l.gat_attention || 0) * 0.05;
   }}
   linkDirectionalParticleColor={() => "#ef4444"} // Glowing red for AI focus
   ```
5. Update `linkLabel` (Tooltip) to show the AI Attention percentage if it exists:
   ```tsx
   linkLabel={(link) => {
     const l = link as AggregatedLink;
     const types = l.types?.join(", ") || l.edge_type || "UNKNOWN";
     const att = l.gat_attention ? `<br/><span style="color: #ef4444; font-weight: bold;">AI Attention: ${(l.gat_attention * 100).toFixed(1)}%</span>` : "";
     return `<div style="background: rgba(0,0,0,0.8); border: 1px solid #334155; padding: 4px 8px; border-radius: 4px; font-size: 10px; font-family: monospace;">
       <span style="color: #94a3b8;">Type:</span> ${types}
       <br/><span style="color: #94a3b8;">Weight:</span> ${l.weight}
       ${att}
     </div>`;
   }}
   ```

## Quality Constraints
- Do not remove the existing logic inside `universal-graph-2d.tsx` (like `drawNode`, avatars, or zoom functions). Just inject the new link props into the `<ForceGraph2D>` component.
- Ensure TypeScript builds without errors.