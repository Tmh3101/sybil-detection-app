---
description: "Phase 2: Frontend Sync for Global GAT Attention and Legend Update"
agent: "edit"
tools: ["editFiles"]
---

# Phase 2: Visualize Global GAT Attention & Update Legend

You are an expert Frontend Developer. The backend now returns `gat_attention` for ALL edges in the Depth-2 ego graph. We need to refine the visual thresholds on the Canvas so it doesn't become visually overwhelming, and update the Legend to explain these effects.

## Step-by-Step Instructions

### Step 1: Refine Canvas Visuals (`src/components/graph/universal-graph-2d.tsx`)
We need to apply a strict threshold so only the most important AI attention paths get the glowing particle effects.
1. Open `src/components/graph/universal-graph-2d.tsx`.
2. Locate the `<ForceGraph2D>` component and its link visual properties.
3. Update the properties to apply a threshold (e.g., `> 0.15`):
   ```tsx
   linkWidth={(link) => {
     const l = link as AggregatedLink;
     const baseWidth = l.weight ? Math.min(l.weight, 5) : 1.8;
     const att = l.gat_attention || 0;
     // Only boost thickness if attention is significant
     return baseWidth + (att > 0.1 ? att * 6 : 0);
   }}
   linkDirectionalParticles={(link) => {
     const l = link as AggregatedLink;
     // Stricter threshold for depth-2 graph to avoid particle spam
     return (l.gat_attention || 0) > 0.15 ? 3 : 0;
   }}
   linkDirectionalParticleWidth={(link) => {
     const l = link as AggregatedLink;
     return 1.5 + (l.gat_attention || 0) * 4;
   }}
   linkDirectionalParticleSpeed={(link) => {
     const l = link as AggregatedLink;
     return 0.01 + (l.gat_attention || 0) * 0.04;
   }}
   linkDirectionalParticleColor={() => "#ef4444"} // Red glow
   ```

### Step 2: Update the Graph Legend (`src/components/graph/graph-legend.tsx`)
Users need to know what the red glowing particles mean.
1. Open `src/components/graph/graph-legend.tsx`.
2. Scroll to the "Direction key" section at the bottom of the legend (where `directed` and `undirected` lines are drawn).
3. Add a new row below them to explain the "AI Attention Focus":
   ```tsx
            {/* AI Attention Focus Key */}
            <div className="flex items-center gap-2 mt-1">
              <div className="relative flex h-2 w-12 items-center justify-center overflow-hidden">
                <div className="absolute h-[2px] w-full bg-[#ef4444] opacity-40"></div>
                <div className="absolute h-[4px] w-[4px] animate-ping rounded-full bg-[#ef4444] shadow-[0_0_8px_#ef4444]"></div>
              </div>
              <span className="font-mono text-[7px] text-[#ef4444] font-bold">
                AI Focus (GAT)
              </span>
            </div>
   ```

## Quality Constraints
- Do not modify the existing node rendering, avatar logic, or zoom functions in `universal-graph-2d.tsx`.
- Ensure TypeScript compiles successfully.
- Make sure the Legend UI remains compact and aligned.
