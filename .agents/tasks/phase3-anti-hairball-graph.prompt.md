---
description: "Optimize Universal Graph 2D to fix the 'hairball' problem by adjusting opacity, edge thickness, and d3-force physics."
agent: "edit"
tools: ["file_search", "read_file", "edit_file"]
---

# Graph Visualization Optimization (Anti-Hairball)

You are an Expert Frontend Engineer specializing in React and Data Visualization (specifically `react-force-graph-2d`). 
Currently, the network graphs are suffering from the "hairball problem" — dense, fully opaque, thick edges are completely obscuring the nodes and making clusters impossible to read. 

Your task is to apply specific visual and physical tweaks to "thin out" the graph and improve readability.

## 🎯 Task Section
Locate the Universal Graph components (both 2D) and implement 3 core changes: drastically reduce edge opacity, decrease edge thickness scaling, and strengthen the d3-force node repulsion.

## 📋 Instructions Section

### Step 1: Adjust Edge Opacity (Transparency)
**Files to check/modify:** `src/components/graph/universal-graph-2d.tsx`.
1. Locate the `linkColor` prop in the Force Graph component.
2. Convert all solid hex or rgb colors for edges into highly transparent `rgba` values (alpha between `0.1` and `0.4`).
   - *Logic:* High-risk/Heuristic edges (like `CO-OWNER` or `SIMILARITY`) can have an alpha of `0.3` or `0.4` so they stand out.
   - *Logic:* Generic social edges (like `FOLLOW`, `UPVOTE`) MUST have a very low alpha, e.g., `rgba(150, 150, 150, 0.1)`.

### Step 2: Reduce Edge Thickness Scaling
1. Locate the `linkWidth` prop.
2. The backend already provides Log-scale weights. Reduce the frontend multiplier so edges are drawn much thinner.
   - *Change:* If it currently says `Math.max(1, link.weight * 1.2)`, change it to something like `Math.max(0.2, link.weight * 0.3)`.

### Step 3: Tweak D3-Force Physics Engine (Repulsion & Distance)
1. Ensure there is a `useRef` pointing to the Force Graph component (e.g., `fgRef`).
2. Add a `useEffect` hook that triggers when `graphData` changes to inject custom D3 physics.
3. Increase node repulsion (`charge`) and stretch the links (`distance`).
   - *Snippet to integrate:*
     ```typescript
     useEffect(() => {
       if (fgRef.current) {
         // Push nodes further apart (default is usually around -30)
         fgRef.current.d3Force('charge').strength(-150); 
         // Stretch the edges to prevent tight clustering
         fgRef.current.d3Force('link').distance(40); 
       }
     }, [graphData]);
     ```

### Step 4: Clutter Reduction (Optional but recommended)
1. If `linkDirectionalParticles` is currently enabled, either turn it off (`0`) or ensure it only activates when hovering over a specific node. Thousands of moving particles will crash the browser on dense graphs.

## Context/Input Section
- The project uses `react-force-graph-2d`.
- Make sure to handle both 2D components if they exist separately.
- Do not remove the reverse edge (`_REV`) filtering logic if it was implemented previously.

## ✅ Quality/Validation Section
1. Verify that TypeScript compilation passes (ensure the `useEffect` doesn't throw null reference errors on `fgRef`).
2. Verify that dense clusters are now semi-transparent, allowing nodes underneath to be visible.
3. Confirm that the physics engine actively pushes nodes apart when the graph loads.