---
description: "Overhaul the EgoGraph2D component to use react-force-graph-2d, displaying a strict multi-layer, multi-graph (Follow, Interact, Co-Owner, Similarity) with node avatars and directed/undirected edge support."
agent: "edit"
tools: ["read_file", "write_file"]
---

# Implement Phase 3: Implement Multi-layer EgoGraph2D with Node Avatars

You are an expert Frontend Developer specializing in data visualization using the `react-force-graph-2d` library, Canvas API, and Next.js.

## Task Section
Our frontend currently lacks a robust graph visualization. We need to implement a detailed, multi-graph visualization that accurately reflects our Sybil detection domain. 

Your task is to rewrite or significantly overhaul the `src/components/graph/ego-graph-2d.tsx` (or similar) component. It must use `react-force-graph-2d` and satisfy all architectural requirements for a complex multi-layer graph.

## Instructions Section

**Step 1: Define Multi-layer Graph Properties**
- Implement a `MULTIGRAPH_SCHEMA` object to define visual properties (colors, directedness, multi-link curvature) for each layer.
- Ensure only `"FOLLOW"` and `"INTERACT"` (or their subtypes like UPVOTE) are rendered as **directed** edges (arrows).
- Ensure `"CO-OWNER"` and `"SIMILARITY"` are rendered as **undirected** edges (straight lines, no arrows).

**Step 2: Implement Node Canvas Rendering with Avatars**
- Configure the `react-force-graph-2d` instance to use `nodeCanvasObject` and `nodeCanvasObjectMode="always"`.
- Inside the callback, implement logic to draw nodes as circles:
  1.  If `node.attributes.picture_url` exists, load and draw the user's avatar clipped within the circle.
  2.  If no avatar exists, draw a stylized silhouette or placeholder image.
- Render the `node.attributes.handle` (user handle) text below the node circle.

**Step 3: Implement Multi-graph Link Rendering**
- Utilize `react-force-graph-2d`'s multi-link capabilities. The backend might send multiple edges between the same node pair (e.g., node A follows node B AND node A and B have CO-OWNER ví).
- Use `linkCurvature` callback to dynamically apply curvature to edges so they don't overlap. (e.g., use `(link) => 0.1 * link.multiLinkIndex`).
- Set appropriate colors for each edge type based on your `MULTIGRAPH_SCHEMA` from Step 1. (e.g., Red for Co-Owner, Blue for Follow, Dashed for Similarity).

**Step 4: Connect Data and State**
- The graph data is already being fetched via hooks. Ensure the `InspectorResponse` data (`response.data.ego_graph.nodes` and `response.data.ego_graph.edges`) is passed as the `graphData` prop.
- Access node statistics using `node.attributes.total_reposts` (not `total_mirrors`) for things like hover tooltips or node sizing, as updated in api.d.ts.

## Context/Input Section
- Target file: `src/components/graph/ego-graph-2d.tsx`
- **Schema to Visualize:** The 12 relations are: `"FOLLOW"`, `"UPVOTE"`, `"REACTION"`, `"COMMENT"`, `"QUOTE"`, `"MIRROR"`, `"COLLECT"`, `"CO-OWNER"`, `"SAME_AVATAR"`, `"FUZZY_HANDLE"`, `"SIM_BIO"`, `"CLOSE_CREATION_TIME"`, `"UNKNOWN"`.
- Directed layers: `FOLLOW`, `INTERACT` (and subtypes like REACTION, UPVOTE).
- Undirected layers: `CO-OWNER`, `SIMILARITY` (and subtypes like SAME_AVATAR, SIM_BIO).
- **Hardcore Industrial Aesthetic:** The overall style should be dark and professional, perhaps with a slight green/cyan CRT glow, fitting the CRT terminal aesthetic.

## Output Section
- Fully functional `EgoGraph2D` component that renders a complex multi-graph in 2D space.
- The component must handle directedness, avatar loading, and multi-graph link separation automatically.

## Quality/Validation Section
- The graph MUST display multiple distinct edges (multi-links) between node pairs if they exist in the backend data.
- Edges for `"CO-OWNER"` MUST NOT have arrows.
- Nodes MUST display the profile picture if available.
- Verify that `total_reposts` is accessed correctly if used.