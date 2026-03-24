---
description: "Refactor page logic in discovery and inspector pages to construct nested payloads and handle the new response schema, including TerminalLog and IndustrialCard updates."
agent: "edit"
tools: ["read_file", "write_file"]
---

# Implement Phase 4: Update Page Logic & Data Binding

You are an expert Frontend Developer specializing in React, Next.js App Router, Zustand/React State, and TypeScript.

## Task Section
We have updated our core TypeScript types (Phase 1), API hooks (Phase 2), and Graph visualizations (Phase 3). 
Your final task is to execute **Phase 4**: Update the Next.js pages (`discovery/page.tsx` and `inspector/page.tsx`) so they construct the correct nested API payloads and correctly pass the new response data down to the UI components (`TerminalLog` and `IndustrialCard`).

## Instructions Section

**Step 1: Fix `src/app/discovery/page.tsx` (Payload Construction)**
- Locate the form submission handler (e.g., `onSubmit` or `handleStart`).
- Currently, it might be passing a flat object. You MUST refactor the payload to match the `DiscoveryStartRequest` nested schema:
  ```javascript
  const payload = {
    time_range: { 
      start_date: formData.startDate, 
      end_date: formData.endDate 
    },
    max_nodes: 2000,
    hyperparameters: { 
      max_epochs: 400, 
      patience: 30 
    }
  };
  ```
- Pass this `payload` to the mutation trigger.

**Step 2: Fix `src/app/inspector/page.tsx` (State & Prop Drilling)**
- Locate where the inference API response is handled and stored in state.
- Ensure the state variable holding the graph data is set using `response.ego_graph` (which contains `nodes` and `edges`).
- Pass the `response.report` object to the components that need it (like the `TerminalLog` and `IndustrialCard`).

**Step 3: Fix `src/components/ui/terminal-log.tsx`**
- Update the component props to accept `report: SybilReport` (or specific fields like `reasoning` and `label`) instead of the old `analysis` object.
- Map the output to read from `report.reasoning`.
- Ensure color mapping aligns with the new 4-tier labels: `"MALICIOUS"` (Red), `"HIGH_RISK"` (Orange), `"LOW_RISK"` (Yellow), `"BENIGN"` (Green).

**Step 4: Fix `src/components/ui/industrial-card.tsx` (Risk Score Formatting)**
- Locate where the Risk Score is displayed.
- The new backend returns `risk_score` as a float between `0.0` and `1.0`. 
- Update the display logic to multiply by 100 and format as a percentage (e.g., `(score * 100).toFixed(0) + "%"`).

## Context/Input Section
- Target files:
  - `src/app/discovery/page.tsx`
  - `src/app/inspector/page.tsx`
  - `src/components/ui/terminal-log.tsx`
  - `src/components/ui/industrial-card.tsx`
- **Schema Reminder**:
  - `InspectorResponse` has `report` (contains `label`, `risk_score`, `reasoning`) and `ego_graph` (contains `nodes` and `edges`).

## Output Section
- Modify the targeted files directly to fix payload construction and prop drilling.
- Ensure all TypeScript errors related to these prop changes are resolved.

## Quality/Validation Section
- The Discovery payload MUST have the `time_range` object.
- The Inspector page MUST NOT reference `response.analysis` or `response.local_graph`. It must strictly use `report` and `ego_graph`.
- The Risk Score MUST be displayed as a percentage on the UI.
