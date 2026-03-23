---
description: "Tích hợp dữ liệu thật từ Modal API vào Module 2 (Profile Inspector) sử dụng React Query."
agent: "software-engineer"
---

# Task 2: Profile Inspector (Module 2) Data Binding

Bạn là một Kỹ sư Frontend đang làm việc trên hệ thống "Sybil Engine". Ở Task trước, tầng API Foundation (Axios + React Query) đã được thiết lập. Nhiệm vụ của bạn bây giờ là kết nối API Real-time của Module 2 vào giao diện tĩnh (Mock UI).

## 🎯 Task Objective

Gọi endpoint `GET /api/v1/inspector/profile/{profile_id}`, lấy dữ liệu và hiển thị lên trang `/inspector`. Bắt buộc phải giữ nguyên phong cách thiết kế "Hardcore Industrial" (Neumorphism, Dark Mode). CHƯA tích hợp đồ thị 3D thật ở bước này (vẫn giữ SVG placeholder ở cột giữa).

## 🧩 TypeScript Interfaces (Tham chiếu)

Cập nhật file `src/types/api.d.ts` dựa trên tài liệu API chính thức:

```typescript
export interface ProfileInfo {
  id: string;
  handle: string;
  picture_url: string;
  owned_by: string;
}

export interface Analysis {
  sybil_probability: number;
  classification: "BENIGN" | "WARNING" | "SYBIL";
  reasoning: string[];
}

export interface LocalGraph {
  nodes: any[]; // Chi tiết sẽ làm ở Task 3
  links: any[];
}

export interface InspectorResponse {
  profile_info: ProfileInfo;
  analysis: Analysis;
  local_graph: LocalGraph;
}
```

## 📋 Step-by-Step Instructions

### Bước 1: Tạo Custom Hook (`src/hooks/use-sybil-inference.ts`)

- Import `useQuery` từ `@tanstack/react-query` và `apiClient` từ `src/lib/api`.
- Tạo hook `useInspectProfile(profileId: string | null)`.
- **Query Key:** `['inspector', profileId]`.
- **Query Function:** Gọi `apiClient.get<InspectorResponse>(/api/v1/inspector/profile/${profileId})`.
- **Options:** Đặt `enabled: !!profileId` để hook không tự chạy khi chưa có ID ví.

### Bước 2: Kích hoạt thanh Tìm kiếm (Search Bar)

- Mở component chứa thanh Search (thường là `src/components/layout/top-header.tsx` hoặc trong chính trang `/inspector`).
- Chuyển component thành `"use client"`.
- Dùng `useRouter` và `useSearchParams` từ `next/navigation`.
- Cài đặt form input: Khi người dùng gõ Wallet Address/Handle và bấm Enter -> Cập nhật URL thành `?wallet={giá_trị_nhập_vào}`.

### Bước 3: Đổ dữ liệu vào UI (`src/app/inspector/page.tsx`)

- Đọc `walletId` từ URL Search Params (e.g., `searchParams.get('wallet')`).
- Gọi hook `useInspectProfile(walletId)`.
- **Xử lý Trạng thái (States):**
  - **Trạng thái Trống (Idle):** Nếu không có `walletId`, hiển thị dòng chữ: "AWAITING TARGET INPUT... PLEASE ENTER WALLET ADDRESS." với font Monospace mờ.
  - **Trạng thái Loading (Cold Start):** Modal có thể mất thời gian để load model. Hãy hiển thị một skeleton loading hoặc thông báo nhấp nháy: `[SYS] WAKING UP AI CORE... WARMING UP TENSORS...`
  - **Trạng thái Lỗi (Error):** Hiển thị ô cảnh báo đỏ `[ERR] FAILED TO FETCH TARGET DATA`.
- **Data Binding (Khi có Data):**
  - **Cột Trái (Diagnostic):** Truyền `data.analysis.sybil_probability` (nhân 100) vào Gauge Chart. Chuyển mảng `data.analysis.reasoning` vào `<TerminalLog>` để in ra các dòng log gõ máy. Đổi màu tùy theo `classification`.
  - **Cột Phải (Metadata):** Thay thế dữ liệu giả trong các `<IndustrialCard>` bằng `data.profile_info.handle`, `data.profile_info.owned_by`, v.v.

## 🛑 Quality Constraints

1. **Zero Layout Shift:** Khi dữ liệu đang load hoặc bị lỗi, khung (Shell) của 3 cột không được phép bị vỡ hoặc co giãn.
2. **Animation:** Đảm bảo Gauge Chart có hiệu ứng chạy kim từ 0 đến % thực tế một cách mượt mà (dùng CSS transition).
3. **KHÔNG** đụng đến thư viện vẽ đồ thị (Force Graph) trong Task này. Chỉ truyền props tĩnh hoặc để nguyên khối `<svg>` giả lập ở cột giữa.

## 📤 Output Format

Cung cấp code hoàn chỉnh (hoặc dùng MCP để ghi trực tiếp) cho:

1. `src/types/api.d.ts` (cập nhật)
2. `src/hooks/use-sybil-inference.ts` (mới)
3. `src/app/inspector/page.tsx` (cập nhật data binding)
4. Các file component con (Gauge, Metadata Card) nếu cần điều chỉnh để nhận `props`.
