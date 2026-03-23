---
description: "Tích hợp API Module 1 (Discovery) với cơ chế Long-polling và trực quan hóa Cluster Map bằng React Query."
agent: "software-engineer"
---

# Task 4: Discovery Lab (Module 1) Batch Processing & Polling

Bạn là một Kỹ sư Frontend chuyên nghiệp. Chúng ta đã hoàn thành Module 2. Nhiệm vụ của bạn bây giờ là kích hoạt **Module 1 (Discovery Lab)** trên trang `/discovery`. Module này yêu cầu gửi lệnh kích hoạt một job huấn luyện AI nặng trên Modal GPU, sau đó liên tục lấy (poll) trạng thái tiến trình cho đến khi hoàn thành và vẽ bản đồ phân cụm (Cluster Map).

## 🎯 Task Objective
Xây dựng form nhập liệu thời gian, gọi API `POST` để start job, sử dụng React Query để poll API `GET` lấy trạng thái (status/progress), in log ra `<TerminalLog>` và cuối cùng vẽ các cụm Sybil lên bản đồ 2D. Vẫn giữ nguyên phong cách thiết kế "Hardcore Industrial".

## 🧩 TypeScript Interfaces (Tham chiếu)
Cập nhật file `src/types/api.d.ts` với các interface sau:

```typescript
export interface TimeRange {
  start_date: string; // YYYY-MM-DD
  end_date: string;   // YYYY-MM-DD
}

export interface DiscoveryStartRequest {
  time_range: TimeRange;
  max_nodes: number;
}

export interface DiscoveryStartResponse {
  task_id: string;
}

export interface DiscoveryStatusResponse {
  task_id: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  progress: number;
  current_step: string;
  graph_data: { nodes: any[]; links: any[] } | null;
  message: string | null;
}
```

## 📋 Step-by-Step Instructions

### Bước 1: Tạo Custom Hook (`src/hooks/use-sybil-discovery.ts`)
Tạo file hook chứa 2 logic chính dùng React Query và `apiClient`:
1. **`useStartDiscovery` (Mutation):** - Sử dụng `useMutation`.
   - Hàm `mutationFn` gọi `POST /api/v1/sybil/discovery/start` với body là `DiscoveryStartRequest`.
2. **`useDiscoveryStatus` (Query Polling):**
   - Sử dụng `useQuery`.
   - Hàm `queryFn` gọi `GET /api/v1/sybil/discovery/status/${taskId}`.
   - **Logic Polling:** Sử dụng `refetchInterval: (query) => { const status = query.state.data?.status; return (status === 'COMPLETED' || status === 'FAILED') ? false : 3000; }`. (Tự động poll mỗi 3 giây nếu chưa xong).

### Bước 2: Xây dựng Form Điều Khiển (`src/app/discovery/page.tsx`)
- Khai báo state cục bộ `taskId` để lưu ID của job đang chạy.
- Khai báo state cho `startDate` và `endDate` (mặc định lấy khoảng 7 ngày trước).
- Trong khối **Top Bar (Control Console)**: 
  - Đặt 2 input `type="date"` với class CSS mang phong cách Neumorphism lõm xuống.
  - Nút "START DISCOVERY" khi bấm sẽ gọi hàm `mutate` của `useStartDiscovery`. Khi thành công, lưu `task_id` trả về vào state `taskId` để kích hoạt hook polling.

### Bước 3: Đổ dữ liệu Log vào `<TerminalLog>`
- Lấy `data` từ `useDiscoveryStatus`.
- Trong khối **Bottom Panel (System Logs)**, truyền thông tin `status`, `progress` (%) và `current_step` vào Terminal.
- Ví dụ dòng log gõ ra: `[${data.status}] Tiến trình: ${data.progress}% - ${data.current_step}`. Nếu có `message` lỗi, in ra màu đỏ.

### Bước 4: Trực quan hóa Cụm (Cluster Map 2D)
- Cài đặt thư viện: `npm install react-force-graph-2d` (nếu chưa có).
- Tạo component `src/components/graph/cluster-map-2d.tsx` (phải `"use client"`).
- Component này nhận `graphData` từ kết quả API. Sử dụng `ForceGraph2D`.
- **Styling Industrial:**
  - Nền trong suốt.
  - `nodeColor`: Hàm trả về màu sắc dựa vào `node.cluster_id`. Hãy định nghĩa một mảng khoảng 5-6 màu Neon (Cyan, Purple, Orange, Red, Green) để gán cho các cluster khác nhau. Các node rủi ro cao (`HIGH_RISK`) thì thêm viền hoặc phát sáng.
  - Tắt tên node (label) để tránh giật lag vì số lượng node có thể lên tới hàng ngàn.
- Import động (Dynamic import với `ssr: false`) component `ClusterMap2D` vào trang `/discovery` tương tự như đã làm ở Task 3. 
- Khi `status === 'COMPLETED'`, ẩn hiệu ứng radar đi và hiển thị `ClusterMap2D` lên.

## 🛑 Quality Constraints
1. **UX/Nút bấm:** Trong lúc đang chạy (Mutation pending hoặc Status chưa COMPLETED), nút "START DISCOVERY" phải bị disable và có hiệu ứng "Processing...".
2. **Quản lý State:** Không được gọi API status nếu `taskId` bị null. 
3. **Hiệu năng Graph:** Vì dữ liệu trả về từ Module 1 có thể chứa hàng ngàn node, hãy dùng `ForceGraph2D` thay vì 3D để đảm bảo máy tính không bị treo.

## 📤 Output Format
Trả về code hoặc dùng MCP để ghi trực tiếp các file:
1. `src/types/api.d.ts` (Bổ sung type).
2. `src/hooks/use-sybil-discovery.ts` (Mới).
3. `src/components/graph/cluster-map-2d.tsx` (Mới).
4. `src/app/discovery/page.tsx` (Cập nhật UI, ghép Hook và Polling).