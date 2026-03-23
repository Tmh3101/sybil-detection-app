---
description: "Tích hợp react-force-graph-3d để vẽ đồ thị mạng lưới Ego-Graph tương tác cho Module 2."
agent: "software-engineer"
---

# Task 3: Interactive 3D Ego-Graph Integration

Bạn là Chuyên gia Frontend chuyên về Data Visualization (WebGL/Three.js) trong môi trường Next.js. Ở Task 2, dữ liệu thật đã được nối vào cột Trái và Phải. Ở Task 3 này, nhiệm vụ của bạn là thay thế cục SVG giả lập ở cột Giữa bằng một đồ thị 3D thực thụ, phô diễn sức mạnh của "Sybil Engine".

## 🎯 Task Objective

Xây dựng component `EgoGraph3D` sử dụng `react-force-graph-3d`, nhận dữ liệu `local_graph` từ API và render dưới dạng đồ thị 3D tương tác. Đồ thị phải tuân thủ nghiêm ngặt phong cách "Hardcore Industrial" (radar tối màu, tia laser sáng, hạt dữ liệu chạy dọc theo cạnh nối).

## 📋 Step-by-Step Instructions

### Bước 1: Cài đặt thư viện

- Chạy lệnh (hoặc yêu cầu người dùng chạy): `npm install react-force-graph-3d three`

### Bước 2: Xây dựng Component Đồ thị (`src/components/graph/ego-graph-3d.tsx`)

- Tạo file mới với directive `"use client"`.
- Import `ForceGraph3D` từ `react-force-graph-3d`.
- **Props yêu cầu:** - `graphData` (chứa `nodes` và `links` từ `LocalGraph`).
  - `targetId` (chuỗi ID của ví đang được soi để làm nổi bật node trung tâm).
  - `classification` (từ `Analysis` để quyết định màu sắc cảnh báo của node trung tâm).
- **Cấu hình ForceGraph3D (Industrial Styling):**
  - `backgroundColor="#00000000"` (Trong suốt để lộ nền lưới Grid của cột giữa).
  - `nodeColor`: Hàm trả về màu sắc. Nếu `node.id === targetId`: Trả về Đỏ Neon (`#ff1744`) nếu là Sybil/Warning, Cyan (`#00f2ff`) nếu Safe. Các node lân cận trả về Xám kim loại (`#64748b`).
  - `nodeVal`: Kích thước node. Node target phải to gấp 3 lần các node thường.
  - `linkColor`: Hàm trả về màu xám mờ mờ (`#334155`), hoặc có chút glow nếu trọng số (weight) cao.
  - `linkDirectionalParticles`: Bật hạt dữ liệu chạy trên cạnh (đặt value khoảng `2` đến `4`) để tạo cảm giác luồng dữ liệu đang truyền tải.
  - `linkDirectionalParticleSpeed`: Tốc độ hạt chạy (khoảng `0.005`).
- **Tương tác (Interaction):**
  - Khi hover/click vào node, hiển thị thuộc tính `handle` và `trust_score`.

### Bước 3: Import Động (Dynamic Import) để chống lỗi SSR (`src/app/inspector/page.tsx`)

- **LƯU Ý TỐI QUAN TRỌNG:** KHÔNG import `EgoGraph3D` theo cách thông thường. Bạn BẮT BUỘC phải dùng `next/dynamic` của Next.js với option `{ ssr: false }`.
- Ví dụ:
  ```tsx
  import dynamic from "next/dynamic";
  const EgoGraph3D = dynamic(() => import("@/components/graph/ego-graph-3d"), {
    ssr: false,
    loading: () => (
      <div className="text-cyan-500 animate-pulse font-mono">
        INITIALIZING 3D RENDER ENGINE...
      </div>
    ),
  });
  ```
- Thay thế khối `<svg>` giả lập ở cột giữa bằng component `<EgoGraph3D />` vừa import động.
- Truyền đủ props: `graphData={data.local_graph}`, `targetId={walletId}`, và `classification={data.analysis.classification}`.

## 🛑 Quality Constraints

1. **Responsive & Resize:** Đồ thị 3D thường bị lỗi không tự co giãn theo container. Hãy lắng nghe event resize của trình duyệt hoặc sử dụng thẻ bọc có kích thước cố định (ví dụ: `width={800} height={600}` thông qua custom hook lấy size của parent div) để đảm bảo UI không bị vỡ.
2. **Hiệu năng:** Không lạm dụng quá nhiều hạt (particles) nếu đồ thị có trên 500 nodes.
3. **Môi trường:** Đảm bảo code không gọi các object `window` hoặc `document` ở mức global (bên ngoài scope của useEffect hoặc ngoài các hàm event handler) để tránh sập Next.js SSR.

## 📤 Output Format

Trả về code hoặc dùng MCP để ghi trực tiếp các file:

1. `src/components/graph/ego-graph-3d.tsx` (Component đồ thị chính)
2. `src/app/inspector/page.tsx` (Bản cập nhật chứa dynamic import và gỡ bỏ SVG mock)
