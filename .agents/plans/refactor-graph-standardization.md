# 📋 Plan: Standardizing & Refactoring Graph System

## 🎯 Mục tiêu

- **DRY (Don't Repeat Yourself):** Loại bỏ việc lặp lại mã nguồn cho màu sắc và Legend UI.
- **Scalability:** Dễ dàng thêm các loại quan hệ hoặc nhãn mới chỉ bằng cách sửa một file duy nhất.
- **Performance separation:** Giữ nguyên sự khác biệt về cấu hình render giữa Ego Graph (chi tiết) và Cluster Map (số lượng lớn).

---

## 🛠️ Phase 1: Core Data & Shared UI Extraction

_Tách phần "xương sống" và "vỏ ngoài" dùng chung ra khỏi các component cụ thể._

- [ ] **Task 1: Tạo Shared Constants (`src/lib/graph-constants.ts`)**
  - [ ] Gom `LABEL_COLORS` và `RELATION_COLORS` từ `ego-graph-2d.tsx` vào file này.
  - [ ] Export các hằng số này để dùng cho toàn dự án.
  - [ ] Thêm một mảng `RELATION_GROUPS` để định nghĩa cách nhóm các loại edge (Co-owner, Follow, Interact, Similarity) giúp render Legend tự động.

- [ ] **Task 2: Tạo Shared Legend Component (`src/components/graph/graph-legend.tsx`)**
  - [ ] Tạo component `GraphLegend` nhận các props tùy chọn để ẩn/hiện các phần.
  - [ ] Sử dụng hằng số từ `src/lib/graph-constants.ts` để render danh sách màu sắc.
  - [ ] Giữ nguyên CSS Hardcore Industrial (border slate-800, font-mono, text-[9px]).

---

## ⚙️ Phase 2: Ego-Graph Refinement (Module 2)

_Tối ưu đồ thị chi tiết cho trang Inspector._

- [ ] **Task 3: Refactor `EgoGraph2D` (`src/components/graph/ego-graph-2d.tsx`)**
  - [ ] Import hằng số và `GraphLegend` mới.
  - [ ] **Implement Edge Aggregation (Góp ý của bạn):**
    - [ ] Sử dụng `useMemo` để gộp các cạnh có cùng `source`, `target`, và `edge_type`.
    - [ ] Tính toán `aggregated_weight` (tổng số cạnh bị gộp).
  - [ ] **Update Render Config:**
    - [ ] Cấu hình `linkWidth` dựa trên `Math.sqrt(aggregated_weight)`.
    - [ ] Cấu hình `linkDirectionalParticles` tỉ lệ thuận với trọng số.
  - [ ] Loại bỏ toàn bộ code JSX render Legend cũ ở cuối file.

---

## 🕸️ Phase 3: Cluster Map Refinement (Module 1)

_Tối ưu đồ thị quy mô lớn cho trang Discovery._

- [ ] **Task 4: Refactor `ClusterMap2D` (`src/components/graph/cluster-map-2d.tsx`)**
  - [ ] Import hằng số và `GraphLegend` mới.
  - [ ] **Tối ưu hiệu năng cho dữ liệu lớn:**
    - [ ] Đảm bảo `linkDirectionalParticles` bằng 0 để tránh giật lag.
    - [ ] Tắt render text/label node nếu số lượng nodes > 500.
    - [ ] Sử dụng các đường line mỏng cố định (`linkWidth={0.5}`) để tập trung vào cấu trúc cụm.
  - [ ] Loại bỏ code JSX Legend cũ.

---

## 🧪 Phase 4: Integration & Verification

_Kiểm tra cuối cùng để đảm bảo không phá vỡ ứng dụng._

- [ ] **Task 5: Update Page Bindings**
  - [ ] Kiểm tra `src/app/inspector/page.tsx` để đảm bảo props truyền vào `EgoGraph2D` vẫn đúng.
  - [ ] Kiểm tra `src/app/discovery/page.tsx` để đảm bảo `ClusterMap2D` hoạt động bình thường.

- [ ] **Task 6: Visual Audit**
  - [ ] Xác nhận bảng màu ở cả 2 trang là đồng nhất.
  - [ ] Xác nhận đồ thị 2D trang Inspector đã gộp cạnh thành công (đường nối dày hơn cho tương tác nhiều).
