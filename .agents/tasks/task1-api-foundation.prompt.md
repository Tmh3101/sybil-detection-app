---
description: "Thiết lập tầng giao tiếp API (Axios) và State Management (React Query) cho Next.js App Router."
agent: "software-engineer"
---

# Task 1: API Foundation & State Providers

Bạn là một Chuyên gia Kỹ sư Frontend (Frontend Engineer) với chuyên môn sâu về Next.js (App Router), Axios và TanStack React Query. Nhiệm vụ của bạn là xây dựng nền móng giao tiếp dữ liệu giữa Frontend và Modal FastAPI Backend cho hệ thống "Sybil Engine".

## 🎯 Task Objective

Thiết lập một đường ống gọi API (API Pipeline) an toàn, dễ bảo trì và tích hợp State Management toàn cục bằng React Query. TUYỆT ĐỐI tuân thủ nguyên tắc YAGNI (You Aren't Gonna Need It) và DRY (Don't Repeat Yourself). Chỉ code những gì được yêu cầu, không thêm tính năng thừa.

## 📋 Step-by-Step Instructions

Thực hiện tạo và chỉnh sửa các file sau (nếu dùng MCP thì ghi trực tiếp, nếu không thì in ra code block hoàn chỉnh):

### Bước 1: Cấu hình Biến Môi trường

- Tạo/Cập nhật file `.env.local` ở thư mục gốc.
- Thêm biến: `NEXT_PUBLIC_API_URL=https://tmh3101--sybil-discovery-engine-fastapi-endpoint.modal.run`

### Bước 2: Khởi tạo Axios Instance (`src/lib/api.ts`)

- Import thư viện `axios`.
- Tạo và export default một `apiClient` (Axios Instance).
- **Cấu hình:** - `baseURL`: Trỏ tới `process.env.NEXT_PUBLIC_API_URL`.
  - `timeout`: Đặt là `30000` (30 giây) vì hệ thống Modal Backend có thể bị Cold Start mất 15-20s.
  - `headers`: Chấp nhận `application/json`.
- **Interceptors:** Thêm một response interceptor cơ bản. Nếu response thành công, trả về `response.data`. Nếu lỗi, log ra console bằng `console.error` và ném lỗi (reject) để UI xử lý sau.

### Bước 3: Thiết lập React Query Provider (`src/providers/query-provider.tsx`)

- Import `QueryClient` và `QueryClientProvider` từ `@tanstack/react-query`.
- Import `ReactQueryDevtools` (chỉ render khi ở môi trường phát triển).
- Khởi tạo một `queryClient` với cấu hình mặc định:
  - `staleTime`: 1 phút (60 \* 1000).
  - `retry`: 1 lần (đề phòng lỗi mạng tạm thời).
  - `refetchOnWindowFocus`: false.
- Tạo component `QueryProvider` bọc `children` bằng `QueryClientProvider` và đính kèm `ReactQueryDevtools` (đặt `initialIsOpen={false}`).
- **Lưu ý:** Bắt buộc phải có directive `"use client"` ở đầu file.

### Bước 4: Tích hợp vào Layout tổng (`src/app/layout.tsx`)

- Import `QueryProvider` vừa tạo.
- Bọc toàn bộ nội dung bên trong thẻ `<body>` (bao gồm MainShell/TopHeader/Sidebar mà bạn đã có) bằng `<QueryProvider>`.
- Đảm bảo không làm hỏng cấu trúc Layout tĩnh hiện tại của dự án.

## 🛑 Quality Constraints

1. **Bắt buộc cài đặt thư viện:** Nhắc nhở người dùng (hoặc tự chạy lệnh nếu có quyền) chạy: `npm install axios @tanstack/react-query @tanstack/react-query-devtools` trước khi chạy code.
2. **Type Safety:** Cố gắng dùng TypeScript chặt chẽ, không dùng `any` bừa bãi trong interceptors.
3. **No UI Logic yet:** Tuyệt đối KHÔNG chạm vào các trang `/inspector` hay `/discovery` trong Task này. Chỉ làm nền móng.

## 📤 Output Format

Trả về code hoàn chỉnh cho 4 file được đề cập ở trên. Đảm bảo có chú thích ngắn gọn (comments) cho những thiết lập quan trọng như timeout 30s.
