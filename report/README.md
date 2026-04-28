# **Project Report: Báo cáo đồ án 1 - Học có giám sát và ứng dụng**

Tài liệu này cung cấp thông tin chi tiết về phần report của đồ án 1 môn **Nhập môn học máy**, bao gồm cấu trúc thư mục, nội dung các chương và hướng dẫn biên dịch mã nguồn LaTeX thành file PDF.

## **Table of Contents**

- [**Project Report**](#project-report-báo-cáo-đồ-án-1---học-có-giám-sát-và-ứng-dụng)
  - [**Table of Contents**](#table-of-contents)
  - [**1. Directory Structure**](#1-directory-structure)
  - [**2. Report Contents (Nội dung báo cáo)**](#2-report-contents-nội-dung-báo-cáo)
  - [**3. How to Compile (Hướng dẫn biên dịch)**](#3-how-to-compile-hướng-dẫn-biên-dịch)

## **1. Directory Structure**

Thư mục `report/` được tổ chức thành các phần riêng biệt để quản lý mã nguồn LaTeX và file kết quả:

```text
report/
├── src/                 # Thư mục chứa toàn bộ mã nguồn LaTeX
│   ├── chapters/        # Các file .tex chứa nội dung chi tiết từng chương
│   ├── graphics/        # Hình ảnh, biểu đồ được chèn vào báo cáo
│   ├── packages/        # Các package tùy chỉnh thêm
│   ├── refs/            # Thư mục chứa file tài liệu tham khảo (.bib)
│   ├── styles/          # File template và style định dạng chuẩn (hcmus-report)
│   └── main.tex         # File gốc để biên dịch toàn bộ báo cáo
├── output/              # Thư mục chứa file PDF sau khi biên dịch
│   └── report.pdf       # Kết quả báo cáo cuối cùng
└── README.md            # Tài liệu bạn đang đọc
```
---

## **2. Report Contents**

Báo cáo trình bày chi tiết về quá trình thực hiện đồ án "Học có giám sát và ứng dụng" cho môn học **Nhập môn học máy**. Báo cáo được chia thành các chương chính (nằm trong thư mục `src/chapters/`):

* **Chương 1: Tổng quan (`01_tong_quan.tex`)**: Giới thiệu về bài toán, mục tiêu của đồ án và cái nhìn tổng quan về các phương pháp học có giám sát.
* **Chương 2: Hồi quy (`02_hoi_quy.tex`)**: Trình bày chi tiết quá trình giải quyết bài toán Hồi quy trên tập dữ liệu mức tiêu thụ năng lượng của thiết bị gia dụng (*Appliances Energy Prediction*). Bao gồm phân tích dữ liệu, thiết kế bộ đặc trưng, lựa chọn mô hình, huấn luyện và đánh giá.
* **Chương 3: Phân lớp (`03_phan_lop.tex`)**: Trình bày chi tiết quá trình giải quyết bài toán Phân lớp trên tập dữ liệu ước lượng số lượng người trong phòng (*Room Occupancy Estimation*). Bao gồm các bước phân tích, tiền xử lý, huấn luyện đa mô hình và đánh giá.
* **Chương 4: So sánh (`04_so_sanh.tex`)**: Đánh giá tổng quan, so sánh hiệu năng, ưu nhược điểm của các thuật toán Hồi quy và Phân lớp đã được tiến hành thực nghiệm.
* **Chương 5: Tổng kết (`05_tong_ket.tex`)**: Kết luận lại những kết quả đạt được của đồ án và đề xuất hướng phát triển tương lai.

## **3. How to Compile**

Mã nguồn của báo cáo sử dụng LaTeX với định dạng class `hcmus-report`. Để biên dịch ra file PDF, ta có thể sử dụng các công cụ phổ biến sau:

### **Cách 1: Sử dụng Overleaf**
1. Nén toàn bộ các file và thư mục bên trong `src/` thành một file `.zip`.
2. Tạo một project mới trên hệ thống [Overleaf](https://www.overleaf.com/) và chọn mục **Upload Project**.
3. Upload file `.zip` vừa tạo lên.
4. Mở file `main.tex` làm file chính và nhấn nút **Compile** (hoặc `Ctrl + S`).

### **Cách 2: Biên dịch cục bộ**
Yêu cầu máy tính đã được cài đặt môi trường LaTeX như TeX Live (trên Linux/Mac) hoặc MiKTeX (trên Windows).
1. Mở terminal hoặc command prompt và di chuyển vào thư mục `src/`:
   ```bash
   cd report/src
   ```
2. Chạy lần lượt các lệnh sau (hoặc dùng `latexmk`) để biên dịch `main.tex` và cập nhật mục lục, tài liệu tham khảo:
   ```bash
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex
   ```
3. File `main.pdf` sẽ được biên dịch thành công trong cùng thư mục `src/`. Ta có thể đổi tên và di chuyển file này sang thư mục `output/` dưới tên `report.pdf` để lưu trữ và nộp bài.
