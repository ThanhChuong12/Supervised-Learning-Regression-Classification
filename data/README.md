# **Data Catalog & Provenance**

Tài liệu này cung cấp thông tin chi tiết về các bộ dữ liệu được sử dụng trong dự án, bao gồm nguồn gốc, cấu trúc từ điển dữ liệu (data dictionary), và các quy định về bản quyền. Dự án này phục vụ hai bài toán Machine Learning cốt lõi: **Hồi quy (Regression)** và **Phân lớp (Classification)**.

## **1. Directory Structure**

Thư mục `data/` được tổ chức theo tiêu chuẩn vòng đời dữ liệu, đảm bảo nguyên tắc không bao giờ ghi đè lên dữ liệu gốc (immutable raw data):

```text
data/
├── raw/                 # Dữ liệu gốc nguyên bản, KHÔNG ĐƯỢC PHÉP CHỈNH SỬA.
│   ├── regression_task.csv
│   └── Occupancy_Estimation.csv
├── interim/             # Dữ liệu trung gian đang trong quá trình tiền xử lý.
├── processed/           # Dữ liệu đã làm sạch, chuẩn hóa (Standardized/Design Matrix).
└── README.md            # Tài liệu bạn đang đọc.

```

---

## **2. Regression Task Dataset (Bài toán Hồi quy)**

> **[Ghi chú cho tác giả:]** *Hãy điền thông tin về bộ dữ liệu Regression của bạn vào định dạng bảng dưới đây tương tự như bài toán Classification.*

| Thông tin | Chi tiết |
| --- | --- |
| **Primary Source** | `[Link Kaggle/UCI của bộ Regression]` |
| **Original Custodian** | `[Tổ chức/Cơ quan cung cấp]` |
| **Temporal Coverage** | `[Khoảng thời gian thu thập]` |
| **Dataset Size** | `[Số lượng] bản ghi với [Số lượng] biến số.` |
| **Target Variable** | `[Tên biến mục tiêu cần dự đoán (liên tục)]` |

### **2.1 Attributes & Preprocessing Notes**

* **Volume:** `...` instances.
* **Dimensionality:** `...` attributes.
* **Key Feature Groups:**
* `[Nhóm đặc trưng 1]: [Liệt kê các cột]`
* `[Nhóm đặc trưng 2]: [Liệt kê các cột]`


* **Data License:** `[Loại giấy phép, ví dụ: CC0, MIT...]`

---

## **3. Classification Task Dataset (Bài toán Phân lớp)**

Dữ liệu được sử dụng là **Room Occupancy Estimation**, tập hợp các chuỗi thời gian ghi nhận từ hệ thống mạng lưới cảm biến IoT đa phương thức (multimodal sensor network) được lắp đặt trong một không gian phòng thực tế.

| Thông tin | Chi tiết |
| --- | --- |
| **Primary Source** | [Room Occupancy Estimation (Kaggle)](https://www.kaggle.com/datasets/ruchikakumbhar/room-occupancy-estimation/data) |
| **Original Custodian** | [UCI Machine Learning Repository (ID: 864)](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) |
| **Temporal Coverage** | *4 ngày liên tục* (Bắt đầu từ 22/12/2017) |
| **Geospatial Coverage** | Căn phòng thí nghiệm tiêu chuẩn kích thước *6m x 4.6m*, trang bị 7 cụm cảm biến. |
| **Observational Unit** | Tần suất lấy mẫu mỗi 30 giây (30-second interval readings). |
| **Dataset Size** | 10,129 bản ghi với 19 biến số đo lường vật lý và thời gian. |

> **Description:** Bộ dữ liệu ghi lại sự biến thiên của các yếu tố môi trường không gian kín bao gồm nhiệt độ, cường độ ánh sáng, mức độ âm thanh, nồng độ CO2 và tín hiệu chuyển động hồng ngoại (PIR). Biến mục tiêu là `Room_Occupancy_Count` (Số lượng người hiện diện trong phòng, dao động từ 0 đến 3).

### **3.1 Dataset Origin & Provenance**

* **Empirical Origin:** Dữ liệu gốc được trích xuất từ nghiên cứu *"Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes"* công bố tại hội nghị IEEE Globecom Workshops 2018. Dữ liệu được thu thập 100% từ môi trường vật lý, không sử dụng dữ liệu tổng hợp (synthetic data).
* **Date of Access:** *March 10, 2026*

### **3.2 Files & Attributes**

Tệp dữ liệu gốc được đặt tại: `data/raw/Occupancy_Estimation.csv`

* **Volume:** 10,129 instances (rows).
* **Dimensionality:** 19 attributes (16 continuous features + 2 temporal features + 1 target).
* **Key Feature Groups:**
* **Temporal:** `Date`, `Time` *(Cần loại bỏ trước khi đưa vào Design Matrix)*.
* **Thermodynamic:** `S1_Temp`, `S2_Temp`, `S3_Temp`, `S4_Temp`.
* **Illuminance:** `S1_Light`, `S2_Light`, `S3_Light`, `S4_Light`.
* **Acoustic:** `S1_Sound`, `S2_Sound`, `S3_Sound`, `S4_Sound`.
* **Air Quality:** `S5_CO2`, `S5_CO2_Slope` (Tốc độ thay đổi nồng độ CO2).
* **Motion:** `S6_PIR`, `S7_PIR` (Cảm biến hồng ngoại thụ động).
* **Target Variable:** `Room_Occupancy_Count` (Multiclass: 0, 1, 2, 3).



### **3.3 Licensing & Usage Rights**

* **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Dữ liệu được phép sử dụng cho mục đích nghiên cứu và giáo dục.
* **Citation Format:**
> *Singh, A. P., Jain, V., Chaudhari, S., Kraemer, F. A., Werner, S., & Garg, V. (2018). Machine Learning-Based Occupancy Estimation Using Multivariate Sensor Nodes. In 2018 IEEE Globecom Workshops (GC Wkshps) (pp. 1-6). IEEE. Dataset retrieved from UCI Machine Learning Repository [[https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)].*



---

## **4. Data Setup & Reproducibility (Hướng dẫn Tái lập)**

Để đảm bảo tính nhất quán của môi trường phát triển, vui lòng tuân thủ các bước sau để lấy dữ liệu nếu bạn clone dự án này từ một thiết bị khác:

1. Đảm bảo đã thiết lập Kaggle API token (`kaggle.json`) trong thư mục `~/.kaggle/`.
2. Chạy đoạn script tự động tải dữ liệu (sẽ tự động tải file nén và giải nén vào thư mục `raw/`):
```bash
# Ví dụ lệnh tải (tùy thuộc vào script bash bạn viết)
kaggle datasets download -d ruchikakumbhar/room-occupancy-estimation -p ./data/raw/ --unzip

```