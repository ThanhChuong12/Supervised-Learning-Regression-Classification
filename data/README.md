# **Data Catalog & Provenance**

Tài liệu này cung cấp thông tin chi tiết về các bộ dữ liệu được sử dụng trong đồ án, bao gồm nguồn gốc, cấu trúc từ điển dữ liệu (data dictionary), và các quy định về bản quyền. Đồ án này phục vụ hai bài toán Machine Learning cốt lõi: **Hồi quy (Regression)** và **Phân lớp (Classification)**.

## **Table of Contents**

- [**Data Catalog \& Provenance**](#data-catalog--provenance)
  - [**Table of Contents**](#table-of-contents)
  - [**1. Directory Structure**](#1-directory-structure)
  - [**2. Regression Task Dataset (Bài toán Hồi quy)**](#2-regression-task-dataset-bài-toán-hồi-quy)
    - [**2.1 Dataset Origin \& Provenance**](#21-dataset-origin--provenance)
    - [**2.2 Files \& Attributes**](#22-files--attributes)
    - [**2.3 Licensing \& Usage Rights**](#23-licensing--usage-rights)
  - [**3. Classification Task Dataset (Bài toán Phân lớp)**](#3-classification-task-dataset-bài-toán-phân-lớp)
    - [**3.1 Dataset Origin \& Provenance**](#31-dataset-origin--provenance)
    - [**3.2 Files \& Attributes**](#32-files--attributes)
    - [**3.3 Licensing \& Usage Rights**](#33-licensing--usage-rights)
  - [**4. Data Setup \& Reproducibility**](#4-data-setup--reproducibility)

## **1. Directory Structure**

Thư mục `data/` được tổ chức theo tiêu chuẩn vòng đời dữ liệu, đảm bảo nguyên tắc không bao giờ ghi đè lên dữ liệu gốc:

```text
data/
├── raw/                 # Dữ liệu gốc nguyên bản.
│   ├── Energy_Use.csv
│   └── Occupancy_Estimation.csv
├── processed/           # Dữ liệu đã làm sạch, chuẩn hóa (Standardized/Design Matrix).
└── README.md            # Tài liệu bạn đang đọc.
```
---

## **2. Regression Task Dataset (Bài toán Hồi quy)**

Dữ liệu được sử dụng là **Appliances Energy Prediction**, bộ dữ liệu ghi nhận mức tiêu thụ năng lượng của các thiết bị gia dụng kết hợp với dữ liệu nhiệt độ, độ ẩm trong nhà đo lường bằng mạng cảm biến không dây ZigBee và kết hợp thông tin thời tiết ngoài trời.

| Thông tin | Chi tiết |
| --- | --- |
| **Primary Source** | [Appliances Energy Prediction (Kaggle)](https://www.kaggle.com/datasets/sohommajumder21/appliances-energy-prediction-data-set) |
| **Original Custodian** | [UCI Machine Learning Repository (ID: 374)](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction) |
| **Temporal Coverage** | *4.5 tháng*. Tần suất lấy mẫu mỗi 10 phút. |
| **Dataset Size** | 19,735 bản ghi với 29 biến số. |
| **Target Variable** | `Appliances` (Mức tiêu thụ năng lượng của thiết bị gia dụng - đơn vị Wh) |

> **Description:** Bộ dữ liệu được ghi nhận với tần suất 10 phút/lần trong khoảng 4.5 tháng. Các điều kiện về nhiệt độ và độ ẩm trong nhà được theo dõi bằng mạng lưới cảm biến không dây ZigBee. Mỗi nút mạng không dây truyền dữ liệu nhiệt độ và độ ẩm khoảng 3.3 phút một lần. Sau đó, dữ liệu không dây được tính trung bình cho mỗi chu kỳ 10 phút. Dữ liệu điện năng tiêu thụ được ghi nhận mỗi 10 phút thông qua đồng hồ đo m-bus. Dữ liệu thời tiết từ trạm khí tượng sân bay gần nhất (Sân bay Chievres, Bỉ) được lấy từ nguồn dữ liệu công khai Reliable Prognosis (rp5.ru), sau đó hợp nhất với các bộ dữ liệu thực nghiệm thông qua cột ngày giờ.

### **2.1 Dataset Origin & Provenance**

* **Empirical Origin:** Dữ liệu gốc được trích xuất từ nghiên cứu *"Data driven prediction models of energy use of appliances in a low-energy house"* công bố trên tạp chí khoa học **Energy and Buildings** (Tập 140, tháng 4 năm 2017) bởi các tác giả thuộc trường Đại học Mons (UMONS), Bỉ. Dữ liệu được đo lường 100% từ môi trường vật lý tại một ngôi nhà kết hợp cùng dữ liệu thời tiết thực tế từ trạm khí tượng.
* **Date of Access:** *March 10, 2026*

### **2.2 Files & Attributes**

Tệp dữ liệu gốc được đặt tại: `data/raw/Energy_Use.csv`

* **Volume:** 19,735 instances (rows).
* **Dimensionality:** 29 attributes (27 continuous features, 1 temporal feature, 1 target).
* **Key Feature Groups:**
  * **Temporal:** 
    * `date`: Thời gian ghi nhận theo định dạng year-month-day hour:minute:second.
  * **Indoor Conditions (Temperature & Humidity):**
    * `T1`, `RH_1`: Nhiệt độ (°C) và Độ ẩm (%) khu vực bếp.
    * `T2`, `RH_2`: Nhiệt độ (°C) và Độ ẩm (%) khu vực phòng khách.
    * `T3`, `RH_3`: Nhiệt độ (°C) và Độ ẩm (%) khu vực phòng giặt.
    * `T4`, `RH_4`: Nhiệt độ (°C) và Độ ẩm (%) phòng làm việc.
    * `T5`, `RH_5`: Nhiệt độ (°C) và Độ ẩm (%) phòng tắm.
    * `T7`, `RH_7`: Nhiệt độ (°C) và Độ ẩm (%) phòng ủi đồ.
    * `T8`, `RH_8`: Nhiệt độ (°C) và Độ ẩm (%) phòng thanh thiếu niên 2.
    * `T9`, `RH_9`: Nhiệt độ (°C) và Độ ẩm (%) phòng bố mẹ.
  * **Outdoor/Weather:** 
    * `T6`, `RH_6`: Nhiệt độ (°C) và Độ ẩm (%) khu vực bên ngoài tòa nhà (mặt phía Bắc).
    * `T_out`: Nhiệt độ ngoài trời (°C).
    * `Press_mm_hg`: Áp suất khí quyển (mm Hg).
    * `RH_out`: Độ ẩm ngoài trời (%).
    * `Windspeed`: Tốc độ gió (m/s).
    * `Visibility`: Tầm nhìn xa (km).
    * `Tdewpoint`: Nhiệt độ đọng sương (°C).
  * **Other Energy Usage:** 
    * `lights`: Mức tiêu thụ năng lượng của các thiết bị chiếu sáng trong nhà (Wh).
  * **Random Variables:** 
    * `rv1`, `rv2`: Biến ngẫu nhiên đưa vào để kiểm thử các mô hình hồi quy và dùng để lọc các thuộc tính không mang tính dự báo.
  * **Target Variable:** 
    * `Appliances`: Mức tiêu thụ năng lượng của thiết bị gia dụng (Wh).

### **2.3 Licensing & Usage Rights**

* **Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Dữ liệu được chia sẻ và tùy chỉnh cho mọi mục đích với điều kiện ghi nhận nguồn gốc rõ ràng.
* **Citation Format:**
> *Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.*

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

## **4. Data Setup & Reproducibility**

Để đảm bảo tính nhất quán của môi trường phát triển và tuân thủ nguyên tắc quản lý dữ liệu, vui lòng thực hiện các bước sau để thiết lập dữ liệu gốc (raw data) khi clone đồ án này về máy:

**Bước 1: Thiết lập Kaggle API**

Đảm bảo bạn đã cài đặt thư viện `kaggle` và đặt file `kaggle.json` (chứa API token của bạn) vào đúng thư mục hệ thống:
* **Linux/Mac:** `~/.kaggle/kaggle.json`
* **Windows:** `C:\Users\<Username>\.kaggle\kaggle.json`

**Bước 2: Chạy Script tải dữ liệu**

Bạn có thể chạy trực tiếp các lệnh sau trong terminal hoặc lưu thành một file script (ví dụ: `scripts/setup_data.sh`) để tự động tải và giải nén dữ liệu vào thư mục `data/raw/`:

```bash
#!/bin/bash

# Create raw data directory if it doesn't exist
mkdir -p ./data/raw/

echo "Downloading Regression dataset (Appliances Energy Prediction)..."
kaggle datasets download -d sohommajumder21/appliances-energy-prediction-data-set -p ./data/raw/ --unzip

echo "Downloading Classification dataset (Room Occupancy Estimation)..."
kaggle datasets download -d ruchikakumbhar/room-occupancy-estimation -p ./data/raw/ --unzip

echo "Data download complete! Please check the ./data/raw/ directory."
```
*(Lưu ý: Sau khi giải nén, bạn có thể cần đổi tên các file `.csv` về đúng chuẩn `Energy_Use.csv` và `Occupancy_Estimation.csv` như mô tả trong cấu trúc thư mục nếu tên file gốc từ Kaggle có sự khác biệt).*
