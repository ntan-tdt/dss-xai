# dss-xai
DSS XAI Streamlit
# Explainable Decision Support System (DSS) using ML & SHAP

## 📌 Giới thiệu
Dự án này xây dựng một **Hệ hỗ trợ ra quyết định (Decision Support System – DSS)** 
cho bài toán **phê duyệt khoản vay**, sử dụng:

- **Machine Learning**: Random Forest
- **Explainable AI (XAI)**: Feature Importance & SHAP
- **Giao diện Web**: Streamlit

Mục tiêu là **không chỉ đưa ra quyết định**, mà còn **giải thích rõ ràng lý do của quyết định**.

---

## 🎯 Bài toán
- **Đầu vào**: Thông tin khách hàng (tuổi, thu nhập, điểm tín dụng, dư nợ, …)
- **Đầu ra**:  
  - Quyết định: ✅ Duyệt / ❌ Từ chối  
  - Xác suất phê duyệt  
  - Giải thích bằng Explainable AI

---

## 🧠 Kiến trúc DSS
User Input
↓
Machine Learning Model (Random Forest)
↓
Decision Output (Approve / Reject)
↓
Explainable AI (Feature Importance + SHAP)
↓
Decision Explanation

---

## 📊 Dataset
- Dataset mô phỏng theo nghiệp vụ ngân hàng
- Các biến chính:
  - age
  - monthly_income
  - credit_score
  - loan_amount
  - employment_years
  - existing_debt
- Nhãn:
  - loan_approved (0 / 1)

---

## 🧪 Explainable AI sử dụng
### 1️⃣ Feature Importance (Global)
- Giải thích **biến nào ảnh hưởng nhiều nhất** đến quyết định
- Phù hợp cho quản lý & policy

### 2️⃣ SHAP (Local Explanation)
- Giải thích **từng quyết định cá nhân**
- Trả lời được câu hỏi:
  > “Vì sao khách hàng này bị từ chối?”

---

## 🖥️ Giao diện Streamlit
- Form nhập dữ liệu khách hàng
- Hiển thị:
  - Quyết định DSS
  - Feature Importance
  - SHAP Table
  - SHAP Bar Plot
  - SHAP Waterfall Plot

---

## 🚀 Cách chạy local

### Cài thư viện
```bash
pip install streamlit shap scikit-learn pandas numpy matplotlib

Chạy ứng dụng
Shellstreamlit run app.pyShow more lines

🌍 Deploy online
Ứng dụng được deploy bằng Streamlit Community Cloud
(Link demo sẽ được cập nhật tại đây)

📌 Công nghệ sử dụng

Python
Scikit-learn
SHAP
Streamlit
Pandas, NumPy, Matplotlib


🎓 Ứng dụng trong học phần

Machine Learning & AI trong DSS
Explainable AI
Data-driven Decision Making


👨‍🎓 Tác giả
Sinh viên: …
Môn học: Machine Learning & AI trong DSS
Giảng viên hướng dẫn: …

---

# ✅ PHẦN B — KỊCH BẢN DEMO + LỜI GIẢNG GIẢI (RẤT QUAN TRỌNG)

Phần này dùng cho:
- ✅ Demo trên lớp  
- ✅ Bảo vệ project  
- ✅ Chấm điểm cuối kỳ  

---

## 🎤 KỊCH BẢN DEMO (5–7 PHÚT)

---

## 🟢 BƯỚC 1 — Giới thiệu bài toán (30–45 giây)

### 🎙️ Lời nói gợi ý:
> “Nhóm em xây dựng một **hệ hỗ trợ ra quyết định cho bài toán phê duyệt khoản vay**.  
> Điểm khác biệt của hệ thống là **không chỉ dự đoán**, mà còn **giải thích được lý do của quyết định**.”

✅ Mục tiêu:
- Đặt bối cảnh DSS
- Nhấn mạnh **Explainable AI**

---

## 🟢 BƯỚC 2 — Nhập dữ liệu khách hàng (1 phút)

👉 Thao tác:
- Mở app Streamlit
- Nhập thông tin khách hàng:
  - Thu nhập trung bình
  - Điểm tín dụng trung bình

### 🎙️ Lời giảng:
> “Đây là dữ liệu đầu vào của DSS, tương ứng với thông tin khách hàng trong thực tế ngân hàng.”

✅ Nhấn mạnh:
- DSS hỗ trợ **quyết định bán cấu trúc**
- Con người vẫn là người ra quyết định cuối

---

## 🟢 BƯỚC 3 — Quyết định DSS (30 giây)

👉 Quan sát:
- ✅ Duyệt / ❌ Từ chối
- Xác suất phê duyệt

### 🎙️ Lời giảng:
> “Hệ thống sử dụng mô hình Random Forest để đưa ra quyết định và xác suất tương ứng.”

✅ Nhấn mạnh:
- ML = **Decision engine**
- DSS = **Decision support**, không thay thế con người

---

## 🟢 BƯỚC 4 — Feature Importance (1 phút)

👉 Chỉ vào biểu đồ Feature Importance

### 🎙️ Lời giảng:
> “Biểu đồ này cho thấy **những yếu tố ảnh hưởng nhiều nhất đến quyết định trên toàn hệ thống**,  
> ví dụ như điểm tín dụng và thu nhập.”

✅ Ý nghĩa DSS:
- Giúp quản lý:
  - Điều chỉnh policy
  - Kiểm tra tính hợp lý của mô hình

---

## 🟢 BƯỚC 5 — SHAP Table (1 phút)

👉 Chỉ vào bảng SHAP

### 🎙️ Lời giảng:
> “SHAP cho phép giải thích **quyết định của từng khách hàng**.  
> Giá trị SHAP dương làm tăng khả năng duyệt,  
> giá trị âm làm tăng khả năng từ chối.”

✅ Điểm cộng rất lớn:
- Trả lời được câu hỏi phản biện
- Phù hợp yêu cầu minh bạch

---

## 🟢 BƯỚC 6 — SHAP Waterfall Plot (WOW MOMENT – 1 phút)

👉 Cuộn xuống Waterfall Plot

### 🎙️ Lời giảng (rất quan trọng):
> “Waterfall Plot cho thấy **quyết định được hình thành như thế nào**,  
> bắt đầu từ giá trị trung bình của mô hình và bị kéo lên hoặc kéo xuống bởi từng yếu tố.”

👉 Chỉ rõ:
- Thu nhập kéo quyết định lên
- Dư nợ kéo quyết định xuống

✅ Đây là **Explainable DSS đúng chuẩn quốc tế**

---

## 🟢 BƯỚC 7 — Kết luận (30–45 giây)

### 🎙️ Lời kết:
> “Hệ thống kết hợp Machine Learning, DSS và Explainable AI,  
> giúp ra quyết định **chính xác, minh bạch và có thể giải trình**.  
> Đây là xu hướng quan trọng trong các hệ thống AI hiện đại.”

---

# ✅ GỢI Ý TRẢ LỜI CÂU HỎI PHẢN BIỆN

**❓ Vì sao cần SHAP?**  
👉 Vì DSS yêu cầu **giải thích quyết định**, không chỉ dự đoán.

**❓ DSS này có triển khai thực tế được không?**  
👉 Có, vì đã có:
- Giao diện
- Giải thích
- Khả năng deploy online

**❓ Mô hình có công bằng không?**  
👉 Có thể kiểm tra thêm fairness – hướng mở rộng.

---
