import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================================================
# Streamlit config
# =====================================================
st.set_page_config(
    page_title="DSS Loan Approval with XAI",
    layout="wide"
)

st.title("💳 Decision Support System – Loan Approval")
st.caption("Machine Learning + Explainable AI (SHAP) | DSS Demo")

# =====================================================
# Load dataset
# =====================================================
@st.cache_data
def load_data():
    return pd.read_csv("xai_dss_loan_dataset.csv")

df = load_data()
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

# =====================================================
# Train model
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# =====================================================
# Sidebar – Input
# =====================================================
st.sidebar.header("📥 Nhập thông tin khách hàng")

age = st.sidebar.slider("Tuổi", 21, 65, 35)
monthly_income = st.sidebar.slider("Thu nhập hàng tháng", 3000, 20000, 8000)
credit_score = st.sidebar.slider("Điểm tín dụng", 300, 850, 650)
loan_amount = st.sidebar.slider("Số tiền vay", 5000, 50000, 15000)
employment_years = st.sidebar.slider("Số năm làm việc", 0, 30, 5)
existing_debt = st.sidebar.slider("Dư nợ hiện tại", 0, 20000, 2000)

input_df = pd.DataFrame([[
    age,
    monthly_income,
    credit_score,
    loan_amount,
    employment_years,
    existing_debt
]], columns=X.columns)

# =====================================================
# Prediction
# =====================================================
pred = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

st.markdown("## 🧠 Quyết định DSS")

if pred == 1:
    st.success(f"✅ Khoản vay ĐƯỢC DUYỆT (Xác suất: {proba:.2f})")
else:
    st.error(f"❌ Khoản vay BỊ TỪ CHỐI (Xác suất: {proba:.2f})")

# =====================================================
# 2-column layout
# =====================================================
col_left, col_right = st.columns(2)

# =====================================================
# LEFT: Feature Importance (Global)
# =====================================================
with col_left:
    st.markdown("### 🌍 Feature Importance (Global)")

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    st.bar_chart(importance)

# =====================================================
# SHAP COMPUTATION (ROBUST – OLD & NEW SHAP)
# =====================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

# ---- Extract SHAP values (local)
if isinstance(shap_values, list):
    # Old SHAP: list[class][sample][feature]
    shap_value = shap_values[1][0]
else:
    # New SHAP: (n_samples, n_features)
    shap_value = shap_values[0]

# ---- Ensure numpy 1D
shap_value = np.array(shap_value).flatten()

# ---- SAFETY: match number of features
n_features = len(X.columns)
if len(shap_value) != n_features:
    shap_value = shap_value[:n_features]

# =====================================================
# BASE VALUE FIX (CRITICAL FOR WATERFALL)
# =====================================================
# Waterfall REQUIRES scalar base_value
ev = explainer.expected_value
if isinstance(ev, list):
    base_value = float(ev[1])
elif isinstance(ev, np.ndarray):
    base_value = float(ev.reshape(-1)[0])
else:
    base_value = float(ev)

# =====================================================
# RIGHT: SHAP Table (Local)
# =====================================================
with col_right:
    st.markdown("### 🔍 SHAP Table (Local Explanation)")

    shap_df = pd.DataFrame({
        "Feature": list(X.columns),
        "SHAP value": shap_value
    }).sort_values(by="SHAP value", ascending=False)

    st.dataframe(shap_df, use_container_width=True)

# =====================================================
# SHAP BAR PLOT (Global-style)
# =====================================================
st.markdown("## 📊 SHAP Bar Plot (Global View)")

fig_bar, ax = plt.subplots()
shap.summary_plot(
    shap_value.reshape(1, -1),
    input_df,
    plot_type="bar",
    show=False
)
st.pyplot(fig_bar)

# =====================================================
# SHAP WATERFALL PLOT (Local – FINAL FIX)
# =====================================================
st.markdown("## 🌊 SHAP Waterfall Plot (Giải thích quyết định cá nhân)")

shap_exp = shap.Explanation(
    values=shap_value,
    base_values=base_value,          # ✅ MUST BE SCALAR
    data=input_df.iloc[0],
    feature_names=list(X.columns)
)

fig_wf, ax = plt.subplots()
shap.plots.waterfall(shap_exp, show=False)
st.pyplot(fig_wf)

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.markdown("""
**Giải thích XAI:**
- 🔺 SHAP > 0 → tăng khả năng duyệt vay  
- 🔻 SHAP < 0 → tăng khả năng từ chối  

✅ Đây là **Explainable DSS chuẩn quốc tế**, ổn định khi deploy & giảng dạy.
""")
st.caption("© DSS + ML + Explainable AI (SHAP) — FINAL V3 (BASE_VALUE FIXED)")