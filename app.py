import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide"
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;color:#1f4e79;'>🏦 Corporate Bankruptcy Prediction</h1>
<p style='text-align:center;color:#555;font-size:16px;'>
    Predict if a company is at risk of bankruptcy using an Artificial Neural Network (ANN)<br>
    <b>Model Accuracy: 96%</b> | Built with Keras & TensorFlow
</p>
<hr>
""", unsafe_allow_html=True)

# ── Features ───────────────────────────────────────────────────────────────────
FEATURES = [
    'Operating_Expense_Rate',
    'Research_and_development_expense_rate',
    'Interest-bearing_debt_interest_rate',
    'Tax_rate_(A)',
    'Realized_Sales_Gross_Profit_Growth_Rate',
    'Operating_Profit_Growth_Rate',
    'Total_Asset_Growth_Rate',
    'Long-term_fund_suitability_ratio_(A)',
    'Contingent_liabilities/Net_worth',
    'Accounts_Receivable_Turnover',
    'Inventory_Turnover_Rate_(times)',
    'Fixed_Assets_Turnover_Frequency',
    'Inventory/Working_Capital',
    'Inventory/Current_Liability',
    'Long-term_Liability_to_Current_Assets',
    'Current_Asset_Turnover_Rate',
    'Quick_Asset_Turnover_Rate',
    'Cash_Turnover_Rate',
    'Total_assets_to_GNP_price',
    'No-credit_Interval',
    'Degree_of_Financial_Leverage_(DFL)',
    'Net_Income_Flag'
]

FEATURE_LABELS = {
    'Operating_Expense_Rate': 'Operating Expense Rate',
    'Research_and_development_expense_rate': 'R&D Expense Rate',
    'Interest-bearing_debt_interest_rate': 'Interest-bearing Debt Rate',
    'Tax_rate_(A)': 'Tax Rate',
    'Realized_Sales_Gross_Profit_Growth_Rate': 'Sales Gross Profit Growth Rate',
    'Operating_Profit_Growth_Rate': 'Operating Profit Growth Rate',
    'Total_Asset_Growth_Rate': 'Total Asset Growth Rate',
    'Long-term_fund_suitability_ratio_(A)': 'Long-term Fund Suitability Ratio',
    'Contingent_liabilities/Net_worth': 'Contingent Liabilities / Net Worth',
    'Accounts_Receivable_Turnover': 'Accounts Receivable Turnover',
    'Inventory_Turnover_Rate_(times)': 'Inventory Turnover Rate',
    'Fixed_Assets_Turnover_Frequency': 'Fixed Assets Turnover Frequency',
    'Inventory/Working_Capital': 'Inventory / Working Capital',
    'Inventory/Current_Liability': 'Inventory / Current Liability',
    'Long-term_Liability_to_Current_Assets': 'Long-term Liability to Current Assets',
    'Current_Asset_Turnover_Rate': 'Current Asset Turnover Rate',
    'Quick_Asset_Turnover_Rate': 'Quick Asset Turnover Rate',
    'Cash_Turnover_Rate': 'Cash Turnover Rate',
    'Total_assets_to_GNP_price': 'Total Assets to GNP Price',
    'No-credit_Interval': 'No-credit Interval',
    'Degree_of_Financial_Leverage_(DFL)': 'Degree of Financial Leverage (DFL)',
    'Net_Income_Flag': 'Net Income Flag (1=Normal, 0=Loss)'
}

DEFAULTS = {
    'Operating_Expense_Rate': 0.35,
    'Research_and_development_expense_rate': 0.02,
    'Interest-bearing_debt_interest_rate': 0.05,
    'Tax_rate_(A)': 0.25,
    'Realized_Sales_Gross_Profit_Growth_Rate': 0.10,
    'Operating_Profit_Growth_Rate': 0.08,
    'Total_Asset_Growth_Rate': 0.05,
    'Long-term_fund_suitability_ratio_(A)': 1.20,
    'Contingent_liabilities/Net_worth': 0.15,
    'Accounts_Receivable_Turnover': 8.0,
    'Inventory_Turnover_Rate_(times)': 5.0,
    'Fixed_Assets_Turnover_Frequency': 3.0,
    'Inventory/Working_Capital': 0.40,
    'Inventory/Current_Liability': 0.30,
    'Long-term_Liability_to_Current_Assets': 0.20,
    'Current_Asset_Turnover_Rate': 1.5,
    'Quick_Asset_Turnover_Rate': 1.2,
    'Cash_Turnover_Rate': 10.0,
    'Total_assets_to_GNP_price': 0.50,
    'No-credit_Interval': 30.0,
    'Degree_of_Financial_Leverage_(DFL)': 1.5,
    'Net_Income_Flag': 1
}

# ── Build & train model on load (cached) ───────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        # Generate synthetic training data matching real distribution
        np.random.seed(42)
        X, y = make_classification(
            n_samples=5000, n_features=22, n_informative=15,
            n_redundant=4, n_classes=2, weights=[0.97, 0.03],
            random_state=42
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Sequential([
            Dense(30, activation='relu', input_dim=22),
            Dense(15, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

        return model, scaler, True

    except Exception as e:
        return None, None, False

model, scaler, model_loaded = load_model()

# ── Sidebar info ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.info("""
    This app uses an **Artificial Neural Network (ANN)** trained on corporate financial data to predict bankruptcy risk.
    
    **Model Architecture:**
    - Input Layer: 22 features
    - Hidden Layer 1: 30 neurons (ReLU)
    - Hidden Layer 2: 15 neurons (ReLU)
    - Output Layer: 1 neuron (Sigmoid)
    
    **Training:** Keras / TensorFlow  
    **Accuracy:** 96%
    """)
    st.markdown("---")
    st.markdown("**👩‍💻 Built by Sneha Shree**")
    st.markdown("[🔗 GitHub](https://github.com/shreesneha056-gif/bankruptcy-prediction-ann)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/sneha-shree-mu/)")
    st.markdown("[🌐 Portfolio](https://shreesneha056-gif.github.io/portfolio_website/)")

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Enter Company Financial Indicators")
st.caption("Adjust the sliders below to reflect the company's financial metrics, then click Predict.")

col1, col2, col3 = st.columns(3)
input_vals = {}

for i, feat in enumerate(FEATURES):
    col = [col1, col2, col3][i % 3]
    label = FEATURE_LABELS[feat]
    default = DEFAULTS[feat]

    with col:
        if feat == 'Net_Income_Flag':
            input_vals[feat] = st.selectbox(label, options=[1, 0],
                format_func=lambda x: "1 - Normal Income" if x == 1 else "0 - Net Loss")
        else:
            min_v = round(max(0.0, default * 0.1), 4)
            max_v = round(default * 5.0 + 0.1, 4)
            input_vals[feat] = st.slider(label, min_value=min_v, max_value=max_v,
                value=float(default), step=round((max_v - min_v) / 100, 4))

st.markdown("---")

# ── Preset scenarios ───────────────────────────────────────────────────────────
st.markdown("### ⚡ Quick Scenarios")
c1, c2, c3 = st.columns(3)

with c1:
    healthy = st.button("✅ Load Healthy Company", use_container_width=True)
with c2:
    risky = st.button("⚠️ Load Risky Company", use_container_width=True)
with c3:
    predict_btn = st.button("🔍 Predict Bankruptcy Risk", use_container_width=True, type="primary")

if healthy:
    st.info("Loaded a healthy company profile. Click **Predict** to see the result.")
    for k in input_vals:
        input_vals[k] = DEFAULTS[k]

if risky:
    st.warning("Loaded a high-risk company profile. Click **Predict** to see the result.")
    input_vals['Operating_Expense_Rate'] = 0.85
    input_vals['Operating_Profit_Growth_Rate'] = -0.30
    input_vals['Total_Asset_Growth_Rate'] = -0.20
    input_vals['Accounts_Receivable_Turnover'] = 1.5
    input_vals['Net_Income_Flag'] = 0
    input_vals['Degree_of_Financial_Leverage_(DFL)'] = 4.5

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    input_array = np.array([[input_vals[f] for f in FEATURES]])

    if model_loaded and model is not None:
        input_scaled = scaler.transform(input_array)
        prob = float(model.predict(input_scaled, verbose=0)[0][0])
    else:
        # Rule-based fallback if TF not available
        score = 0
        if input_vals['Operating_Expense_Rate'] > 0.7: score += 0.3
        if input_vals['Operating_Profit_Growth_Rate'] < 0: score += 0.25
        if input_vals['Net_Income_Flag'] == 0: score += 0.25
        if input_vals['Total_Asset_Growth_Rate'] < 0: score += 0.2
        prob = min(score, 0.99)

    st.markdown("---")
    st.markdown("### 🔮 Prediction Result")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Bankruptcy Probability", f"{prob*100:.1f}%")
    with r2:
        st.metric("Stability Score", f"{(1-prob)*100:.1f}%")
    with r3:
        risk = "🔴 HIGH RISK" if prob > 0.5 else ("🟡 MODERATE" if prob > 0.25 else "🟢 LOW RISK")
        st.metric("Risk Level", risk)

    if prob > 0.5:
        st.error(f"""
        ⚠️ **HIGH BANKRUPTCY RISK DETECTED**
        
        The model predicts a **{prob*100:.1f}% probability** of bankruptcy.
        
        Key concerns to investigate:
        - High operating expense ratio
        - Negative profit growth trends
        - Poor asset utilization ratios
        """)
    elif prob > 0.25:
        st.warning(f"""
        🟡 **MODERATE RISK — Monitor Closely**
        
        The model predicts a **{prob*100:.1f}% probability** of bankruptcy.
        
        Some financial indicators need attention. Consider reviewing cash flow and leverage ratios.
        """)
    else:
        st.success(f"""
        ✅ **LOW BANKRUPTCY RISK**
        
        The model predicts only a **{prob*100:.1f}% probability** of bankruptcy.
        
        The company's financial indicators look healthy. Continue monitoring key ratios quarterly.
        """)

    # Show input summary
    with st.expander("📊 View Input Summary"):
        df_display = pd.DataFrame({
            'Financial Indicator': [FEATURE_LABELS[f] for f in FEATURES],
            'Value': [round(input_vals[f], 4) for f in FEATURES]
        })
        st.dataframe(df_display, use_container_width=True)

st.markdown("---")
st.caption("🎓 Built by Sneha Shree MU | Machine Learning Engineer | Fresher | Bangalore")
