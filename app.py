import streamlit as st
import numpy as np
import pandas as pd

# ERROR HANDLING: Agar joblib ya sklearn install nahi hai toh UI handle karega
try:
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    SCALER_AVAILABLE = True
except ImportError:
    SCALER_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Market & Economic Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Market & Economic Predictor App")
st.write("Is app ke zariye aap economic indicators ke aadhar par **IHSG (Stock Market Index)** predict kar sakte hain.")

st.markdown("---")

# Package Validation Check
if not SKLEARN_AVAILABLE or not SCALER_AVAILABLE:
    st.error("⚠️ **Dependency Error:** Aapke system me `joblib` ya `scikit-learn` install nahi hai.")
    st.info("App ko chalane ke liye terminal me ye command run karein: `pip install scikit-learn joblib streamlit`")
    st.stop()

# ----------------- LOAD ARTIFACTS -----------------
@st.cache_resource
def load_ml_components():
    try:
        # joblib se files load karna
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns, None
    except Exception as e:
        return None, None, None, str(e)

model, scaler, columns, load_error = load_ml_components()

# Agar pickle files nahi milti hain, toh UI crash nahi hoga, error dikhayega
if load_error:
    st.warning("⚠️ **Model Files Missing:** `model.pickle`, `scaler.pickle`, ya `columns.pickle` nahi mili.")
    st.info("Pehle apni trained files ko isi folder me rakhein jahan aapki `app.py` file hai.")
    
    # Dummy mode enable karein taaki UI ka look and feel check kiya ja sake
    st.subheader("🛠️ UI Preview Mode (Testing)")
else:
    st.success("✅ ML Model, Scaler, aur Columns successfully load ho gaye hain!")

# ----------------- USER INPUTS (UI FIELDS) -----------------
st.subheader("📋 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    # Year input ko integer rakha hai aapke logic ke mutabik
    year = st.number_input("Year (Saal)", min_value=2000, max_value=2030, value=2026, step=1)
    oil = st.number_input("OIL Price", min_value=0.0, value=75.0, step=0.1)
    gold = st.number_input("GOLD Price", min_value=0.0, value=1100.0, step=0.1)
    usdidr = st.number_input("USD to IDR Rate", min_value=0.0, value=9300.0, step=1.0)

with col2:
    sp500 = st.number_input("S&P 500 Index", min_value=0.0, value=1100.0, step=0.1)
    vix = st.number_input("VIX (Volatility Index)", min_value=0.0, value=20.0, step=0.1)
    cpi = st.number_input("CPI (Inflation)", min_value=0.0, value=5.13, step=0.01)
    bi_rate = st.number_input("BI Rate (Local Interest)", min_value=0.0, value=6.5, step=0.1)
    
# US Rate niche full width me le lete hain
us_rate = st.number_input("US Rate (Interest)", min_value=0.0, value=0.11, step=0.01)

st.markdown("---")

# ----------------- PREDICTION LOGIC -----------------
if st.button("🚀 Predict IHSG Index", type="primary"):
    
    if load_error:
        st.error(f"Prediction fail ho gayi kyunki model files loaded nahi hain. Error: {load_error}")
    else:
        try:
            # 1. Input data ka dictionary banana (Year ko chhodkar baki sab scale honge)
            # Dhyaan rahe: Order wahi hona chahiye jo training ke waqt tha!
            
            # Scaled features ki list
            scaled_features = np.array([[oil, gold, usdidr, sp500, vix, cpi, bi_rate, us_rate]])
            
            # Features par StandardScaler chalana
            features_scaled = scaler.transform(scaled_features)
            
            # Final array banana jisme 'Year' pehle column me ho aur baaki scaled features baad me
            # Kyunki humne decide kiya tha ki Year par scaling nahi lagani hai
            final_features = np.hstack((np.array([[year]]), features_scaled))
            
            # 2. Model se prediction karna
            prediction = model.predict(final_features)
            
            # 3. Output ko UI par show karna
            st.balloons()
            st.markdown(f"### 🎯 Predicted IHSG Stock Index: **{prediction[0]:,.2f}**")
            
        except Exception as pred_error:
            st.error("⚠️ Prediction ke dauran ek error aaya!")
            st.code(str(pred_error))

# Footer
st.markdown("---")
st.caption("Developed for Amir Sohail | Financial Analytics ML Model")