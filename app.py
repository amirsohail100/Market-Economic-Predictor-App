import streamlit as st
import numpy as np
import pandas as pd

# ERROR HANDLING: Packages verify karne ke liye safe initialization
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

# Page Layout Setup
st.set_page_config(
    page_title="Market & Economic Predictor",
    page_icon="📈",
    layout="centered"
)

st.title("📈 ! Market & Economic Predictor App")
st.write("Is app ke zariye aap economic indicators ke aadhar par **IHSG (Stock Market Index)** predict kar sakte hain.")
st.markdown("---")

# Package Check Alert
if not SKLEARN_AVAILABLE or not SCALER_AVAILABLE:
    st.error("⚠️ **Dependency Error:** System me `joblib` ya `scikit-learn` install nahi hai.")
    st.info("App chalane ke liye terminal me ye command run karein: `pip install scikit-learn joblib streamlit`")
    st.stop()

# ----------------- LOAD ARTIFACTS (.pkl files) -----------------
@st.cache_resource
def load_ml_components():
    try:
        # Pkl files load karna
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns, None
    except Exception as e:
        # Agar koi version mismatch ya file loading bug ho toh exact traceback return karna
        return None, None, None, str(e)

model, scaler, columns, load_error = load_ml_components()

# UI System Error Handler
if load_error:
    st.error(f"❌ **System Technical Error Details:** {load_error}")
    st.warning("⚠️ **Model Status:** `.pkl` files perfectly load nahi ho paayi hain.")
    st.info("💡 **Fix Hint:** Agar version error hai toh requirement.txt file sync karke Streamlit Cloud dashboard se App ko **Reboot** karein.")
    st.markdown("---")
    st.subheader("🛠️ UI Preview Mode (Testing Only)")
else:
    st.success("✅ Production Gradient Boosting Model aur Preprocessing Scaler load ho gaya hai!")

# ----------------- USER INPUTS (STREAMLIT UI) -----------------
st.subheader("📋 Enter Input Features")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year (Saal)", min_value=2000, max_value=2030, value=2026, step=1)
    oil = st.number_input("OIL Price", min_value=0.0, value=75.30, step=0.1)
    gold = st.number_input("GOLD Price", min_value=0.0, value=1100.30, step=0.1)
    usdidr = st.number_input("USD to IDR Rate", min_value=0.0, value=9302.00, step=1.0)

with col2:
    sp500 = st.number_input("S&P 500 Index", min_value=0.0, value=1100.00, step=0.1)
    vix = st.number_input("VIX (Volatility Index)", min_value=0.0, value=20.00, step=0.1)
    cpi = st.number_input("CPI (Inflation)", min_value=0.0, value=5.13, step=0.01)
    bi_rate = st.number_input("BI Rate (Local Interest)", min_value=0.0, value=6.50, step=0.1)

# Full Width Input Field
us_rate = st.number_input("US Rate (Interest)", min_value=0.0, value=0.11, step=0.01)

st.markdown("---")

# ----------------- PREDICTION PIPELINE EXECUTION -----------------
if st.button("🚀 Predict IHSG Index", type="primary"):
    
    if load_error:
        st.error("Prediction execute nahi ho sakti jab tak artifacts loading crash mode me hain.")
    else:
        try:
            # 1. Continuous features ka array format scaling ke liye
            raw_continuous_features = np.array([[oil, gold, usdidr, sp500, vix, cpi, bi_rate, us_rate]], dtype=float)
            
            # 2. Fitted StandardScaler se transform karna (Transforming 8 features)
            features_scaled = scaler.transform(raw_continuous_features)
            
            # 3. EXACT ALIGNMENT MATCH: Continuous scaled features pehle aur year column aakhiri me stack hoga
            year_input = np.array([[int(year)]])
            final_model_input = np.hstack((features_scaled, year_input))
            
            # 4. DataFrame Wrapper: Taaki features sequence exact list se identify ho sake
            feature_names = ['OIL', 'GOLD', 'USDIDR', 'SP500', 'VIX', 'CPI', 'BI_rate', 'US_rate', 'year']
            final_df = pd.DataFrame(final_model_input, columns=feature_names)
            
            # 5. Gradient Boosting Prediction
            prediction = model.predict(final_df)
            
            # Results UI Showcase
            st.balloons()
            st.markdown("### 🎯 Prediction Result")
            st.success(f"Predicted IHSG Stock Index Target: **{prediction[0]:,.3f}**")
            
        except Exception as pred_error:
            st.error("⚠️ Prediction pipeline mapping execute karte waqt array structural error aaya!")
            st.code(str(pred_error))

# Footer Segment
st.markdown("---")
st.caption("Developed for Amir Sohail | Advanced Financial Analytics ML Pipeline")