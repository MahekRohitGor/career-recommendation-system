import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------
# Load model and dataset safely
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'model', 'career_model.pkl')
data_path = os.path.join(BASE_DIR, 'dataset', 'career_data_preprocessed.csv')

model = joblib.load(model_path)
df = pd.read_csv(data_path)
feature_cols = [c for c in df.columns if c != 'Recommended_Career']

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="Career Recommendation System", page_icon="🎓", layout="wide")

st.title("🎓 Intelligent Career Recommendation System")
st.markdown("##### *Predict your ideal career based on skills and interests*")

st.sidebar.header("📋 Input Details")

# ----------------------------
# Education Mapping
# ----------------------------
st.sidebar.markdown("### 🎓 Education Level Mapping")
st.sidebar.info("1 → Bachelor  \n2 → Master  \n3 → PhD")

education_mapping = {
    "Bachelor's Degree (1)": 1,
    "Master's Degree (2)": 2,
    "PhD (3)": 3
}

# ----------------------------
# User Input Section
# ----------------------------
user_input = {}

for col in feature_cols:
    col_min = int(df[col].min())
    col_max = int(df[col].max())

    # Special handling for Education column
    if "Education" in col:
        edu_choice = st.sidebar.selectbox(
            "Select your Education Level",
            list(education_mapping.keys())
        )
        user_input[col] = education_mapping[edu_choice]

    elif df[col].nunique() == 2:
        user_input[col] = st.sidebar.selectbox(col.replace("_", " "), [0, 1])

    elif col_min == col_max:
        st.sidebar.write(f"⚠️ {col}: only one unique value ({col_min}) in data.")
        user_input[col] = col_min

    else:
        default_value = int(df[col].mean()) if not np.isnan(df[col].mean()) else col_min
        user_input[col] = st.sidebar.slider(col, col_min, col_max, default_value)

# ----------------------------
# Prediction Section
# ----------------------------
if st.sidebar.button("🔮 Recommend Career"):
    input_data = np.array([list(user_input.values())]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data).max()

    st.success(f"🎯 **Recommended Career:** {prediction}")
    st.info(f"Model Confidence: {probs*100:.2f}%")

    # Feature Importance Chart
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    important_features = importances.sort_values(ascending=False)[:10]

    st.subheader("📊 Key Features Impacting Career Prediction")
    st.bar_chart(important_features)

# ----------------------------
# Dataset Insights
# ----------------------------
with st.expander("🔍 View Dataset Insights"):
    st.write(df.describe())
    st.write("Total Samples:", len(df))