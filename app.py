# app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("heart.csv")

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# -----------------------------
# SIDEBAR MENU
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "Prediction",
        "Statistical Analysis"
    ]
)

# =====================================================
# INTRODUCTION PAGE
# =====================================================
if page == "Introduction":

    st.title("❤️ Heart Disease Prediction App")

    st.image(
        "https://images.unsplash.com/photo-1584515933487-779824d29309",
        use_container_width=True
    )

    st.markdown("""
    ## About This Project

    This Streamlit application predicts whether a patient
    is likely to have heart disease based on medical data.

    ### Features
    - Heart Disease Prediction
    - Dataset Visualization
    - Statistical Analysis
    - Machine Learning Model

    ### Machine Learning Model Used
    Logistic Regression

    ### Model Accuracy
    """)

    st.success(f"Accuracy: {accuracy * 100:.2f}%")

# =====================================================
# PREDICTION PAGE
# =====================================================
elif page == "Prediction":

    st.title("🩺 Heart Disease Prediction")

    st.subheader("Enter Patient Details")

    age = st.number_input("Age", 1, 100, 50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

    if st.button("Predict"):

        input_data = [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]]

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High chance of Heart Disease")
        else:
            st.success("✅ Low chance of Heart Disease")

# =====================================================
# STATISTICAL ANALYSIS PAGE
# =====================================================
elif page == "Statistical Analysis":

    st.title("📊 Statistical Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Dataset Statistics")
    st.write(df.describe())

    st.subheader("Target Value Count")
    st.bar_chart(df["target"].value_counts())

    st.subheader("Correlation Heatmap Data")
    st.write(df.corr())

    st.subheader("Column Selector")

    selected_column = st.selectbox(
        "Choose a column",
        df.columns
    )

    st.line_chart(df[selected_column])
