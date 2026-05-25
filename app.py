


# app.py
import streamlit as st
import pandas as pd

# Page title
st.set_page_config(page_title="Simple Streamlit App", layout="centered")

# Title
st.title("❤️ Heart Disease Dataset Viewer")

# Load dataset
df = pd.read_csv("heart.csv")

# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Basic info
st.subheader("Dataset Information")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# Column selector
column = st.selectbox("Select a column to view", df.columns)

# Show selected column data
st.subheader(f"Values in '{column}'")
st.write(df[column])

# Simple statistics
st.subheader("Statistics")
st.write(df.describe())

# Footer
st.success("Streamlit app running successfully!")

