import streamlit as st
import pandas as pd
import plotly.express as px

st.title("RetailPulse Dashboard")
df = pd.read_excel("merged_cleaned_retail_data.xlsx")
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.subheader("Dataset Shape")
st.write(df.shape)
numeric_cols = df.select_dtypes(include='number').columns
st.sidebar.title("Filters")

selected_column = st.sidebar.selectbox( 
    "Select Numeric Column",
    numeric_cols
)
st.subheader(f"Distribution of {selected_column}")

fig = px.histogram(
    df,
    x=selected_column
)

st.plotly_chart(fig)
st.subheader("Missing Values")

st.write(df.isnull().sum())
st.subheader("Correlation Matrix")

corr = df[numeric_cols].corr()

st.dataframe(corr)