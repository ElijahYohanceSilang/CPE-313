import pickle
import sklearn
import streamlit as st

st.title("Model Version Checker")

try:
    with open("Team18/team18_model.pkl", "rb") as f:
        # This looks into the pickle metadata
        model = pickle.load(f)
        st.write("Successfully loaded!")
        st.write(f"Your Current Streamlit Scikit-Learn Version: {sklearn.__version__}")
except Exception as e:
    st.error(f"Error: {e}")
