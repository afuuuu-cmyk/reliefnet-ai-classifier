import streamlit as st
import pickle
import os

# 🔍 Path to the model
MODEL_PATH = os.path.join("model", "relief_model.pkl")

# ✅ Use updated Streamlit caching
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Make sure 'relief_model.pkl' exists in the 'model/' folder.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()

# 🖼️ UI setup
st.set_page_config(page_title="ReliefNet Classifier", page_icon="🆘")
st.title("🆘 ReliefNet Help Categorizer")
st.markdown("Enter a help request below, and we'll predict the type of help you need.")

# ✍️ Text input from user
text_input = st.text_area("📝 Enter your help request", placeholder="e.g., I need urgent blood for my father")

# 📤 Predict on button click
if st.button("🔍 Predict Category"):
    if text_input.strip() == "":
        st.warning("Please enter a help request first.")
    else:
        try:
            prediction = model.predict([text_input])[0]
            st.success(f"✅ Predicted Category: **{prediction}**")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

