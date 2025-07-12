import streamlit as st
import pickle
import os

# Path to the model
MODEL_PATH = os.path.join("model", "relief_model.pkl")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please upload 'relief_model.pkl' inside the 'model/' folder.")
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Emoji map
CATEGORY_ICONS = {
    "Emergency": "🚨",
    "Medical": "🧑‍⚕️",
    "Food": "🍲",
    "Transport": "🚕",
    "Supplies": "📦"
}

# App layout
st.set_page_config(page_title="ReliefNet Help Categorizer", page_icon="🆘")
st.title("🆘 ReliefNet - AI Help Request Classifier")
st.markdown("💬 *Enter a help request below. We'll detect the type of help you need and suggest a category.*")

# Sidebar
with st.sidebar:
    st.markdown("## 📘 About ReliefNet AI")
    st.markdown("""
    This tool uses a machine learning model to classify help requests into:
    - 🚨 Emergency
    - 🧑‍⚕️ Medical
    - 🍲 Food
    - 🚕 Transport
    - 📦 Supplies
    \nBuilt with ❤️ using Streamlit + scikit-learn.
    """)

# Input area
text_input = st.text_area("📝 Enter your help request", placeholder="e.g. I need urgent medicine for my mom")

# Predict button
if st.button("🔍 Predict Category"):
    if text_input.strip() == "":
        st.warning("Please type a help request.")
    else:
        try:
            category = model.predict([text_input])[0]
            emoji = CATEGORY_ICONS.get(category, "❓")
            st.success(f"{emoji} Predicted Category: **{category}**")
        except Exception as e:
            st.error(f"❌ Error: {e}")
