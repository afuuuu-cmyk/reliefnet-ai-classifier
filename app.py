
import streamlit as st
import pickle
import os

# Load model
MODEL_PATH = os.path.join("model", "relief_model.pkl")

@st.cache(allow_output_mutation=True)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# UI
st.title("🆘 ReliefNet Help Categorizer")
st.write("Enter a help request message, and we'll predict its category.")

text_input = st.text_area("Enter request (e.g., 'I need urgent blood help')")

if st.button("Predict Category"):
    if text_input.strip() == "":
        st.warning("Please enter a request.")
    else:
        pred = model.predict([text_input])[0]
        st.success(f"✅ Predicted Category: **{pred}**")
