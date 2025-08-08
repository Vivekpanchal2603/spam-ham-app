import streamlit as st
import pickle
import numpy as np

# Load the model and vectorizer
model = pickle.load(open('spam_classifier_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# App Title
st.title("📬 Spam/Ham Mail Classifier")

# Instructions
st.subheader("Enter the email text below:")
st.text("The model will predict whether the email is spam or ham.")

# Text input
input_mail = st.text_area("✉️ Paste your email here:")

# Predict Button
if st.button("Predict"):
    if input_mail.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize the input
        input_data = vectorizer.transform([input_mail])

        # Make prediction
        prediction = model.predict(input_data)

        # Show result
        if prediction[0] == 1:
            st.success("✅ This email is **Ham** (Safe).")
            st.balloons()
        else:
            st.error("🚫 This email is **Spam**. Be cautious!")
