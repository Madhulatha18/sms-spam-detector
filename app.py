import streamlit as st
import joblib

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

st.title("SMS Spam Detector")
msg = st.text_area("Paste a message:")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        pred = model.predict(tfidf.transform([msg]))[0]
        if pred == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")