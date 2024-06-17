import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import warnings
warnings.filterwarnings("ignore")

# Function to load TF-IDF vectorizer
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

# Function to load the trained model
def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

# Function to predict toxicity
def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Streamlit UI
st.set_page_config(page_title="Toxicity Detection App", page_icon=":warning:")

# Page Title
st.title("Web Mining Project - Toxic Comment Classification")

# Description and Image
st.write("""
This app used to classify text as toxic or non-toxic. 
It's a part of our Web Mining project focused on toxic comment classification.
""")

image = "img.png"
st.image(image, caption="Toxic Comment Classification", use_column_width=True)

st.subheader("Purpose:")
st.markdown("""
- **Identifies harmful comments**: The application recognizes and flags comments that contain harmful or offensive language.
- **Helps maintain online safety**: By detecting toxic comments, the app contributes to creating safer online environments by minimizing the spread of negativity and abuse.
- **Enhances user experience**: Users can interact with online platforms more comfortably knowing that harmful comments are being filtered out.
- **Supports content moderation**: Content moderators can use the app to efficiently review and address potentially harmful comments, maintaining the integrity of online communities.
- **Fosters positive interactions**: By reducing the visibility of toxic comments, the app encourages constructive discussions and promotes a more positive online atmosphere.
""")

# Text Input
st.header("Toxic Comment Classification")
text_input = st.text_input("Enter your comment")

# Analyse Button
if text_input is not None:
    if st.button("Analyse"):
        # st.balloons()
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info(f"The text is {result}.")

