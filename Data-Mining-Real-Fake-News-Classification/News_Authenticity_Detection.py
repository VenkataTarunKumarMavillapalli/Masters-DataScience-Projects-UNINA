import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from transformers import BertTokenizer, BertForSequenceClassification

# Download VADER lexicon if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# -----------------------------
# Load TF-IDF and Word2Vec Models
# -----------------------------
@st.cache_resource
def load_tfidf_model():
    tfidf_model = tf.keras.models.load_model('/Users/rahulreddykarri/Downloads/models/tfidf_transformer_model.keras')
    with open('/Users/rahulreddykarri/Downloads/models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_model, tfidf_vectorizer

@st.cache_resource
def load_word2vec_model():
    word2vec_model = tf.keras.models.load_model('/Users/rahulreddykarri/Downloads/models/word2vec_transformer_model.keras')
    w2v_model = Word2Vec.load('/Users/rahulreddykarri/Downloads/models/word2vec_model.bin')
    return word2vec_model, w2v_model

@st.cache_resource
def load_scaler():
    with open('/Users/rahulreddykarri/Downloads/models/numerical_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

tfidf_model, tfidf_vectorizer = load_tfidf_model()
word2vec_model, w2v_model = load_word2vec_model()
scaler = load_scaler()  # In case you need it for additional numerical features

# Function to convert tokens to Word2Vec vectors
def get_doc_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# -----------------------------
# Initialize BERT Model & Sentiment Analyzer
# -----------------------------
if 'bert_models_loaded' not in st.session_state:
    with st.spinner('Loading BERT model and sentiment analyzer...'):
        st.session_state.bert_model = BertForSequenceClassification.from_pretrained(
            '/Users/rahulreddykarri/Downloads/models/BERT_model/pretrainedbert'
        )
        st.session_state.bert_tokenizer = BertTokenizer.from_pretrained(
            '/Users/rahulreddykarri/Downloads/models/BERT_model/pretrainedtokenizer'
        )
        st.session_state.bert_model.eval()
        st.session_state.sentiment_analyzer = SentimentIntensityAnalyzer()
        st.session_state.bert_models_loaded = True

# Function for BERT prediction with sentiment analysis
def predict_authenticity_and_sentiment(user_input):
    # BERT prediction
    tokenized_input = st.session_state.bert_tokenizer(
        user_input, truncation=True, padding=True, return_tensors='pt'
    )
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']
    
    with torch.no_grad():
        outputs = st.session_state.bert_model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    prediction = 'Real' if predicted_class == 1 else 'Fake'
    confidence = probabilities[0][predicted_class].item()
    
    # Sentiment analysis using VADER
    sentiment_scores = st.session_state.sentiment_analyzer.polarity_scores(user_input)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return prediction, confidence, sentiment

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    sentiment_scores = st.session_state.sentiment_analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment

# Function to analyze text metrics
def analyze_text_metrics(text):
    if not text.strip():
        return None
    
    # Basic counts
    total_chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    words = text.split()
    word_count = len(words)
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)
    paragraphs = text.split("\n\n")
    paragraph_count = sum(1 for p in paragraphs if p.strip())
    
    # Lexical analysis
    unique_words = set(word.lower() for word in words)
    unique_word_count = len(unique_words)
    
    # Calculate average word length
    total_word_length = sum(len(word) for word in words)
    avg_word_length = total_word_length / word_count if word_count > 0 else 0
    
    # Calculate average sentence length
    sentence_word_counts = [len(sentence.split()) for sentence in sentences]
    avg_sentence_length = sum(sentence_word_counts) / len(sentence_word_counts) if sentence_count > 0 else 0
    
    metrics = {
        "total_chars": total_chars,
        "chars_no_spaces": chars_no_spaces,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "unique_word_count": unique_word_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length
    }
    
    return metrics


# -----------------------------
# Streamlit App UI Setup
# -----------------------------
st.title("News Authenticity Detection")

# Text input from user
user_input = st.text_area("Enter the text of the news article:", key="news_text", height=200)

# Display text metrics if text is entered
if user_input:
    metrics = analyze_text_metrics(user_input)
    if metrics:
        st.subheader("Text Analysis")
        
        # Create three columns for the metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("Basic Counts")
            st.metric("Characters (total)", metrics["total_chars"])
            st.metric("Characters (no spaces)", metrics["chars_no_spaces"])
            st.metric("Words", metrics["word_count"])
        
        with col2:
            st.info("Document Structure")
            st.metric("Sentences", metrics["sentence_count"])
            st.metric("Paragraphs", metrics["paragraph_count"])
        
        with col3:
            st.info("Lexical Analysis")
            st.metric("Unique Words", metrics["unique_word_count"])
            st.metric("Avg. Word Length", f"{metrics['avg_word_length']:.2f}")
            st.metric("Avg. Sentence Length", f"{metrics['avg_sentence_length']:.2f}")


# Dropdown for selecting the prediction model
selected_model = st.selectbox("Select Prediction Model", 
                              ("Transformers - TFIDF", "Transformers - Word2vec", "BERT-BASE-UNCASED"))

# Side-by-side buttons: Predict and Clear
col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("Predict")
with col2:
    clear_clicked = st.button("Clear")

# Clear button functionality
if clear_clicked:
    st.session_state.news_text = ""
    st.success("Input cleared.")

# -----------------------------
# Prediction Logic
# -----------------------------
if predict_clicked:
    if not user_input.strip():
        st.error("Please enter the text of the news article.")
    else:
        if selected_model == "Transformers - TFIDF":
            with st.spinner("Analyzing with TF-IDF Transformer..."):
                st.info("Predicting authenticity and sentiment using Transformers - Word2vec model...")
                tokens = word_tokenize(user_input)
                
                tfidf_features = tfidf_vectorizer.transform([user_input])
                
                tfidf_features_dense = tfidf_features.toarray()
                
                expected_features = 5013
                current_feature_size = tfidf_features_dense.shape[1]
                if current_feature_size < expected_features:
                    
                    padding = np.zeros((tfidf_features_dense.shape[0], expected_features - current_feature_size))
                    tfidf_features_dense = np.hstack([tfidf_features_dense, padding])
                
                
                tfidf_features_dense = tfidf_features_dense.reshape(1, 1, expected_features)
                
                tfidf_prediction_score = tfidf_model.predict(tfidf_features_dense)[0][0]
                tfidf_prediction = (tfidf_prediction_score > 0.5).astype("int32")
                
                # Get confidence score
                if tfidf_prediction == 1:
                    prediction = "Real"
                    confidence = float(tfidf_prediction_score)
                else:
                    prediction = "Fake"
                    confidence = 1 - float(tfidf_prediction_score)
                
                # Get sentiment
                sentiment = analyze_sentiment(user_input)
                
                st.success("Prediction complete.")
                
                # Display results in the same format as BERT
                st.subheader("Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Sentiment", sentiment)
        
        elif selected_model == "Transformers - Word2vec":
            with st.spinner("Analyzing with Word2Vec Transformer..."):
                st.info("Predicting authenticity and sentiment using Transformers - Word2vec model...")
                tokens = word_tokenize(user_input)
                
                word2vec_features = get_doc_vector(tokens, w2v_model)
                
                
                word2vec_features = word2vec_features.reshape(1, -1)
                
                expected_feature_size = 113
                current_feature_size = word2vec_features.shape[1]
                if current_feature_size < expected_feature_size:
                    
                    padding = np.zeros((word2vec_features.shape[0], expected_feature_size - current_feature_size))
                    word2vec_features = np.hstack([word2vec_features, padding])
                    
                
                word2vec_features = word2vec_features.reshape(1, 1, expected_feature_size)
                
                
                word2vec_prediction_score = word2vec_model.predict(word2vec_features)[0][0]
                word2vec_prediction = (word2vec_prediction_score > 0.5).astype("int32")
                
                # Get confidence score
                if word2vec_prediction == 1:
                    prediction = "Real"
                    confidence = float(word2vec_prediction_score)
                else:
                    prediction = "Fake"
                    confidence = 1 - float(word2vec_prediction_score)
                
                # Get sentiment
                sentiment = analyze_sentiment(user_input)
                
                st.success("Prediction complete.")
                
                # Display results in the same format as BERT
                st.subheader("Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Sentiment", sentiment)
        
        elif selected_model == "BERT-BASE-UNCASED":
            with st.spinner("Analyzing with BERT..."):
                st.info("Predicting authenticity and sentiment using BERT model...")
                prediction, confidence, sentiment = predict_authenticity_and_sentiment(user_input)
                st.success("Prediction complete.")
                
                st.subheader("Results:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", prediction)
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Sentiment", sentiment)
