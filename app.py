import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import joblib

# Load models and tokenizers
model = load_model('rnn_lstm_final.h5')
loaded_model = joblib.load("my_rnn_model.joblib")

with open("tokenizer_and_sequences.pkl", "rb") as f:
    tokenizer, data = pickle.load(f)

model1 = AutoModelForSequenceClassification.from_pretrained('punjabiSentimentAnalysis')
tokenizer1 = AutoTokenizer.from_pretrained('punjabiSentimentAnalysis')

model_summ = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS")
tokenizer_summ = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS",
                                                do_lower_case=False, use_fast=False, keep_accents=True)
bos_id = tokenizer_summ._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer_summ._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer_summ._convert_token_to_id_with_added_voc("<pad>")

# Define helper functions
def is_valid_punjabi_text(text):
    english_alphabet = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    numbers = set("0123456789")
    punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    for char in text:
        if char in english_alphabet or char in numbers or char in punctuation:
            return False
    return True

def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    return "Negative" if predicted_class == 0 else "Positive"

def summarize(text):
    input_ids = tokenizer_summ(f"{text} </s> <2pa>", add_special_tokens=False, return_tensors="pt",
                                padding=True).input_ids
    model_output = model_summ.generate(input_ids, use_cache=True, no_repeat_ngram_size=3, num_beams=5,
                                        length_penalty=0.8, max_length=20, min_length=1, early_stopping=True,
                                        pad_token_id=pad_id, bos_token_id=bos_id, eos_token_id=eos_id,
                                        decoder_start_token_id=tokenizer_summ._convert_token_to_id_with_added_voc("<2pa>"))
    decoded_output = tokenizer_summ.decode(model_output[0], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    return decoded_output

def process_input(text):
    a = [text]
    a = tokenizer.texts_to_sequences(a)
    a = np.array(a)
    a = pad_sequences(a, padding='post', maxlen=100)
    a = a.reshape((a.shape[0], a.shape[1], 1))
    prediction = model.predict(np.array(a))
    for row in prediction:
        element1 = row[0]
        element2 = row[1]
        return "Negative" if element1 > element2 else "Positive"

# Streamlit app
st.title("Indic Sentence Summarization & Sentiment Analysis")
st.header("Insightful Echoes: Crafting Summaries with Sentiments (for ਪੰਜਾਬੀ Text)")

model_choice = st.selectbox("Select the Model", ["Indic-Bert", "RNN"])
summarize_before_sentiment = st.checkbox("Summarize before analyzing sentiment")
user_input = st.text_area("Enter some text here")

if st.button("Analyze Sentiment"):
    if not is_valid_punjabi_text(user_input):
        st.warning("Please enter valid Punjabi text.")
    else:
        sentiment_output = ""
        if summarize_before_sentiment:
            summarized_text = summarize(user_input)
            sentiment_bert = predict_sentiment(summarized_text, model1, tokenizer1)
            sentiment_output = f'Sentiment (Indic-BERT): {sentiment_bert}\nSummary: {summarized_text}'
        else:
            sentiment_bert = predict_sentiment(user_input, model1, tokenizer1)
            sentiment_output = f'Sentiment (Indic-BERT): {sentiment_bert}'

        if model_choice == "RNN":
            sentiment_rnn = process_input(user_input)
            sentiment_output += f"\nSentiment (Bidirectional LSTM): {sentiment_rnn}"

            if summarize_before_sentiment:
                summarized_text_rnn = summarize(user_input)
                sentiment_output += f"\nSummary (Bidirectional LSTM): {summarized_text_rnn}"

        st.text_area("Sentiment Output", sentiment_output, height=200)
