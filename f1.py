# import numpy as np
# import sys
#
# import tensorflow as tf
#
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QCheckBox, QMainWindow, QTextEdit, QSpacerItem, QSizePolicy, QComboBox, QStyledItemDelegate
# from PyQt5.QtGui import QPixmap, QPainter, QPalette, QBrush
# from PyQt5.QtCore import Qt
# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# import pickle
# import joblib
#
# model = load_model('rnn_lstm_final.h5')
# loaded_model = joblib.load("my_rnn_model.joblib")
#
# # Load tokenizer and padded sequences
# with open("tokenizer_and_sequences.pkl", "rb") as f:
#     tokenizer, data = pickle.load(f)
#
# class CustomComboBoxDelegate(QStyledItemDelegate):
#     def sizeHint(self, option, index):
#         size_hint = super().sizeHint(option, index)
#         size_hint.setHeight(80)  # Set the height to 40 pixels, adjust as necessary
#         return size_hint
#
# delegate = CustomComboBoxDelegate()
#
# class BackgroundWidget(QWidget):
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.background_image = QPixmap("sentiback3.jpg")
#
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.drawPixmap(self.rect(), self.background_image)
#
# class App(QMainWindow):
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle("Indic Sentence Summarization & Sentiment Analysis")
#         self.setGeometry(300, 300, 1000, 1000)
#         heading_label1 = QLabel("<font color='white' face='Times New Roman'>Insightful Echoes: </font>")
#         font = heading_label1.font()
#         font.setPointSize(32)
#         heading_label1.setFont(font)
#         heading_label2 = QLabel("<font color='white' face='Times New Roman'> "
#                                 "Crafting Summaries with Sentiments(for ਪੰਜਾਬੀ Text)</font>")
#         font = heading_label2.font()
#         font.setPointSize(32)
#         heading_label2.setFont(font)
#         central_widget = BackgroundWidget(self)
#         self.setCentralWidget(central_widget)
#
#         layout = QVBoxLayout(central_widget)
#         layout.setAlignment(Qt.AlignCenter)
#         layout.addWidget(heading_label1)
#         layout.addWidget(heading_label2)
#
#         spacer = QSpacerItem(40, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
#         layout.addItem(spacer)
#
#         self.model_combobox = QComboBox()
#         self.model_combobox.addItems(["Indic-Bert", "RNN"])
#         self.model_combobox.setGeometry(150, 150, 150, 80)
#         self.model_combobox.setStyleSheet("QComboBox { color: white; background-color: violet; min-height: 30px; border: 5px solid black; border-width: 6px 5px 1px 1px;}")
#         self.model_combobox.setMinimumWidth(10)
#         self.model_combobox.setItemDelegate(delegate)
#         layout.addWidget(self.model_combobox)
#         self.selected_model = None
#
#         self.model_combobox.currentIndexChanged.connect(self.handle_model_selection)
#
#         sentiment_layout = QVBoxLayout()
#         sentiment_label = QLabel("<font color='white' size='8'>Enter some text here:</font>")
#         self.user_input = QLineEdit()
#         self.user_input.setStyleSheet("background-color: rgba(200, 255, 255, 0.2); color: white;")
#         self.user_input.setFixedSize(1000, 100)
#
#         self.sentiment_output = QLabel()
#         self.sentiment_output.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); color: white size='8';")
#         self.sentiment_output.setFixedSize(1000, 100)
#
#         self.output_text = QTextEdit()
#         self.output_text.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); color: white size='8';")
#         self.output_text.setFixedSize(1000, 200)
#
#         self.summarize_before_sentiment_checkbox = QCheckBox("Summarize before analyzing sentiment")
#         self.summarize_before_sentiment_checkbox.setStyleSheet("QCheckBox::indicator { background-color: white; }")
#         self.summarize_before_sentiment_checkbox.setStyleSheet("QCheckBox { color: violet; }")
#
#         analyze_button = QPushButton("Analyze Sentiment")
#         analyze_button.clicked.connect(self.analyze_sentiment)
#         analyze_button.setStyleSheet("background-color: violet;")
#         analyze_button.setFixedSize(200, 40)
#         sentiment_layout.addWidget(sentiment_label)
#         sentiment_layout.addWidget(self.user_input)
#         sentiment_layout.addWidget(self.summarize_before_sentiment_checkbox)
#         sentiment_layout.addWidget(analyze_button)
#         sentiment_layout.addWidget(self.sentiment_output)
#         layout.addLayout(sentiment_layout)
#
#         spacer = QSpacerItem(40, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
#         layout.addItem(spacer)
#
#         self.model1 = AutoModelForSequenceClassification.from_pretrained('punjabiSentimentAnalysis')
#         self.tokenizer1 = AutoTokenizer.from_pretrained('punjabiSentimentAnalysis')
#         self.model2 = load_model('rnn_lstm_final.h5')
#         with open("tokenizer_and_sequences.pkl", "rb") as f:
#             tokenizer, data = pickle.load(f)
#         self.tokenizer2 = tokenizer
#
#         self.model_summ = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS")
#         self.tokenizer_summ = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS",
#                                                             do_lower_case=False, use_fast=False, keep_accents=True)
#         self.bos_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("<s>")
#         self.eos_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("</s>")
#         self.pad_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("<pad>")
#
#     def handle_model_selection(self, index):
#         if index == 0:
#             self.selected_model = (self.model1, self.tokenizer1)
#         elif index == 1:
#             self.selected_model = (self.model2, self.tokenizer2)
#
#     def predict_sentiment(self, text, model, tokenizer):
#         inputs = tokenizer(text, return_tensors="pt")
#         outputs = model(**inputs)
#         predicted_class = torch.argmax(outputs.logits, dim=-1).item()
#         if predicted_class == 0:
#             return "Negative"
#         else:
#             return "Positive"
#
#     def summarize(self, text):
#         input_ids = self.tokenizer_summ(f"{text} </s> <2pa>", add_special_tokens=False, return_tensors="pt",
#                                         padding=True).input_ids
#         model_output = self.model_summ.generate(input_ids, use_cache=True, no_repeat_ngram_size=3, num_beams=5,
#                                                 length_penalty=0.8, max_length=20, min_length=1, early_stopping=True,
#                                                 pad_token_id=self.pad_id, bos_token_id=self.bos_id,
#                                                 eos_token_id=self.eos_id,
#                                                 decoder_start_token_id=self.tokenizer_summ._convert_token_to_id_with_added_voc(
#                                                     "<2pa>"))
#         decoded_output = self.tokenizer_summ.decode(model_output[0], skip_special_tokens=True,
#                                                     clean_up_tokenization_spaces=False)
#         return decoded_output
#
#     def process_input(self):
#         input_text = self.user_input.text()
#         a = [input_text]
#         a = self.tokenizer2.texts_to_sequences(a)
#         a = np.array(a)
#         a = pad_sequences(a, padding='post', maxlen=100)
#         a = a.reshape((a.shape[0], a.shape[1], 1))
#         prediction = self.model2.predict(np.array(a))
#         for row in prediction:
#             element1 = row[0]
#             element2 = row[1]
#             if element1 > element2:
#                 self.sentiment_output.setText("Negative")
#             else:
#                 self.sentiment_output.setText("Positive")

import numpy as np
import sys

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, \
    QCheckBox, QMainWindow, QTextEdit, QSpacerItem, QSizePolicy, QComboBox, QStyledItemDelegate
from PyQt5.QtGui import QPixmap, QPainter, QPalette, QBrush
from PyQt5.QtCore import Qt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pickle
import joblib

model = load_model('rnn_lstm_final.h5')
loaded_model = joblib.load("my_rnn_model.joblib")

# Load tokenizer and padded sequences
with open("tokenizer_and_sequences.pkl", "rb") as f:
    tokenizer, data = pickle.load(f)


class CustomComboBoxDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size_hint = super().sizeHint(option, index)
        size_hint.setHeight(40)  # Set the height to 40 pixels, adjust as necessary
        return size_hint


delegate = CustomComboBoxDelegate()


class BackgroundWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_image = QPixmap("sentiback3.jpg")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background_image)


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Indic Sentence Summarization & Sentiment Analysis")
        self.setGeometry(100, 100, 600, 600)
        heading_label1 = QLabel("<font color='white' face='Elephant'>Insightful Echoes: </font>")
        font = heading_label1.font()
        font.setPointSize(32)

        heading_label1.setFont(font)
        heading_label2 = QLabel("<font color='white' face='Elephant'> "
                                "Crafting Summaries with Sentiments for Punjabi(ਪੰਜਾਬੀ) Text</font>")
        font = heading_label2.font()
        font.setPointSize(32)
        heading_label2.setFont(font)
        central_widget = BackgroundWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        spacer = QSpacerItem(40, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        layout.addWidget(heading_label1)
        layout.addWidget(heading_label2)

        spacer = QSpacerItem(40, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)
        # Create a QLabel for the heading
        heading_label4 = QLabel("<font color='white' size='8'>Select the Model Here:</font>", self)

        self.model_combobox = QComboBox(self)
        self.model_combobox.setFixedSize(400, 200)
        self.model_combobox.setGeometry(200, 150, 30, 30)
        self.model_combobox.addItems(["Indic-Bert", "RNN"])

        self.model_combobox.setStyleSheet(
            "QComboBox { color: black; background-color: white; min-height: 40px; min-width: 50px; border: 5px solid black; border-width: 6px 5px 1px 1px;}")
        self.model_combobox.setMinimumWidth(20)
        self.model_combobox.setItemDelegate(delegate)
        layout.addWidget(heading_label4)
        layout.addWidget(self.model_combobox)
        self.selected_model = None

        self.model_combobox.currentIndexChanged.connect(self.handle_model_selection)

        sentiment_layout = QVBoxLayout()
        sentiment_label = QLabel("<font color='white' size='8'>Enter some text here:</font>")
        self.user_input = QLineEdit()
        self.user_input.setStyleSheet("background-color: white ; color: black;")
        self.user_input.setFixedSize(1000, 100)

        self.sentiment_output = QTextEdit()
        self.sentiment_output.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); color: white size='8';")
        self.sentiment_output.setFixedSize(1000, 100)

        self.output_text = QTextEdit()
        self.output_text.setStyleSheet("background-color: rgba(255, 255, 255, 0.2); color: white size='8';")
        self.output_text.setFixedSize(1000, 200)

        self.summarize_before_sentiment_checkbox = QCheckBox("Summarize before analyzing sentiment")
        self.summarize_before_sentiment_checkbox.setStyleSheet("QCheckBox::indicator { background-color: white; }")
        self.summarize_before_sentiment_checkbox.setStyleSheet("QCheckBox { color: violet; font-size: 16px }")
        # self.summarize_before_sentiment_checkbox.setFixedSize(200, 400)

        analyze_button = QPushButton("Analyze Sentiment")
        analyze_button.clicked.connect(self.analyze_sentiment)
        analyze_button.setStyleSheet("background-color: violet;")
        analyze_button.setFixedSize(200, 40)
        sentiment_layout.addWidget(sentiment_label)
        sentiment_layout.addWidget(self.user_input)
        sentiment_layout.addWidget(self.summarize_before_sentiment_checkbox)
        sentiment_layout.addWidget(analyze_button)

        sentiment_layout.addWidget(self.sentiment_output)
        layout.addLayout(sentiment_layout)

        spacer = QSpacerItem(40, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        self.model1 = AutoModelForSequenceClassification.from_pretrained('punjabiSentimentAnalysis')
        self.tokenizer1 = AutoTokenizer.from_pretrained('punjabiSentimentAnalysis')
        self.model2 = load_model('rnn_lstm_final.h5')
        with open("tokenizer_and_sequences.pkl", "rb") as f:
            tokenizer, data = pickle.load(f)
        self.tokenizer2 = tokenizer

        self.model_summ = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS")
        self.tokenizer_summ = AutoTokenizer.from_pretrained("ai4bharat/MultiIndicSentenceSummarizationSS",
                                                            do_lower_case=False, use_fast=False, keep_accents=True)
        self.bos_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("<s>")
        self.eos_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("</s>")
        self.pad_id = self.tokenizer_summ._convert_token_to_id_with_added_voc("<pad>")

    def handle_model_selection(self, index):
        if index == 0:
            self.selected_model = (self.model1, self.tokenizer1)
        elif index == 1:
            self.selected_model = (self.model2, self.tokenizer2)
    def is_valid_punjabi_text(self, text):
        english_alphabet = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        numbers = set("0123456789")
        punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

        for char in text:
            if char in english_alphabet or char in numbers or char in punctuation:
                return False
        return True
    def predict_sentiment(self, text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=-1).item()
        if predicted_class == 0:
            return "Negative"
        else:
            return "Positive"

    def summarize(self, text):
        input_ids = self.tokenizer_summ(f"{text} </s> <2pa>", add_special_tokens=False, return_tensors="pt",
                                        padding=True).input_ids
        model_output = self.model_summ.generate(input_ids, use_cache=True, no_repeat_ngram_size=3, num_beams=5,
                                                length_penalty=0.8, max_length=20, min_length=1,
                                                early_stopping=True,
                                                pad_token_id=self.pad_id, bos_token_id=self.bos_id,
                                                eos_token_id=self.eos_id,
                                                decoder_start_token_id=self.tokenizer_summ._convert_token_to_id_with_added_voc(
                                                    "<2pa>"))
        decoded_output = self.tokenizer_summ.decode(model_output[0], skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
        return decoded_output

    def process_input(self):
        input_text = self.user_input.text()
        a = [input_text]
        a = self.tokenizer2.texts_to_sequences(a)
        a = np.array(a)
        a = pad_sequences(a, padding='post', maxlen=100)
        a = a.reshape((a.shape[0], a.shape[1], 1))
        prediction = self.model2.predict(np.array(a))
        for row in prediction:
            element1 = row[0]
            element2 = row[1]
            if element1 > element2:
                self.sentiment_output.setText("Negative")
            else:
                self.sentiment_output.setText("Positive")

    def analyze_sentiment(self):
        user_input = self.user_input.text()

        # Validate the input text
        # Validate the input text
        if not self.is_valid_punjabi_text(user_input):
            QMessageBox.warning(self, "Invalid Input", "Please enter valid Punjabi text.")
            return

        summarize_before_sentiment = self.summarize_before_sentiment_checkbox.isChecked()

        if summarize_before_sentiment:
            summarized_text = self.summarize(user_input)
            sentiment_bert = self.predict_sentiment(summarized_text, self.model1, self.tokenizer1)
            sentiment_output = f'Sentiment (Indic-BERT): {sentiment_bert}\nSummary: {summarized_text}'
        else:
            sentiment_bert = self.predict_sentiment(user_input, self.model1, self.tokenizer1)
            sentiment_output = f'Sentiment (Indic-BERT): {sentiment_bert}'

        if self.model_combobox.currentIndex() == 1:
            self.process_input()  # This sets the sentiment_output for the LSTM model
            sentiment_output += f"\nSentiment (Bidirectional LSTM): {self.sentiment_output.toPlainText()}"

            if summarize_before_sentiment:
                summarized_text_lstm = self.summarize(user_input)
                sentiment_lstm = self.sentiment_output.toPlainText()
                sentiment_output += f"\nSummary (Bidirectional LSTM): {summarized_text_lstm}\nSentiment (Bidirectional LSTM): {sentiment_lstm}"

        self.sentiment_output.setText(sentiment_output)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())
