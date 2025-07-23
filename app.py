from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

app = Flask(_name_)
max_length = 200

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction >= 0.5 else "Negative"

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        review = request.form["review"]
        sentiment = predict_sentiment(review)
    return render_template("index.html", sentiment=sentiment)

if _name_ == "_main_":
    app.run(debug=True)