from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)

# Load the model and tokenizer
model_path = './NLP_Model/model'

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set the model to evaluation mode

def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        sentiment = 'Positive' if predicted_class.item() == 1 else 'Negative'
        confidence_score = confidence.item()
    return sentiment, confidence_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['reviewText']
        sentiment, confidence = predict_sentiment(review_text)
        return render_template('result.html', review=review_text, sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)