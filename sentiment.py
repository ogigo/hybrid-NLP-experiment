import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5ForSequenceClassification, AdamW

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForSequenceClassification.from_pretrained(model_name)

sentiment_labels = ["negative", "positive"]

def analyze_sentiment(text):
    inputs = tokenizer("classify: " + text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    sentiment_label = sentiment_labels[predicted_label]
    confidence_score = probabilities[0, predicted_label].item()
    return sentiment_label, confidence_score

