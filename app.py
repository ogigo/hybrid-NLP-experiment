from flask import Flask, render_template, request
from transformers import pipeline
from sentiment import analyze_sentiment
from summarize import generate_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.form['text']
    result = analyze_sentiment(text)
    return render_template('result.html', result=result)

@app.route('/summarize', methods=['POST'])
def text_summarization():
    text = request.form['text']

    try:
        summary = generate_summary(text, max_length=150, min_length=50)
        if isinstance(summary, list):
            result = summary[0]['summary_text']
        else:
            result = summary
    except Exception as e:
        result = f"Error during summarization: {str(e)}"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
