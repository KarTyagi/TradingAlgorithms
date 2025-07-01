import re
from symbols import SYMBOLS
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def initialize_analyzer():
    try:
        analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        return analyzer
    except Exception as e:
        print(f"Error initializing model: {e}")
        raise

def initialize_absa():
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    absa_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return absa_pipeline

def sentiment_score(label, score):
    if label == "NEGATIVE":
        return round(-1.0 * score ** 3, 3)
    elif label == "POSITIVE":
        return round(score ** 3, 3) 
    return 0.000

def analyze_sentiment(analyzer, text):
    try:
        result = analyzer(text)
        return result[0]
    except Exception as e:
        return {"label": "ERROR", "score": 0.0, "error": str(e)}

def analyze_aspect_sentiment(absa_pipeline, text, aspect):
    input_text = f"{text} [SEP] {aspect}"
    result = absa_pipeline(input_text)
    label = result[0]['label'].lower()
    score = result[0]['score']
    if label == "negative":
        mapped_score = -score
    elif label == "positive":
        mapped_score = score
    else:
        mapped_score = 0.0
    return label, score, mapped_score
