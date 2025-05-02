from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer
import logging
from summa import summarizer as textrank_summarizer
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def count_words(text):
    """Helper function to count words"""
    return len(re.findall(r'\w+', text))

def load_summarizer():
    """Initialize summarizer with error handling"""
    try:
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Test the model with a proper test sentence
        test_result = summarizer("This is a test sentence to check if the summarizer is working properly.", 
                               max_length=20, min_length=10)
        logger.info("Model loaded successfully")
        return summarizer
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

try:
    summarizer = load_summarizer()
except Exception as e:
    logger.error(f"Critical error - cannot load model: {str(e)}")
    summarizer = None

def safe_summarize(text, max_length=150, min_length=50):
    """Wrapper with error handling"""
    if not summarizer:
        return {"error": "Model not available"}, 503
        
    try:
        tokens = tokenizer.encode(text)
        if len(tokens) > 1024:
            logger.warning("Input too long, truncating...")
            text = tokenizer.decode(tokens[:1000])  # Leave room for special tokens
            
        result = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            num_beams=4,
            truncation=True
        )
        
        # Calculate statistics
        input_words = count_words(text)
        output_words = count_words(result[0]['summary_text'])
        reduction = round((1 - (output_words / input_words)) * 100) if input_words > 0 else 0
        
        return {
            "summary": result[0]['summary_text'],
            "stats": {
                "input_words": input_words,
                "output_words": output_words,
                "reduction": reduction
            }
        }, 200
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        return {"error": str(e)}, 500

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    if not summarizer:
        return jsonify({"error": "Service unavailable"}), 503
        
    text = request.json.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
        
    logger.info(f"Received text with {len(text)} characters")
    
    # First try with the main model
    result, status = safe_summarize(text)
    if status == 200:
        return jsonify(result)
        
    # Fallback to extractive summarization
    try:
        logger.warning("Using extractive fallback")
        summary = textrank_summarizer.summarize(text, ratio=0.4)
        if not summary:
            raise ValueError("Empty summary generated")
            
        # Calculate statistics for fallback
        input_words = count_words(text)
        output_words = count_words(summary)
        reduction = round((1 - (output_words / input_words)) * 100) if input_words > 0 else 0
        
        return jsonify({
            "summary": summary, 
            "fallback": True,
            "stats": {
                "input_words": input_words,
                "output_words": output_words,
                "reduction": reduction
            }
        })
        
    except Exception as e:
        logger.error(f"Extractive fallback failed: {str(e)}")
        return jsonify({"error": "All summarization methods failed"}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)