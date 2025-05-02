from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load summarization model (BART by default)
summarizer = pipeline("summarization", 
                     model="facebook/bart-large-cnn",
                     tokenizer="facebook/bart-large-cnn")

@app.route("/")
def home():
    return render_template("index.html")  # HTML frontend

@app.route("/summarize", methods=["POST"])
def summarize():
    if request.method == "POST":
        text = request.json.get("text", "")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        try:
            # Generate summary (adjust max_length as needed)
            summary = summarizer(text, max_length=300, min_length=100, do_sample=False)
            return jsonify({"summary": summary[0]["summary_text"]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)