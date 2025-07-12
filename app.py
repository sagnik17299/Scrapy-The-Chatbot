from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import re
import spacy
from transformers import pipeline

app = Flask(__name__)

# Load models once globally
nlp = spacy.load('en_core_web_sm')
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        raw_text = ' '.join(p.get_text() for p in paragraphs)
        cleaned_text = re.sub(r'\s+|[^A-Za-z0-9.,]+', ' ', raw_text).strip()
        doc = nlp(cleaned_text)
        filtered_tokens = [token.text for token in doc if not token.is_punct]
        final_text = ' '.join(filtered_tokens)
        return final_text
    except Exception as e:
        return f"Error scraping the website: {str(e)}"

def answer_question(context, question):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error answering question: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = ''
    url = ''
    question = ''
    if request.method == 'POST':
        url = request.form.get('url')
        question = request.form.get('question')
        if url and question:
            context = scrape_website(url)
            if context.startswith("Error"):
                answer = context
            else:
                answer = answer_question(context, question)
        else:
            answer = "Please provide both URL and question."
    return render_template('index.html', answer=answer, url=url, question=question)

if __name__ == "__main__":
    app.run(debug=True)
