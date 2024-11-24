pip install gradio transformers requests beautifulsoup4
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import gradio as gr

# Initialize the summarization pipeline with a pre-trained model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def fetch_article_text(url):
    """Fetches the text of an article from the provided URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extracting the article content based on common HTML tags used for articles.
    paragraphs = soup.find_all('p')
    article_text = " ".join([para.get_text() for para in paragraphs if para.get_text()])
    
    return article_text

def summarize_article(url):
    """Summarizes the article fetched from the provided URL."""
    article_text = fetch_article_text(url)
    
    # Check if article is long enough for summarization
    if len(article_text.split()) < 100:
        return "Article is too short to summarize."
    
    # Generate a more detailed summary
    summary = summarizer(article_text, max_length=500, min_length=150, do_sample=False)
    return summary[0]['summary_text']

# Create a Gradio interface for the summarization function
iface = gr.Interface(fn=summarize_article, 
                     inputs=gr.Textbox(label="Enter Article URL", placeholder="Enter the URL of the article here..."),
                     outputs="text", 
                     live=True, 
                     title="Article Summarizer",
                     description="Enter a URL of a news article to summarize it. The summary will be generated in a paragraph format containing more details.")

# Launch the Gradio app
iface.launch()
