import os
from textblob import TextBlob , Word
import spacy
import pandas as pd
import string
import click

# Try to load spaCy's small English model once, at import time.
# We'll gracefully warn the user if it's missing at runtime.
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None




# -----------------------------
# Helper: Read input file
# -----------------------------
def read_input_file(input_file):
    """
    Reads the input"""
    file_name = input_file.name 
    if file_name.endswith(".txt"):
        with open(file_name, 'r' , encoding = "utf-8") as f:
            sentense  = [line.strip() for line in f if line.strip()]
    
    elif file_name.endwith('.csv'):
        df = pd.read_csv(file_name)
        sentense = df["comments"].dropna().tolist()
    else:
        raise click.ClickException("Unsupported file format. Please provide a .txt or .csv file.")
    return sentense


# -----------------------------
# Helper: Remove stopwords
# -----------------------------

def remove_stopwords_from_sentence(sentences):
    """
    Remove stopwords from a list of sentences using spaCy or TextBlob.
    """
    if nlp is None:
        click.secho("spaCy model not found. Falling back to TextBlob for stopword removal.", fg="yellow")
        return 
    cleaned_sentences = []
    
    for sentence in sentences:
        doc = nlp(sentence) if nlp else TextBlob(sentence)
        tokens= [token.text for token in doc if token.text.lower() not in nlp.Defaults.stop_words and  token.text not in string.punctuation] 
        cleaned_sentences.append(" ".join(tokens))
    
    return cleaned_sentences

  

# -----------------------------
# Helper: Normalize sentences
# -----------------------------

def normalized_sentences(sentences):
    normalized = []
    if nlp is None:
        raise click.ClickException("spaCy model not found. Please run 'nlpcli setup' to download required models.")
    for sentence in sentences:
        sentence = sentence.lower()
        sentence_clean =  sentence.translate(str.maketrans("" , "" , string.punctuation))

        doc = nlp(sentence_clean)
        lemmenize_token = [token.lemma_ for token in doc if token.is_alpha]
        normalized_sentence = " ".join(lemmenize_token)
        
        # Optionally, you can also use TextBlob for additional correction
        tb_sentence = str(TextBlob(normalized_sentence).correct())
        
        normalized.append(tb_sentence)
    return normalized
    


# -----------------------------
# Helper: Stem sentences
# -----------------------------

def stem_sentences(sentences):
    stemmed_sentences = []
    for sentence in sentences:
        sentence_clean = sentence.lower().translate(str.maketrans("", "", string.punctuation))
        # Tokenize with spaCy
        doc = nlp(sentence_clean)
        stemmed_tokens = [Word(token.text).stem() for token in doc if token.is_alpha]
        stemmed_sentence = " ".join(stemmed_tokens)
        stemmed_sentences.append(stemmed_sentence)

    return stemmed_sentences


# -----------------------------
# Helper: Get sentiment
# -----------------------------

def get_sentiment(sentences):
    results = []
    for sentence in sentences:
        doc = nlp(sentence)
        text =" ".join([token.text for token in doc])
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        results.append({
            "sentence": sentence,
            "polarity": polarity,
            "subjectivity": subjectivity
        })

    return results








