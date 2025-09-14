def download_resources():
    """
    Download required TextBlob corpora and spaCy models.
    Call this explicitly before using NLP features.
    """
    print("Downloading TextBlob corpora...")
    try:
        from textblob import download_corpora
        download_corpora.download_all()
    except Exception as e:
        print("Failed to download TextBlob corpora:", e)

    print("Downloading spaCy English model...")
    try:
        from spacy.cli import download
        download("en_core_web_sm")
    except Exception as e:
        print("Failed to download spaCy model:", e)

    print("All resources downloaded successfully.")

    