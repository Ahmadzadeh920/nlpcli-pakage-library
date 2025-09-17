# nlpcli

[![PyPI version](https://img.shields.io/pypi/v/nlpcli.svg)](https://pypi.org/project/nlpcli/)
[![Python Version](https://img.shields.io/pypi/pyversions/nlpcli.svg)](https://pypi.org/project/nlpcli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/Ahmadzadeh920/nlpcli-pakage-library/tests.yml?branch=main)](https://github.com/Ahmadzadeh920/nlpcli-pakage-library/actions)

**An open-source Python CLI tool for NLP tasks**

`nlpcli` is a command-line interface library that simplifies natural language processing (NLP) tasks such as removing stopwords, normalizing text, stemming, and sentiment analysis.  
It is built with [Click](https://click.palletsprojects.com/) and provides an interactive, colorful CLI with configuration support.

---

## ✨ Features

- 📝 **Stopword Removal** — Remove stopwords using spaCy or TextBlob.  
- 🔄 **Text Normalization** — Clean and normalize sentences.  
- 🌱 **Stemming** — Perform stemming on sentences.  
- 😊 **Sentiment Analysis** — Get polarity & subjectivity for each sentence.  
- ⚙️ **Configurable** — Supports `config.ini` for default settings.  
- 🎨 **Interactive CLI** — Colorful, user-friendly prompts.  

---

## 📦 Installation

Install from PyPI (after publishing):

```bash
pip install nlpcli
```
Then, download the required NLP resources (must be done once before first use):
``` bash
nlpcli-download
```
Or install from source:

```bash
git clone https://github.com/Ahmadzadeh920/nlpcli-pakage-library.git
cd nlpcli-pakage-library
pip install .
```
---
## ⚡ Usage

The CLI tool is available as nlpcli once installed.


Note: Before using nlpcli for the first time, you must download the necessary NLP models and resources. Run:

```bash
  nlpcli-download

```

Run it with an input file (.csv or .txt):
```bash
nlpcli input.txt
```

---
## 🛠 Commands
### 1. Remove Stopwords
```bash
nlpcli input.txt remove_stop_words
```
Removes stopwords and shows an example. Optionally save results.

### 2. Normalize Sentences

```bash
nlpcli input.txt normalize
```
Normalizes text and saves to normalized_sentences.txt if desired.

### 3. Stem Sentences
```bash
nlpcli input.txt stem
```
Applies stemming. Example and optional file export included.

### 4. Sentiment Analysis
```bash
nlpcli input.txt sentiment
```
Computes sentiment for each sentence:

```bash
Sentence: I love open-source projects.
Polarity: 0.500, Subjectivity: 0.600
```
Exports results to sentiments_sentences.txt if selected.

---

## 📂 Project Structure
```bash

nlpcli/
├── src/nlpcli/
│   ├── cli.py          # CLI entry point
│   ├── services.py     # Core NLP functions
│   ├── config.ini      # Default configuration
│   └── __init__.py
├── tests/              # Unit tests
├── README.md           # Documentation
├── pyproject.toml      # Project metadata
└── poetry.lock

```
---

## ⚙️ Configuration

Default options can be stored in config.ini, automatically loaded by the CLI.

Example:
```bash
[settings]
language = en
output_format = txt
```

---

## 🧪 Development

Clone and install dependencies:

```bash

poetry install

```

Run tests:

```bash
poetry run pytest
```

---
## 🚀 Roadmap

Planned features and improvements:

- [ ] **Lemmatization** — Add support for lemmatizing words.  
- [ ] **Language Detection** — Automatically detect the language of input text.  
- [ ] **Additional Input Formats** — Support JSON, Excel, and other file types.  
- [ ] **Enhanced Visualization** — Use `rich` for better CLI output formatting.  


---
## 📜 License
This project is licensed under the MIT License

----

## 👩‍💻 Author

**Fatemeh Ahmadzadeh**  
- 📧 Email: [ahmadzade920@gmail.com](mailto:ahmadzade920@gmail.com)  
- 🌐 GitHub: [@Ahmadzadeh920](ahmadzadeh920.github.io)  
- 💼 LinkedIn: [Fatemeh Ahmadzadeh](https://www.linkedin.com/in/f-ahmadz/)  











