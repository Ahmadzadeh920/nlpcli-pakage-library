import os
import click
from nlpcli.cli import main
import pytest
from click.testing import CliRunner
import tempfile


# Fixture to create a temporary input file with sentences
@pytest.fixture
def input_file(tmp_path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("This is the first sentence.\nThis is the second sentence.")

    return str(file_path)

def test_main_loads_sentences(input_file):
    runner = CliRunner()
    result = runner.invoke(main, [input_file])
    assert result.exit_code == 0
    assert "Loaded 2 sentences from" in result.output

def test_remove_stop_words(input_file, monkeypatch):
    runner = CliRunner()

     # Auto-answer prompts: first "no"
    monkeypatch.setattr("click.prompt" ,lambda *args , **kwargs: "no")
    result = runner.invoke(main , [input_file , "remove_stop_words"])

    assert result.exit_code ==0
    assert "Stopwords removed" in result.output
    assert "Cleaned sentences not saved." in result.output


def test_normalize_sentences(input_file, monkeypatch):
    runner = CliRunner()

    monkeypatch.setattr("click.prompt" ,lambda *args , **kwargs: "no")

    result = runner.invoke(main , [input_file , "normalize"] )

    assert result.exit_code == 0
    assert "Normalization completed" in result.output
    assert "Cleaned sentences not saved." in result.output


def test_stem_sentences(input_file, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("click.prompt", lambda *args, **kwargs: "no")

    result = runner.invoke(main, [input_file, "stem"])
    assert result.exit_code == 0
    assert "stemmed:" in result.output
    assert "not saved" in result.output.lower()


def test_sentiment_analysis(input_file, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr("click.prompt", lambda *args, **kwargs: "no")

    result = runner.invoke(main, [input_file, "sentiment"])
    assert result.exit_code == 0
    assert "Sentiment analysis completed" in result.output
    assert "Sentence:" in result.output


def test_remove_stop_words_save_file(input_file , tmp_path, monkeypatch):
    runner = CliRunner()
    responses = iter(["yes", str(tmp_path)])
    monkeypatch.setattr(click, "prompt", lambda *args, **kwargs: next(responses))
    result = runner.invoke(main, [input_file, "remove_stop_words"])
    assert result.exit_code == 0

    saved_file = tmp_path / "cleaned_sentences.txt"
    assert saved_file.exists()
    assert "Cleaned sentences saved" in result.output