from click.testing import CliRunner
from nlp.nlpcli import main, nlp


def test_tokenize():
    runner = CliRunner()
    result = runner.invoke(main, ['sample.txt', 'tokenize'])
    assert result.exit_code == 0
    assert "Total tokens: " in result.output
   
   


def test_sentiment():
    runner = CliRunner()
    result = runner.invoke(main, ['sample.txt', 'sentiment'])
    assert result.exit_code == 0
    assert "Sentiment:" in result.output
    assert "Average Subjectivity: " in result.output
    assert "Average Polarity: " in result.output
    

def test_entity():
    if nlp is None:
        return  # Skip test if spaCy is not available

    runner = CliRunner()
    result = runner.invoke(main, ['sample.txt','entity'])
    assert result.exit_code == 0
    assert "ENTITY" in result.output or "No entities found." in result.output
    