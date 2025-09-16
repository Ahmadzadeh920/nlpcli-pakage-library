import os
import click
from textblob import TextBlob
import spacy
import pyfiglet
import click_config_file
import click_params as cp
import ast  # to safely parse the list from the file
from click_help_colors import HelpColorsGroup , HelpColorsCommand
import pandas as pd
from .services import (read_input_file,
                        remove_stopwords_from_sentence, 
                        normalized_sentences,
                        stem_sentences,
                        get_sentiment,
                        )




DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.ini")




# -----------------------------
# CLI Main Function
# -----------------------------

@click.version_option(version='1.0.0', prog_name='NLP CLI')
@click.group(invoke_without_command=True,cls = HelpColorsGroup , help_headers_color='yellow', help_options_color='green')
@click.argument('input_file',
    type=click.File('r'),  
    required=True) 
@click.pass_context
@click_config_file.configuration_option(default = DEFAULT_CONFIG)
def main(ctx, input_file):
   
    """
    Read comments from INPUT_FILE (.csv or .txt).
    Store all sentences in ctx.obj["sentences"] 
    and filename in ctx.obj["file_name"].
    """
    ctx.ensure_object(dict)
    sentences = read_input_file(input_file)
    if not sentences:
        raise click.ClickException("No valid sentences found in the input file.")
    ctx.obj["sentences"] = sentences
    ctx.obj["file_name"] = input_file.name
    click.secho(f"Loaded {len(sentences)} sentences from {input_file.name}", fg="cyan")


# -----------------------------
# Subcommand: Remove stopwords
# -----------------------------

@main.command("remove_stop_words", help_headers_color='blue', help_options_color='green')
@click.pass_context
def remove_stop_words(ctx): 
    """
    Remove stopwords from the list of sentences using spaCy or TextBlob.
    """
    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.secho("No sentences found to process.", fg="yellow")
        return

    cleaned = remove_stopwords_from_sentence(sentences)
    click.secho("Stopwords removed. Example output:" , fg="green")
    click.echo(f"Original: {sentences[0:2]}")
    click.echo(f"Cleaned : {cleaned[0:2]}")

     # Prompt to save cleaned sentences

    save =click.prompt("Do you want to save all cleaned sentences? (yes/no)" , type=str, default="no") 
    if save.lower() in ("yes", "y"):
        path = click.prompt("Enter output file path", type=str,)
        if not os.path.isdir(path):
            raise click.ClickException("The specified path is not a valid directory.")
            

        elif os.path.isfile(path):
            raise click.ClickException("The specified path is a file. Please provide a directory path.")

        else:
            with open(os.path.join(path, "cleaned_sentences.txt"), 'w', encoding="utf-8") as f:
                for sent in cleaned:
                    f.write(sent + "\n")
            click.secho(f"Cleaned sentences saved to {os.path.join(path, 'cleaned_sentences.txt')}", fg="green")
    else:
        click.secho("Cleaned sentences not saved.", fg="yellow")          



# -----------------------------
# Subcommand: Normalize sentences
# -----------------------------

@main.command("normalize" ,  help_headers_color='blue', help_options_color='green')
@click.pass_context
def normalize(ctx):
    """
    Normalize loaded sentences using spaCy and TextBlob
    """
    
    if "sentences" not in ctx.obj:
        click.echo("No sentences loaded. Please provide an input file first.")
        return

    sentences = ctx.obj["sentences"]
 
    cleaned_sentences = normalized_sentences(sentences)
    click.echo("Normalization completed. Example output:")
    click.echo(f"Original: {sentences[0]}")
    click.echo(f"Normalized: {cleaned_sentences[0]}")
    # Prompt to save normalized sentences

    save = click.prompt("Do you want to save all normalized sentences? (yes/no)", type=str, default="no")

    if save.lower() in ("yes", "y"):
        path = click.prompt("Enter output file path", type=str,)
        if not os.path.isdir(path):
            raise click.ClickException("The specified path is not a valid directory.")
            
        elif os.path.isfile(path):
            raise click.ClickException("The specified path is a file. Please provide a directory path.")

    
        else:
            with open(os.path.join(path, "normalized_sentences.txt"), 'w', encoding="utf-8") as f: 
                for sent in cleaned_sentences:
                    f.write(sent + "\n")
            click.secho(f"normalized sentences saved to {os.path.join(path, 'normalized_sentences.txt')}", fg="green")
    else:
        click.secho("Cleaned sentences not saved.", fg="yellow")          


# -----------------------------
# Subcommand: Stem sentences
# -----------------------------

@main.command("stem" ,  help_headers_color='blue', help_options_color='green')
@click.pass_context
def stem(ctx):
    """
    Stem loaded sentences using TextBlob and spaCy
    """
    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.echo("No sentences loaded. Please provide an input file first.")
        return
 
    stem_sentence = stem_sentences(sentences)
    click.echo("Normalization completed. Example output:")
    click.echo(f"Original: {sentences[0]}")
    click.echo(f"stemmed: {stem_sentence[0]}")
    # Prompt to save stem_of  sentences

    save = click.prompt("Do you want to save all stem of sentences? (yes/no)", type=str, default="no")

    if save.lower() in ("yes", "y"):
        path = click.prompt("Enter output file path", type=str,)
        if not os.path.isdir(path):
            raise click.ClickException("The specified path is not a valid directory.")
            
        elif os.path.isfile(path):
            raise click.ClickException("The specified path is a file. Please provide a directory path.")

    
        else:
            with open(os.path.join(path, "stemmed_sentences.txt"), 'w', encoding="utf-8") as f: 
                for sent in stem_sentence:
                    f.write(sent + "\n")
            click.secho(f"stemmed sentences saved to {os.path.join(path, 'normalized_sentences.txt')}", fg="green")
    else:
        click.secho("The stem of sentences not saved.", fg="yellow")          




@main.command('sentiment',  help_headers_color='blue', help_options_color='green')
@click.pass_context
def sentiments(ctx):
    """
    Compute sentiment of each sentence using TextBlob
    """
    
    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.echo("No sentences loaded. Please provide an input file first.")
        return

    sentiments = get_sentiment(sentences)
    click.echo("Sentiment analysis completed. Example output:")
    click.echo(f"Sentence: {sentiments[0]['sentence']}")
    click.echo(f"Polarity: {sentiments[0]['polarity']:.3f}, Subjectivity: {sentiments[0]['subjectivity']:.3f}")
    # Prompt to save ssetiments

    save = click.prompt("Do you want to save all sentiment results? (yes/no)", type=str, default="no")

    if save.lower() in ("yes", "y"):
        path = click.prompt("Enter output file path", type=str,)
        if not os.path.isdir(path):
            raise click.ClickException("The specified path is not a valid directory.")
            
        elif os.path.isfile(path):
            raise click.ClickException("The specified path is a file. Please provide a directory path.")

    
        else:
            with open(os.path.join(path, "sentiments_sentences.txt"), 'w', encoding="utf-8") as f: 
                f.write("sentence |polarity|Subjectivity "+ "\n")
                for sent in sentiments:
                    f.write(sent["sentence"] +"|"+str(sent["polarity"])+"|"+str(sent["subjectivity"])+"\n")
            click.secho(f"sentiments of sentences saved to {os.path.join(path, 'sentiments_sentences.txt')}", fg="green")
    else:
        click.secho("The sentiments of sentences not saved.", fg="yellow")          






if __name__ =="__main__":
    main(obj={})