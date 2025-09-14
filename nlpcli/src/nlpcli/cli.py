import os
import click
from textblob import TextBlob
import spacy
import pyfiglet
import click_config_file
import click_params as cp
import ast  # to safely parse the list from the file




DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config.ini")


# Try to load spaCy's small English model once, at import time.
# We'll gracefully warn the user if it's missing at runtime.
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None




@click.version_option(version='1.0.0', prog_name='NLP CLI')

@click.group(invoke_without_command=True)
@click.argument('input_file',
    type=click.File('r'),  
    required=True) 
@click.pass_context
@click_config_file.configuration_option(default = DEFAULT_CONFIG)
def main(ctx, input_file):
   
    """
    NLP Command Line Interface.

    Reads the input file once, stores its contents in Click's context (ctx.obj),
    and makes it available to all subcommands.
    """
    text = input_file.read()
    try:
        # Safely parse list of sentences/paragraphs
        sentences_list = ast.literal_eval(text)
        if not isinstance(sentences_list, list):
            raise ValueError("The file must contain a Python list. this list must contain comments on your post")
    except Exception as e:
        raise click.ClickException(f"Invalid file format: {e}")

    # ctx.ensure_object(dict) makes sure it exists and is a dict.
    ctx.ensure_object(dict)
    # Initialize ctx.obj as a dict and stash our text + filename
    ctx.obj["sentences"] = sentences_list
    ctx.obj["file_name"] = input_file.name
    click.secho(f"process this file {input_file.name}" , fg ="green")
    # If no subcommand was given, print help so the user sees available commands
    if ctx.invoked_subcommand is None:
        click.echo()
        click.echo(ctx.get_help())




@main.command()
@click.pass_context
@click_config_file.configuration_option(default=DEFAULT_CONFIG)
def tokenize(ctx):
    """
    Tokenize each sentence in the input list and print tokens.
    Uses spaCy when available, otherwise falls back to TextBlob.
    """
    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.secho("No sentences found to tokenize.", fg="yellow")
        return

    all_tokens = []

    for sentence in sentences:
        if nlp is not None:
            doc = nlp(sentence)
            tokens = [t.text for t in doc if not t.is_space]
        else:
            # Fallback: TextBlob's word tokenizer
            tokens = list(TextBlob(sentence).words)

        all_tokens.extend(tokens)

    click.secho(f"Total tokens: {len(all_tokens)}", fg="cyan")
    click.secho(f"Tokens: {all_tokens}")


@main.command()
@click.pass_context
def sentiment(ctx):
    """
    Analyze overall sentiment of a list of sentences with TextBlob.
    Prints average polarity (-1..1) and subjectivity (0..1) + a simple label.
    """
    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.secho("No sentences found to analyze.", fg="yellow")
        return

    total_polarity = 0.0
    total_subjectivity = 0.0

    for sentence in sentences:
        blob = TextBlob(sentence)
        total_polarity += blob.sentiment.polarity
        total_subjectivity += blob.sentiment.subjectivity

    # Average over all sentences
    n = len(sentences)
    avg_polarity = total_polarity / n
    avg_subjectivity = total_subjectivity / n

    # Simple labeling rule-of-thumb
    if avg_polarity > 0.1:
        label = "positive"
        color = "green"
    elif avg_polarity < -0.1:
        label = "negative"
        color = "red"
    else:
        label = "neutral"
        color = "yellow"

    click.secho(f"Sentiment: {label}", fg=color)
    click.echo(f"Average Polarity: {avg_polarity:.3f}")
    click.echo(f"Average Subjectivity: {avg_subjectivity:.3f}")



@main.command()
@click.option(
    "--no-positions", is_flag=True, help="Hide start/end character offsets."
)
@click.pass_context
def entity(ctx, no_positions):
    """Recognize named entities in a list of sentences"""
    if nlp is None:
        raise click.ClickException(
            "spaCy model 'en_core_web_sm' is not installed.\n"
            "Install it with: python -m spacy download en_core_web_sm"
        )

    sentences = ctx.obj.get("sentences", [])
    if not sentences:
        click.secho("No sentences found to analyze.", fg="yellow")
        return

    all_ents = []

    # Process each sentence and collect entities
    for sentence in sentences:
        doc = nlp(sentence)
        for ent in doc.ents:
            all_ents.append(ent)

    if not all_ents:
        click.secho("No named entities found.", fg="yellow")
        return

    # Header
    if no_positions:
        click.echo(f"{'ENTITY':40}  {'LABEL':10}")
        click.echo(f"{'-'*40}  {'-'*10}")
    else:
        click.echo(f"{'ENTITY':40}  {'LABEL':10}  {'SPAN':12}")
        click.echo(f"{'-'*40}  {'-'*10}  {'-'*12}")

    # Display all entities
    for ent in all_ents:
        entity_text = ent.text.replace("\n", " ")
        if no_positions:
            click.echo(f"{entity_text[:40]:40}  {ent.label_:10}")
        else:
            span = f"{ent.start_char}-{ent.end_char}"
            click.echo(f"{entity_text[:40]:40}  {ent.label_:10}  {span:12}")





if __name__ =="__main__":
    main()