import tiktoken
import ftfy
from bs4 import BeautifulSoup
import pickle

from .logger import whi

# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# string at the end of the completion
completion_finish = "\n END"

# characters that separate the cloze in the completion of the model
cloze_completion_separator = "\n'''\n"

def convert_paste(paste):
    try:
        print("TODO: fix this.")
        # is text
        return ftfy.fix_text(str(paste))
    except:
        # is image
        return paste

# used to count the number of tokens for chatgpt
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenize = tokenizer.encode

transcript_template = """
Contexte: "CONTEXT"

Transcript:
'''
TRANSCRIPT
'''
"""



def check_source(source):
    "makes sure the source is only an img"
    whi("Checking source")
    if source:
        soup = BeautifulSoup(source, 'html.parser')
        imgs = soup.find_all("img")
        source = "</br>".join([str(x) for x in imgs])
        assert source, f"invalid source: {source}"
        # save for next startup
        with open("./cache/voice_cards_last_source.pickle", "wb") as f:
            pickle.dump(source, f)
    else:
        source = ""
    return source
