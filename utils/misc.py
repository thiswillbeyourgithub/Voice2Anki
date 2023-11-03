from pathlib import Path
import gradio as gr
import cv2
import tiktoken
from .logger import red, whi, trace


# string at the end of the prompt
prompt_finish = "\n\n###\n\n"

# string at the end of the completion
completion_finish = "\n END"

# used to count the number of tokens for chatgpt
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokenize = tokenizer.encode

transcript_template = """
Context: 'CONTEXT'
Transcript: 'TRANSCRIPT'
""".strip()

def rgb_to_bgr(image):
    """gradio is turning cv2's BGR colorspace into RGB, so
    I need to convert it again"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# to know if the backend chosen by the user is anki or markdown
class backend_config_class:
    "module used to store information from VoiceToFormattedText.py to the main .py files"
    def __init__(self):
        pass
global backend_config
backend_config = backend_config_class()

# create dummy button to use the preprocessing code if needed
dummy_btn = gr.Audio(
        source="microphone",
        type="filepath",
        label="dummy_audio",
        format="mp3",
        value=None,
        container=False)

@trace
def format_audio_component(audio):
    """to make the whole UI faster and avoid sending multiple slightly
    differently processed audio to whisper: preprocessing and postprocessing
    are disabled but this sometimes make the audio component output a dict
    instead of the mp3 audio path. This fixes it while still keeping the cache
    working."""
    if isinstance(audio, dict):
        if "is_file" in audio:
            audio = audio["name"]
        else:
            new_audio = dummy_btn.preprocess(audio)
            red(f"Unexpected dict instead of audio for '{audio['name']}' -> '{new_audio}'")
            audio = new_audio
    elif isinstance(audio, (str, type(Path()))):
        whi(f"Not audio formating needed for '{audio}'")
    else:
        raise ValueError(red(f"Unexpected audio format for {audio}: {type(audio)}"))
    return audio
