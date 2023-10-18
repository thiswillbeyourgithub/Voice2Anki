import cv2
import tiktoken


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
    VERSION = 0.1
    def __init__(self):
        self.backend = "not yet initialized"
global backend_config
backend_config = backend_config_class()

def format_audio_component(audio):
    """to make the whole UI faster, preprocessing and postprocessing is disabled
    but this sometimes make the audio component output a dict instead of
    the audio path. This fixes it."""
    if isinstance(audio, dict):
        assert audio["is_file"], "unexpected dict instead of audio"
        audio = audio["name"]
    return audio
