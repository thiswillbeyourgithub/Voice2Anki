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
