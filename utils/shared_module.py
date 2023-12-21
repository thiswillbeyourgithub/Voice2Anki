import pandas as pd

class SharedModule:
    """module used to store information from VoiceToFormattedText.py to
    the main .py files"""
    VERSION = 0.2
    memory_metric = None
    media_folder = None
    anki_media = None
    debug = None
    disable_tracing = None
    disable_timeout = None
    compact_js = None
    backend = None
    pv = None

    llm_price = {
            "gpt-3.5-turbo-1106": (0.001, 0.002),
            "gpt-4-1106-preview": (0.01, 0.03),
            }

    # sox effect when loading a sound
    preprocess_sox_effects = [
            # isolate voice frequency
            # -2 is for a steeper filtering
            # ["highpass", "-1", "100"],
            # ["lowpass", "-1", "3000"],
            # # removes high frequency and very low ones
            # ["highpass", "-2", "50"],
            # ["lowpass", "-2", "5000"],
            # # # normalize audio
            # ["norm"],
            # max silence should be 1s
            ["silence", "-l", "1", "0", "0.5%", "-1", "1.0", "0.1%"],

            # # remove leading silence
            # ["vad", "-p", "0.2", "-t", "5"],

            # # and ending silence, this might be unecessary for splitted audio
            # ["reverse"],
            # ["vad", "-p", "0.2", "-t", "5"],
            # ["reverse"],

            # add blank sound to help whisper
            # ["pad", "0.2@0"],

            ]
    # sox effects when splitting long audio
    splitter_sox_effects = [
            # isolate voice frequency
            ["highpass", "-1", "100"],
            ["lowpass", "-1", "3000"],
            # -2 is for a steeper filtering: removes high frequency and very low ones
            # ["highpass", "-2", "50"],
            # ["lowpass", "-2", "5000"],

            ["norm"],  # normalize audio

            # max silence should be 3s
            ["silence", "-l", "1", "0", "0.1%", "-1", "2.0", "0.1%"],

            ["norm"],
            ]

    message_buffer = []
    max_message_buffer = 50

    audio_slot_nb = 3
    future_gallery_slot_nb = 60

    dirload_queue_columns = [
            "n",
            "path",
            "temp_path",
            "loaded",
            "sound_preprocessed",
            "transcribed",
            "alfreded",
            "ankified",
            "moved",
            ]
    dirload_queue = pd.DataFrame(columns=dirload_queue_columns).set_index("path")

    llm_to_db_buffer = {}
    latest_stt_used = None
    latest_llm_used = None

    running_threads = {
            "saving_chatgpt": [],
            "saving_whisper": [],
            "transcribing_audio": [],
            "audio_to_anki": [],
            "ocr": [],
            "timeout": [],
            }

    added_note_ids = []

    def reset(self):
        "used to reset the values when the gradio page is reloaded"
        print("Resetting shared module.")
        self.dirload_queue = pd.DataFrame(columns=self.dirload_queue_columns).set_index("path")
        self.llm_to_db_buffer = {}
        self.latest_stt_used = None
        self.latest_llm_used = None
        for k in self.running_threads:
            self.running_threads[k] = []
        self.added_note_ids = []
        self.pv.running_tasks = {k: None for k in self.pv.approved_keys}
        self.pv.cache_values = {k: None for k in self.pv.approved_keys}

    def __setattr__(self, name, value):
        "forbid creation of new attributes."
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError(f'Cannot set name {name} on object of type {self.__class__.__name__}')



shared = SharedModule()
