class SharedModule:
    """module used to store information from VoiceToFormattedText.py to
    the main .py files"""
    # sox effect when loading a sound
    preprocess_sox_effects = [
            # isolate voice frequency
            # -2 is for a steeper filtering
            ["highpass", "-1", "100"],
            ["lowpass", "-1", "3000"],
            # removes high frequency and very low ones
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],
            # # normalize audio
            ["norm"],
            # max silence should be 1s
            ["silence", "-l", "1", "0.1", "1%", "-1", "1.0", "1%"],

            # remove leading silence
            ["vad", "-p", "0.2", "-t", "5"],

            # and ending silence, this might be unecessary for splitted audio
            ["reverse"],
            ["vad", "-p", "0.2", "-t", "5"],
            ["reverse"],

            # add blank sound to help whisper
            ["pad", "0.2@0"],

            ]
    # sox effects when splitting long audio
    splitter_sox_effects = [
            # isolate voice frequency
            # -2 is for a steeper filtering
            ["highpass", "-1", "100"],
            ["lowpass", "-1", "3000"],
            # removes high frequency and very low ones
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],

            ["norm"],  # normalize audio

            # max silence should be 3s
            ["silence", "-l", "1", "0.1", "0.1%", "-1", "2.0", "0.1%"],

            ["norm"],
            ]

    max_message_buffer = 50

    audio_slot_nb = 5
    future_gallery_slot_nb = 50

    dirload_queue = []
    dirload_doing = []

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
        self.dirload_queue = []
        self.dirload_doing = []
        self.llm_to_db_buffer = {}
        self.latest_stt_used = None
        self.latest_llm_used = None
        for k in self.running_threads:
            self.running_threads[k] = []
        self.added_note_ids = []



shared = SharedModule()
