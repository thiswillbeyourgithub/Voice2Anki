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

    dirload_queue = []
    dirload_doing = []

    audio_slot_nb = 5
    future_gallery_slot_nb = 50

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



shared = SharedModule()

def reset_shared():
    "called when the gradio page is loaded. As otherwise it means the attributes of shared at not in sync anymore."
    global shared
    print("Reset shared module.")
    shared = SharedModule()
