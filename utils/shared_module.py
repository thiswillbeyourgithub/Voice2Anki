class SharedModule:
    """module used to store information from VoiceToFormattedText.py to
    the main .py files"""
    threads = []

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
            ["silence", "-l", "1", "0.1", "0.1%", "-1", "1.0", "0.1%"],

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

    llm_cache = {}

    llm_to_db_buffer = {}


shared = SharedModule()
