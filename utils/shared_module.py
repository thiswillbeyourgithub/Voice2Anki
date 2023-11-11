class SharedModule:
    """module used to store information from VoiceToFormattedText.py to
    the main .py files"""
    threads = []

    # shared to be used by media.py or audio_splitter.py
    sox_effects = [
            # isolate voice frequency
            # -2 is for a steeper filtering
            ["highpass", "-1", "100"],
            ["lowpass", "-1", "3000"],
            # removes high frequency and very low ones
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],

            # max silence should be 2s
            ["silence", "-l", "1", "0.1", "0.05%", "-1", "2.0", "0.05%"],

            ["norm"],  # normalize audio

            # # remove leading silence
            ["vad"],

            # and ending silence, this might be unecessary for splitted audio
            ["reverse"],
            ["vad"],
            ["reverse"],
            ]


shared = SharedModule()
