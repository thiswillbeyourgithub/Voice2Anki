import queue
import time
from scipy.io.wavfile import write
import pickle
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pyclip
import hashlib
import torchaudio

from joblib import Memory

from .logger import whi, red, trace
from .anki_utils import anki_media
from .ocr import get_text
from .profiles import ValueStorage
from .misc import rgb_to_bgr

soundpreprocess_cache = Memory("cache/sound_preprocessing_cache", verbose=0)

@trace
def get_image(gallery):
    whi("Getting image from clipboard")
    try:
        # load from clipboard
        pasted = pyclip.paste()
        decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
        decoded = rgb_to_bgr(decoded)

        if decoded is None:
            whi("Image from clipboard was Nonetype")
            return gallery

        if gallery is None:
            return [decoded]

        if isinstance(gallery, list):
            out = []
            for im in gallery:
                out.append(
                        rgb_to_bgr(
                            cv2.imread(
                                im["name"],
                                flags=1)
                            )
                        )
            out += [decoded]

            whi("Loaded image from clipboard.")
            return out

        else:
            red(f'gallery is not list or None but {type(gallery)}')
            return None

    except Exception as err:
        red(f"Error: {err}")
        return None


@trace
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


@trace
def get_img_source(gallery, queue=queue.Queue()):
    whi("Getting source from image")
    try:
        assert isinstance(gallery, (type(None), list)), "Gallery is not a list or None"
        if gallery is None:
            queue.put(red("No image in gallery."))
        if len(gallery) == 0:
            queue.put(red("0 image found in gallery."))
        source = ""
        for img in gallery:
            decoded = cv2.imread(img["name"], flags=1)
            img_hash = hashlib.md5(decoded).hexdigest()
            new = anki_media / f"{img_hash}.png"
            if not new.exists():
                cv2.imwrite(str(new), decoded)

            ocr = ""
            try:
                ocr = get_text(str(new))
            except Exception as err:
                red(f"Error when OCRing image: '{err}'")
            if ocr:
                ocr = ocr.replace("\"", "").replace("'", "")
                ocr = f"title=\"{ocr}\" "

            newsource = f'<img src="{new.name}" {ocr}type="made_by_WhisperToAnki">'

            # only add if not duplicate, somehow
            if newsource not in source:
                source += newsource

        source = check_source(source)
        queue.put(source)
    except Exception as err:
        queue.put(red(f"Error getting source: '{err}'"))


# @trace
def reset_image():
    whi("Reset images.")
    return None


# @trace
def reset_audio():
    whi("Resetting all audio")
    return None, None, None, None, None

@trace
def roll_audio(
        txt_profile,
        audio_mp3_1,
        audio_mp3_2,
        audio_mp3_3,
        audio_mp3_4,
        audio_mp3_5):
    # if 2-5 are None, keep the 1
    if audio_mp3_2 is None and audio_mp3_3 is None and audio_mp3_4 is None and audio_mp3_5 is None:
        return audio_mp3_1, None, None, None, None

    audio_mp3_1 = None
    while audio_mp3_1 is None:  # roll
        audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5 = audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5, None

    return audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5


@trace
@soundpreprocess_cache.cache
def sound_preprocessing(audio_mp3_n):
    "removing silence, maybe try to enhance audio, apply filters etc"
    whi(f"Cleaning {audio_mp3_n} with torchaudio")

    if audio_mp3_n is None:
        whi("Not cleaning sound because received None")
        return None

    waveform, sample_rate = torchaudio.load(audio_mp3_n)

    # voice activity detector (i.e. trims the beginning of the sound until you speak)
    # vad_waveform = torchaudio.functional.vad(
    #         waveform=waveform,
    #         sample_rate=sample_rate,
    #         trigger_level=5.0,
    #         trigger_time=0.25,
    #         search_time=0.5,
    #         allowed_gap=0.10,
    #         pre_trigger_time=0.0,
    #         boot_time=0.35,
    #         noise_up_time=0.1,
    #         noise_down_time=0.01,
    #         noise_reduction_amount=1.5,
    #         measure_freq=20.0,
    #         measure_duration=None,
    #         measure_smooth_time=0.4,
    #         hp_filter_freq=50.0,
    #         lp_filter_freq=6000.0,
    #         hp_lifter_freq=150.0,
    #         lp_lifter_freq=2000.0,
    #         )

    sox_effects = [
            # ["norm"],  # normalize audio

            # isolate voice frequency
            # -2 is for a steeper filtering
            ["highpass", "-1", "100"],
            ["lowpass", "-1", "3000"],
            # removes high frequency and very low ones
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],

            # # max silence should be 2s
            ["silence", "-l", "1", "2.0", "5%", "-1", "2.0", "5%"],

            # # remove leading silence
            ["vad"],

            # # and ending silence
            ["reverse"],
            ["vad"],
            ["reverse"],
            ]
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            sox_effects,
            )

    write(audio_mp3_n, sample_rate, waveform.numpy().T)

    whi("Done preprocessing audio")
    return audio_mp3_n
