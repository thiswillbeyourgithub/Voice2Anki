import json
from pathlib import Path
import tempfile
from scipy.io.wavfile import write
import pickle
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pyclip
import hashlib
from torchaudio import load
from torchaudio.functional import vad

from .logger import whi, red
from .anki import anki_media
from .ocr import get_text
from .profiles import previous_values


def get_image(gallery):
    whi("Getting image from clipboard")
    try:
        # load from clipboard
        pasted = pyclip.paste()
        decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
        if decoded is None:
            whi("Decoded image was Nonetype")
            return gallery
        if gallery is None:
            return [decoded]
        if isinstance(gallery, list):
            out = [
                    cv2.imread(
                        i["name"],
                        flags=1,
                        ) for i in gallery
                    ] + [decoded]
            whi("Loaded image from clipboard.")
            return out
        else:
            red(f'gallery is not list or None but {type(gallery)}')
            return None
    except Exception as err:
        red(f"Error: {err}")
        return None


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


def get_img_source(gallery):
    whi("Getting source from image")
    try:
        assert isinstance(gallery, (type(None), list)), "Gallery is not a list or None"
        if gallery is None:
            return red("No image in gallery.")
        if len(gallery) == 0:
            return red("0 image found in gallery.")
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
        return source
    except Exception as err:
        return red(f"Error getting source: '{err}'")


def reset_image():
    whi("Reset images.")
    return None


def reset_audio(audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5):
    whi("Resetting all audio")
    return None, None, None, None, None


def load_next_audio(txt_profile, audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5):
    whi("Rolling over audio samples")
    pv = previous_values(txt_profile)
    pv["audio_numpy_1"] = audio_numpy_2
    pv["audio_numpy_2"] = audio_numpy_3
    pv["audio_numpy_3"] = audio_numpy_4
    pv["audio_numpy_4"] = audio_numpy_5
    pv["audio_numpy_5"] = None

    return audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5, None


class audio_saver:
    def save_audio(self, txt_profile, audio_numpy_n, n):
        whi(f"Saving audio from #{n} to profile")
        if audio_numpy_n is None:
            whi("Not saving because sound is None")
            return
        pv = previous_values(txt_profile)
        pv[f"audio_numpy_{n}"] = audio_numpy_n

    def n1(self, txt_profile, audio_numpy):
        return self.save_audio(txt_profile, audio_numpy, n=1)

    def n2(self, txt_profile, audio_numpy):
        return self.save_audio(txt_profile, audio_numpy, n=2)

    def n3(self, txt_profile, audio_numpy):
        return self.save_audio(txt_profile, audio_numpy, n=3)

    def n4(self, txt_profile, audio_numpy):
        return self.save_audio(txt_profile, audio_numpy, n=4)

    def n5(self, txt_profile, audio_numpy):
        return self.save_audio(txt_profile, audio_numpy, n=5)


def sound_preprocessing(audio_numpy_n):
    "removing silence, maybe try to enhance audio, apply filters etc"
    whi("Cleaning sound with torchaudio")

    if audio_numpy_n is None:
        whi("Not cleaning sound because received None")
        return None

    # save as wav file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="preprocessing")
    write(tmp.name, audio_numpy_n[0], audio_numpy_n[1])

    tens = load(tmp.name)
    cleaned = vad(
            waveform=tens[0],
            sample_rate=tens[1],
            trigger_level=7.0,
            trigger_time=0.25,
            search_time=1.0,
            allowed_gap=0.25,
            pre_trigger_time=0.0,
            boot_time=0.35,
            noise_up_time=0.1,
            noise_down_time=0.01,
            noise_reduction_amount=1.35,
            measure_freq=20.0,
            measure_duration=None,
            measure_smooth_time=0.4,
            hp_filter_freq=50.0,
            lp_filter_freq=6000.0,
            hp_lifter_freq=150.0,
            lp_lifter_freq=2000.0,
            )

    Path(tmp.name).unlink(missing_ok=True)
    audio_numpy_n = tuple((audio_numpy_n[0], cleaned.numpy().T))

    whi("Done preprocessing audio")
    return audio_numpy_n
