from pathlib import Path
import tempfile
from scipy.io.wavfile import write, read
import pickle
from bs4 import BeautifulSoup
from speechbrain.pretrained import WaveformEnhancement
import cv2
import numpy as np
import pyperclip3
import hashlib
from unsilence import Unsilence
import torchaudio

from .logger import whi, yel, red
from .anki import anki_media
from .profiles import previous_values


def get_image(gallery):
    whi("Getting image from clipboard")
    try:
        # load from clipboard
        pasted = pyperclip3.paste()
        decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
        if decoded is None:
            whi("Decoded image was Nonetype")
            return gallery
        if gallery is None:
            return [decoded]
        if isinstance(gallery, list):
            out = [
                    cv2.imread(
                        i["name"]
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
            decoded = cv2.imread(img["name"])
            img_hash = hashlib.md5(decoded).hexdigest()
            new = anki_media / f"{img_hash}.png"
            if not new.exists():
                cv2.imwrite(str(new), decoded)
            newsource = f'<img src="{new.name}" type="made_by_WhisperToAnki">'
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


def reset_audio(audio1, audio2, audio3):
    whi("Resetting all audio")
    return None, None, None


def load_next_audio(audio1, audio2, audio3):
    whi("Rolling over audio samples")
    if audio1 is None:
        whi("Cannot load next audio if audio #1 is None")
        return audio1, audio2, audio3
    return audio2, audio3, None


def save_audio(profile, audio_numpy):
    whi("Saving audio from #1 to profile")
    pv = previous_values(profile)
    pv["audio_numpy"] = audio_numpy


def enhance_audio(audio_numpy):
    raise NotImplementedError(
        "Enhancing the audio automatically is not supported for now")
    whi("Cleaning voice")
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="enhance")
        write(tmp.name, audio_numpy[0], audio_numpy[1])
        cleaned = voice_cleaner.enhance_file(tmp.name)
        torchaudio.save(tmp.name, cleaned.unsqueeze(0).cpu(), audio_numpy[0])
        enhanced_audio_numpy = read(tmp.name)
        Path(tmp.name).unlink(missing_ok=False)

        whi("Done cleaning audio")
        return enhanced_audio_numpy

    except Exception as err:
        red(f"Error when cleaning voice: '{err}'")
        Path(tmp.name).unlink(missing_ok=True)
        return audio_numpy


def remove_silences(audio_numpy):
    whi("Removing silences")
    try:
        # first saving numpy audio to file
        # note: audio_numpy is a 2-tuple (samplerate, array)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="unsilence")
        write(tmp.name, audio_numpy[0], audio_numpy[1])
        u = Unsilence(tmp.name)
        u.detect_silence()

        # do it only if its worth it as it might degrade audio quality?
        estimated_time = u.estimate_time(audible_speed=1, silent_speed=2)  # Estimate time savings
        before = estimated_time["before"]["all"][0]
        after = estimated_time["after"]["all"][0]
        if after / before > 0.9 and before - after < 5:
            whi(f"Not removing silence (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            Path(tmp.name).unlink(missing_ok=False)
            return audio_numpy  # return untouched

        yel(f"Removing silence: {before:.1f}s -> {after:.1f}s")
        u.render_media(tmp.name, audible_speed=1, silent_speed=2, audio_only=True)
        unsilenced_audio_numpy = read(tmp.name)
        whi("Done removing silences")
        Path(tmp.name).unlink(missing_ok=False)
        return unsilenced_audio_numpy
    except Exception as err:
        red(f"Error when removing silences: '{err}'")
        Path(tmp.name).unlink(missing_ok=True)
        return audio_numpy


# load voice cleaning model
voice_cleaner = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="cache/pretrained_models/mtl-mimic-voicebank",
)
