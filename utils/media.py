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


def load_next_audio(audio1, audio2, audio3, audio4, audio5):
    whi("Rolling over audio samples")
    return audio2, audio3, audio4, audio5, None


def save_audio(profile, audio_numpy_1):
    whi("Saving audio from #1 to profile")
    pv = previous_values(profile)
    pv["audio_numpy_1"] = audio_numpy_1


def save_audio2(profile, audio_numpy_2):
    whi("Saving audio from #2 to profile")
    pv = previous_values(profile)
    pv["audio_numpy_2"] = audio_numpy_2


def save_audio3(profile, audio_numpy_3):
    whi("Saving audio from #3 to profile")
    pv = previous_values(profile)
    pv["audio_numpy_3"] = audio_numpy_3


def save_audio4(profile, audio_numpy_4):
    whi("Saving audio from #4 to profile")
    pv = previous_values(profile)
    pv["audio_numpy_4"] = audio_numpy_4


def save_audio5(profile, audio_numpy_5):
    whi("Saving audio from #5 to profile")
    pv = previous_values(profile)
    pv["audio_numpy_5"] = audio_numpy_5


def enhance_audio(audio_numpy_1):
    raise NotImplementedError(
        "Enhancing the audio automatically is not supported for now")
    whi("Cleaning voice")
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="enhance")
        write(tmp.name, audio_numpy_1[0], audio_numpy_1[1])
        cleaned = voice_cleaner.enhance_file(tmp.name)
        torchaudio.save(tmp.name, cleaned.unsqueeze(0).cpu(), audio_numpy_1[0])
        enhanced_audio_numpy_1 = read(tmp.name)
        Path(tmp.name).unlink(missing_ok=False)

        whi("Done cleaning audio")
        return enhanced_audio_numpy_1

    except Exception as err:
        red(f"Error when cleaning voice: '{err}'")
        Path(tmp.name).unlink(missing_ok=True)
        return audio_numpy_1


def remove_silences(audio_numpy_1):
    whi("Removing silences")
    try:
        # first saving numpy audio to file
        # note: audio_numpy_1 is a 2-tuple (samplerate, array)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="unsilence")
        write(tmp.name, audio_numpy_1[0], audio_numpy_1[1])
        u = Unsilence(tmp.name)
        u.detect_silence()

        # by how much to speed up silence
        silence_speed = 2

        # do it only if its worth it as it might degrade audio quality?
        estimated_time = u.estimate_time(audible_speed=1, silent_speed=silence_speed)  # Estimate time savings
        before = estimated_time["before"]["all"][0]
        after = estimated_time["after"]["all"][0]
        if after / before > 0.9 and before - after < 5:
            whi(f"Not removing silence (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            Path(tmp.name).unlink(missing_ok=False)
            return audio_numpy_1  # return untouched

        if after > 30:
            silence_speed += 2
            if after > 60:
                silence_speed += 3
                whi(f"Removing silence: longer than 60s detected so speeding up even more (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            else:
                whi(f"Removing silence: longer than 30s detected so speeding up a bit more (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            estimated_time = u.estimate_time(audible_speed=1, silent_speed=silence_speed)  # Estimate time savings
            before = estimated_time["before"]["all"][0]
            after = estimated_time["after"]["all"][0]

        yel(f"Removing silence: {before:.1f}s -> {after:.1f}s")
        u.render_media(tmp.name, audible_speed=1, silent_speed=silence_speed, audio_only=True)
        unsilenced_audio_numpy_1 = read(tmp.name)
        whi("Done removing silences")
        Path(tmp.name).unlink(missing_ok=False)
        return unsilenced_audio_numpy_1
    except Exception as err:
        red(f"Error when removing silences: '{err}'")
        Path(tmp.name).unlink(missing_ok=True)
        return audio_numpy_1


# load voice cleaning model
voice_cleaner = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="cache/pretrained_models/mtl-mimic-voicebank",
)
