from speechbrain.pretrained import WaveformEnhancement
import time
from pathlib import Path
import ankipandas as akp
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pyperclip3
import hashlib
from unsilence import Unsilence
import torchaudio

from .logger import whi, yel, red


def get_image(source, gallery):
    whi("Getting image")
    try:
        if source:
            # load from source if present
            soup = BeautifulSoup(source, 'html.parser')
            path = soup.find_all('img')[0]['src']
            decoded = cv2.imread(str(anki_media / path))
            return decoded

        # else load from clipboard
        pasted = pyperclip3.paste()
        decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
        if gallery is None:
            return [decoded]
        if isinstance(gallery, list):
            out = [
                    cv2.imread(
                        i["name"]
                        ) for i in gallery
                    ] + [decoded]
            return out
        raise Exception(f'gallery is not list or None but {type(gallery)}')
    except Exception as err:
        return red(f"Error: {err}")


def get_img_source(decoded):
    whi("Getting source from image")
    try:
        img_hash = hashlib.md5(decoded).hexdigest()
        new = anki_media / f"{img_hash}.png"
        if not new.exists():
            cv2.imwrite(str(new), decoded)
        source = f'<img src="{new.name}" '
        source += 'type="made_by_WhisperToAnki">'
        return source
    except Exception as err:
        return red(f"Error getting source: '{err}'")


def reset_audio():
    whi("Reset audio.")
    return None


def enhance_audio(audio_path):
    whi("Cleaning voice")
    try:
        cleaned_sound = voice_cleaner.enhance_file(audio_path)

        # overwrites previous sound
        torchaudio.save(audio_path, cleaned_sound.unsqueeze(0).cpu(), 16000)

        whi("Done cleaning audio")
        return audio_path

    except Exception as err:
        red(f"Error when cleaning voice: '{err}'")
        return audio_path


def remove_silences(audio_path):
    whi("Removing silences")
    try:
        u = Unsilence(audio_path)
        u.detect_silence()

        # do it only if its worth it as it might degrade audio quality?
        estimated_time = u.estimate_time(audible_speed=1, silent_speed=2)  # Estimate time savings
        before = estimated_time["before"]["all"][0]
        after = estimated_time["after"]["all"][0]
        if after / before  > 0.9 and before - after < 5:
            whi(f"Not removing silence (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            return audio_path

        yel(f"Removing silence: {before:.1f}s -> {after:.1f}s")
        u.render_media(audio_path, audible_speed=1, silent_speed=2, audio_only=True)
        whi("Done removing silences")
        return audio_path
    except Exception as err:
        red(f"Error when removing silences: '{err}'")
        return audio_path

# load anki profile using ankipandas just to get the media folder
db_path = akp.find_db(user="Main")
red(f"WhisperToAnki will use anki collection found at {db_path}")

# check that akp will not go in trash
if "trash" in str(db_path).lower():
    red("Ankipandas seems to have "
        "found a collection that might be located in "
        "the trash folder. If that is not your intention "
        "cancel now. Waiting 10s for you to see this "
        "message before proceeding.")
    time.sleep(1)
anki_media = Path(db_path).parent / "collection.media"
assert anki_media.exists(), "Media folder not found!"

# load voice cleaning model
voice_cleaner = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="cache/pretrained_models/mtl-mimic-voicebank",
)
