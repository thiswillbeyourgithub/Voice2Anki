from pydub import AudioSegment
import soundfile as sf
from pathlib import Path
import gradio as gr
import queue
import pickle
from bs4 import BeautifulSoup
import cv2
import numpy as np
import pyclip
import hashlib
import torchaudio

from joblib import Memory

from .logger import whi, red, trace, Timeout
from .ocr import get_text
from .shared_module import shared

soundpreprocess_cache = Memory("cache/sound_preprocessing_cache", verbose=0)
soundpreprocess_cache.clear()  # clear the cache on startup


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

        if hasattr(gallery, "root"):
            gallery = gallery.root
        if isinstance(gallery, list):
            out = []
            for im in gallery:
                out.append(
                        rgb_to_bgr(
                            cv2.imread(
                                im.image.path,
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


#@Timeout(120)
@trace
def get_img_source(gallery, queue=queue.Queue(), use_html=True):
    whi("Getting source from image")

    try:
        if hasattr(gallery, "root"):
            gallery = gallery.root
        assert isinstance(gallery, (type(None), list)), "Gallery is not a list or None"
        if gallery is None:
            return queue.put(red("No image in gallery."))
        if len(gallery) == 0:
            return queue.put(red("0 image found in gallery."))

        source = ""
        for img in gallery:
            try:
                decoded = cv2.imread(img.image.path, flags=1)
            except:
                decoded = cv2.imread(img["image"]["path"], flags=1)
            img_hash = hashlib.md5(decoded).hexdigest()
            new = shared.anki_media / f"{img_hash}.png"
            if not new.exists():
                cv2.imwrite(str(new), decoded)

            try:
                ocr = get_text(str(new))
            except Exception as err:
                ocr = red(f"Error when OCRing image: '{err}'")

            if use_html:
                if ocr:
                    ocr = ocr.replace("\"", "").replace("'", "")
                    ocr = f"title=\"{ocr}\" "

                newsource = f'<img src="{new.name}" {ocr}type="made_by_WhisperToAnki">'

                # only add if not duplicate, somehow
                if newsource not in source:
                    source += newsource

                source = check_source(source)
            else:
                source += "\n" + ocr

        return queue.put(source)
    except Exception as err:
        return queue.put(red(f"Error getting source: '{err}'"))

@trace
def ocr_image(gallery):
    "use OCR to get the text of an image to display in a textbox"
    q = queue.Queue()
    get_img_source(gallery, q, use_html=False)
    return q.get()


# @trace
def reset_gallery():
    whi("Reset images.")
    shared.pv["gallery"] = None
    return None


# @trace
def reset_audio():
    whi("Resetting all audio")
    return None, None, None, None, None

@trace
def sound_preprocessing(audio_mp3_path):
    "removing silence, maybe try to enhance audio, apply filters etc"
    whi(f"Preprocessing {audio_mp3_path}")

    if audio_mp3_path is None:
        whi("Not cleaning sound because received None")
        return None

    # load from file
    waveform, sample_rate = torchaudio.load(audio_mp3_path)

    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            shared.preprocess_sox_effects,
            )

    # write to file as wav
    sf.write(str(audio_mp3_path), waveform.numpy().T, sample_rate, format='wav')
    temp = AudioSegment.from_wav(audio_mp3_path)
    new_path = Path(audio_mp3_path).parent / (Path(audio_mp3_path).stem + "_processed" + Path(audio_mp3_path).suffix)
    temp.export(new_path, format="mp3")

    whi(f"Done preprocessing {audio_mp3_path} to {new_path}")
    return new_path

@trace
def force_sound_processing(audio_mp3_path):
    assert audio_mp3_path is not None, "Received None in force_sound_processing"
    audio_mp3_path = format_audio_component(audio_mp3_path)

    # load from file
    waveform, sample_rate = torchaudio.load(audio_mp3_path)

    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            shared.force_preprocess_sox_effects,
            )

    # write to file as wav
    sf.write(str(audio_mp3_path), waveform.numpy().T, sample_rate, format='wav')
    temp = AudioSegment.from_wav(audio_mp3_path)
    new_path = Path(audio_mp3_path).parent / (Path(audio_mp3_path).stem + "_processed" + Path(audio_mp3_path).suffix)
    temp.export(new_path, format="mp3")

    whi(f"Done forced preprocessing {audio_mp3_path} to {new_path}")
    return new_path


@trace
def format_audio_component(audio):
    """to make the whole UI faster and avoid sending multiple slightly
    differently processed audio to whisper: preprocessing and postprocessing
    are disabled but this sometimes make the audio component output a dict
    instead of the mp3 audio path. This fixes it while still keeping the cache
    working."""
    if isinstance(audio, dict):
        new_audio = audio["path"]
        if new_audio.startswith("http"):
            new_audio = new_audio.split("file=")[1]
        whi(f"Preprocessed audio manually: '{audio}' -> '{new_audio}'")
        audio = new_audio
    elif isinstance(audio, (str, type(Path()))):
        whi(f"Not audio formating needed for '{audio}'")
    else:
        raise ValueError(red(f"Unexpected audio format for {audio}: {type(audio)}"))
    return audio


def rgb_to_bgr(image):
    """gradio is turning cv2's BGR colorspace into RGB, so
    I need to convert it again"""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@trace
def load_future_galleries():
    """load the saved images beforehand to reorder so that the empty
    galleries are moved at the end"""
    saved_fg = [shared.pv[f"future_gallery_{fg:03d}"] for fg in range(1, shared.future_gallery_slot_nb + 1)]
    while None in saved_fg:
        saved_fg.remove(None)
    if len(saved_fg) < shared.future_gallery_slot_nb:
        saved_fg.extend([None] * ( shared.future_gallery_slot_nb - len(saved_fg)))
    assert len(saved_fg) == shared.future_gallery_slot_nb
    for i, fg in enumerate(range(1, shared.future_gallery_slot_nb + 1)):
        im = saved_fg[i]
        getattr(shared.pv, f"save_future_gallery_{fg:03d}")(im)
    return saved_fg


def create_audio_compo():
    return gr.Microphone(
            type="filepath",
            format="mp3",
            value=None,
            container=False,
            show_share_button=False,
            show_download_button=True,
            waveform_options={"show_controls": False, "show_recording_waveform": False},
            elem_id="Audio_component_Voice2Anki",
            elem_classes="Audio_component_Voice2Anki",
            # min_width=10,
            editable=False,
            )


@trace
def roll_audio(*slots):
    assert len(slots) > 1, f"invalid number of audio slots: {len(slots)}"
    slots = list(slots)
    if all((slot is None for slot in slots)):
        return slots
    if all((slot is None for slot in slots[1:])):
        return slots
    slots[0] = None
    while slots[0] is None:
        slots.pop(0)
        audio_mp3 = create_audio_compo()
        slots.append(audio_mp3)

    return slots
