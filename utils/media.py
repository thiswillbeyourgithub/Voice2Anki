import time
from typing import List, Union, Tuple
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


@trace
def get_image(gallery) -> List[gr.Gallery]:
    whi("Getting image from clipboard")
    assert shared.pv["enable_gallery"], "Incoherent UI"
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
                if isinstance(im, tuple):
                    assert Path(im[0]).exists(), f"Missing image from tuple {im}"
                    assert im[1] is None, f"Unexpected tupe: {im}"
                    out.append(
                            rgb_to_bgr(
                                cv2.imread(
                                    im[0],
                                    flags=1)
                                )
                            )
                else:
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
def check_source(source: str) -> str:
    "makes sure the source is only an img"
    whi("Checking source")
    assert shared.pv["enable_gallery"], "Incoherent UI"
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
def get_img_source(gallery, queue=queue.Queue(), use_html=True) -> None:
    whi("Getting source from image")
    assert shared.pv["enable_gallery"], "Incoherent UI"

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
                try:
                    path = img["image"]["path"]
                    if path.startswith("http"):
                        path = path.split("file=")[1]
                    cnt = 0
                    while not Path(path).exists():
                        time.sleep(1)
                        cnt += 1
                        if cnt == 10:
                            raise Exception(f"img not found in path: {path}")
                    decoded = cv2.imread(path, flags=1)
                except:
                    # must be a tuple
                    assert isinstance(img, tuple), f"Invalid img type: {img}"
                    assert len(img) == 2, f"Invalid img: {img}"
                    assert img[1] is None, f"Invalid img: {img}"
                    cnt = 0
                    while not Path(img[0]).exists():
                        time.sleep(1)
                        cnt += 1
                        if cnt == 10:
                            raise Exception(f"img not found: {img[0]}")
                    decoded = cv2.imread(img[0], flags=1)
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

                newsource = f'<img src="{new.name}" {ocr}type="made_by_Voice2Anki">'

                # only add if not duplicate, somehow
                if newsource not in source:
                    source += newsource

                source = check_source(source)
            else:
                source += "\n" + ocr

        return queue.put(source)
    except Exception as err:
        return queue.put(red(f"Error getting source: '{err}' from {gallery}"))

@trace
def ocr_image(gallery) -> None:
    "use OCR to get the text of an image to display in a textbox"
    q = queue.Queue()
    get_img_source(gallery, q, use_html=False)
    return q.get()


# @trace
def reset_gallery() -> None:
    whi("Reset images.")
    shared.pv["gallery"] = None
    assert shared.pv["enable_gallery"], "Incoherent UI"


# @trace
def reset_audio() -> List[dict]:
    whi("Resetting all audio")
    return [gr.update(value=None, label=f"Audio #{i+1}") for i in range(shared.audio_slot_nb)]

@trace
def sound_preprocessing(audio_mp3_path):
    "removing silence, maybe try to enhance audio, apply filters etc"
    whi(f"Preprocessing {audio_mp3_path}")

    if audio_mp3_path is None:
        whi("Not cleaning sound because received None")
        return None

    assert "_proc" not in str(audio_mp3_path), f"Audio already processed apparently: {audio_mp3_path}"

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
    new_path = Path(audio_mp3_path).parent / (Path(audio_mp3_path).stem + "_proc" + Path(audio_mp3_path).suffix)
    temp.export(new_path, format="mp3")

    whi(f"Done preprocessing {audio_mp3_path} to {new_path}")
    return new_path

@trace
def force_sound_processing(audio_mp3: dict) -> str:
    assert audio_mp3 is not None, "Received None in force_sound_processing"

    audio_mp3 = format_audio_component(audio_mp3)
    waveform, sample_rate = torchaudio.load(audio_mp3)

    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            shared.force_preprocess_sox_effects,
            )

    # write to file as wav
    sf.write(str(audio_mp3), waveform.numpy().T, sample_rate, format='wav')
    temp = AudioSegment.from_wav(audio_mp3)
    new_path = Path(audio_mp3).parent / (Path(audio_mp3).stem + "_fproc" + Path(audio_mp3).suffix)
    temp.export(new_path, format="mp3")

    whi(f"Done forced preprocessing {audio_mp3} to {new_path}")
    return new_path


# @trace
def format_audio_component(audio: Union[str, gr.Audio]) -> str:
    """to make the whole UI faster and avoid sending multiple slightly
    differently processed audio to whisper: preprocessing and postprocessing
    are disabled but this sometimes make the audio component output a dict
    instead of the mp3 audio path. This fixes it while still keeping the cache
    working."""
    if isinstance(audio, dict):
        new_audio = audio["path"]
        if new_audio.startswith("http"):
            new_audio = new_audio.split("file=")[1]
        # whi(f"Preprocessed audio manually: '{audio}' -> '{new_audio}'")
        audio = new_audio
    elif isinstance(audio, (str, type(Path()))):
        # whi(f"No audio formating needed for '{audio}'")
        pass
    else:
        raise ValueError(red(f"Unexpected audio format for {audio}: {type(audio)}"))
    return audio


def rgb_to_bgr(image):
    """gradio is turning cv2's BGR colorspace into RGB, so
    I need to convert it again"""
    assert shared.pv["enable_gallery"], "Incoherent UI"
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


@trace
def roll_future_galleries(*fg: Tuple[dict]) -> List[dict]:
    "pop the first future gallery and send it to the main gallery"
    assert shared.pv["enable_queued_gallery"], "Incoherent UI"
    return list(fg) + [None]


def create_audio_compo(**kwargs) -> gr.Microphone:
    defaults = {
            "type": "filepath",
            "format": "mp3",
            "value": None,
            "container": False,
            "show_share_button": False,
            "show_download_button": True,
            "waveform_options": {"show_controls": False, "show_recording_waveform": False},
            "elem_id": "Audio_component_Voice2Anki",
            "min_width": "1%",
            "editable": True,
            "scale": 1,
            "label": "Untitled",
            "show_label": True,
            }
    defaults.update(kwargs)
    return gr.Microphone(**kwargs)


@trace
def roll_audio(*slots) -> List[dict]:
    assert len(slots) > 1, f"invalid number of audio slots: {len(slots)}"
    slots = list(slots)
    if all((slot is None for slot in slots)):
        return slots
    if all((slot is None for slot in slots[1:])):
        return slots

    slots.pop(0)

    # update the name of each audio to its neighbour
    for i, s in enumerate(slots):
        if s is None:
            continue
        slots[i] = {
                "__type__": "update",  # this is how gr.update works
                "label": slots[i]["orig_name"],
                "value": slots[i]["path"],
                }
    while None in slots:
        slots.remove(None)

    while len(slots) < shared.audio_slot_nb:
        slots.append(
                {
                    "__type__": "update",
                    "label": "New",
                    "value": None,
                    }
                )

    return slots


def update_audio_slots_txts(*audio_slots_txts) -> List[str]:
    """ran frequently to update the content of the textbox of each pending
    audio to display the transcription and cloze
    """
    if shared.pv["enable_dirload"] is False:
        # the components are invisible so return None
        return [None for i in audio_slots_txts]

    df = shared.dirload_queue
    if df.empty:
        return ["Dirload not yet loaded" for i in audio_slots_txts]

    try:
        df = df[df["loaded"] == True]
        if df.empty:
            return ["Empty" for i in audio_slots_txts]

        trans = df["transcribed"].tolist()
        while len(trans) < len(audio_slots_txts):
            trans.append("Pending?")

        alf = df["alfreded"].tolist()
        while len(alf) < len(trans):
            alf.append("Pending?")

        output = [f"{t}\n-----\n{f}" for t, f in zip(trans, alf)]

        return output
    except Exception as err:
        return [err for i in audio_slots_txts]
