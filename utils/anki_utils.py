import queue
from typing import List, Union
from datetime import datetime
import rtoml
import sys
import importlib.util
from functools import partial
import asyncio
import aiohttp
import gradio as gr
import re
import threading
import shutil
import hashlib
from pathlib import Path
import ankipandas as akp
import time
import urllib.request
import json

from .logger import red, whi, trace, Timeout
from .shared_module import shared
from .media import format_audio_component


def _request_wrapper(action, **params):
    return {'action': action, 'params': params, 'version': 6}


@trace
async def anki_request_async(url, request):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request) as response:
            data = await response.json()
            return data


def call_anki(action, **params):
    """ bridge between local python libraries and AnnA Companion addon
    (a fork from anki-connect) """

    if "async_mode" in params:
        use_async = True
        del params["async_mode"]
    else:
        use_async = False

    requestJson = json.dumps(_request_wrapper(action, **params)
                             ).encode('utf-8')

    if not use_async:
        try:
            response = json.load(urllib.request.urlopen(
                urllib.request.Request(
                    'http://localhost:8765',
                    requestJson)))
        except (ConnectionRefusedError, urllib.error.URLError) as e:
            red(f"{str(e)}: is Anki open and 'AnkiConnect' "
                "enabled? Firewall issue?")
            raise Exception(f"{str(e)}: is Anki open and 'AnkiConnect' "
                            "addon' enabled? Firewall issue?")
    else:
        try:
            response = anki_request_async(
                    url='http://localhost:8765',
                    request=requestJson)
        except Exception as e:
            red(f"{str(e)}: is Anki open and 'AnkiConnect' "
                "enabled? Firewall issue?")
            raise Exception(f"{str(e)}: is Anki open and 'AnkiConnect' "
                            "addon' enabled? Firewall issue?")

    if len(response) != 2:
        red('response has an unexpected number of fields')
        raise Exception('response has an unexpected number of fields')
    if 'error' not in response:
        red('response is missing required error field')
        raise Exception('response is missing required error field')
    if 'result' not in response:
        red('response is missing required result field')
        raise Exception('response is missing required result field')
    if response['error'] is not None:
        red(response['error'])
        raise Exception(response['error'])
    return response['result']


@trace
def check_anki_models() -> None:
    """checks for notetype in anki, if no appropriate model is found, create
    one"""
    models = call_anki(action="modelNames")
    if "Clozolkor" in models:
        shared.anki_notetype = "Clozolkor"
    elif "Voice2Anki" in models:
        shared.anki_notetype = "Voice2Anki"
    else:
        gr.Error(red("Anki does not contain a notetype 'Voice2Anki' nor 'Clozolkor', creating it."))
        # create note type model that has the right fields
        call_anki(
                action="createModel",
                modelName="Voice2Anki",
                inOrderFields=["body", "source", "source_extra", "source_audio", "GPToAnkiMetadata"],
                isCloze=True,
                cardTemplates=[
                    {
                        "Front": "{{cloze:body}}",
                        "Back": "{{cloze:body}}<br><br>{{source}}<br>{{source_extra}}<br>{{source_audio}}",
                        },
                    ],
                )
        gr.Error(red("Done creating notetype 'Voice2Anki'"))
        shared.anki_notetype = "Voice2Anki"
    red(f"Anki notetype detected: '{shared.anki_notetype}'")

@trace
def add_note_to_anki(
        bodies: List[str],
        source: str,
        source_extra: str,
        source_audio: str,
        notes_metadata: List[dict],
        tags: List,
        deck_name: str = "Default",
        ) -> List[Union[int, str]]:
    """create a new cloze directly in anki"""
    assert isinstance(tags, list), "tags have to be a list"
    if not shared.anki_notetype:
        check_anki_models()
    model_name = shared.anki_notetype

    if [f for f in shared.func_dir.iterdir() if f.name.endswith("flashcard_editor.py")]:
        red("Found flashcard_editor.py")
        spec = importlib.util.spec_from_file_location(
                "flashcard_editor.cloze_editor",
                (shared.func_dir / "flashcard_editor.py").absolute()
                )
        editor_module = importlib.util.module_from_spec(spec)
        sys.modules["editor_module"] = editor_module
        spec.loader.exec_module(editor_module)
        cloze_editor = editor_module.cloze_editor
    else:
        red("Not flashcard_editor.py found")
        cloze_editor = lambda x: x

    notes = [
            {
                "deckName": deck_name,
                "modelName": model_name,
                "fields": {
                    "body": cloze_editor(body).strip().replace("\n", "<br>"),
                    "source": source,
                    "source_extra": source_extra,
                    "source_audio": source_audio,
                    "GPToAnkiMetadata": rtoml.dumps(note_metadata, pretty=True).replace("\n", "<br>"),
                    },
                "tags": tags,
                "options": {"allowDuplicate": False},
                } for body, note_metadata in zip(bodies, notes_metadata)
            ]


    res = call_anki(
            action="addNotes",
            notes=notes,
            )
    return res


@trace
def add_audio_to_anki(audio_mp3: Union[str, dict], queue: queue.Queue) -> None:
    whi("Sending audio to anki")
    try:
        # get the right path
        if isinstance(audio_mp3, dict):
            test = shared.anki_media / audio_mp3["orig_name"]
            if test.exists():
                audio_mp3 = test
            else:
                audio_mp3 = format_audio_component(audio_mp3)
        else:
            audio_mp3 = format_audio_component(audio_mp3)
        assert Path(audio_mp3).exists(), f"Missing {audio_mp3}"

        # get hash of audio
        with open(audio_mp3, "rb") as audio_file:
            content = audio_file.read()
        audio_hash = hashlib.md5(content).hexdigest()[:10]

        # create the right name
        audio_file_name = str(Path(audio_mp3).name).replace(" ", "_").replace("/", "").replace(".mp3", "")
        # add hash to name only if missing
        if audio_hash in audio_file_name:
            red(f"Audio hash already in filename: {audio_file_name}")
            audio_file_name = f"Voice2Anki_{audio_file_name}.mp3"
        else:
            audio_file_name = f"Voice2Anki_{audio_file_name}_{audio_hash}.mp3"

        # if the name contains a timestamp, move it as the end
        # as it can take a lot of length and anki crops the names sometimes
        sp = [s for s in audio_file_name.split("_") if s.isdigit()]
        for tstamp in sp:
            date = datetime.fromtimestamp(int(tstamp))
            if date.year > 1900 and date.year < 2030:
                audio_file_name = audio_file_name.replace(f"{tstamp}", "")
                audio_file_name = Path(audio_file_name).stem + f"_{tstamp}.mp3"
                audio_file_name = audio_file_name.replace("__", "_")

        audio_path = anki_media / audio_file_name
        if (audio_path).exists():
            red(f"Audio hash already exists! {audio_path}")
        else:
            shutil.copy2(audio_mp3, audio_path)
            assert (audio_path).exists(), "audio file not found in anki media!"

        html = f"[sound:{audio_path.name}]"
        queue.put(html)
    except Exception as err:
        queue.put(red(f"\n\nError when copying audio to anki media: '{err}'"))


@trace
@Timeout(5)
async def get_card_status(txt_chatgpt_cloz: str, return_bool=False) -> Union[str, bool]:
    """return depending on if the card written in
    txt_chatgpt_cloz is already in anki or not"""
    if [f for f in shared.func_dir.iterdir() if f.name.endswith("flashcard_editor.py")]:
        red("Found flashcard_editor.py")
        spec = importlib.util.spec_from_file_location(
                "flashcard_editor.cloze_editor",
                (shared.func_dir / "flashcard_editor.py").absolute()
                )
        editor_module = importlib.util.module_from_spec(spec)
        sys.modules["editor_module"] = editor_module
        spec.loader.exec_module(editor_module)
        cloze_editor = editor_module.cloze_editor
    else:
        red("Not flashcard_editor.py found")
        cloze_editor = lambda x: x

    cloz = cloze_editor(txt_chatgpt_cloz)
    cloz = cloz.replace("\"", "\\\"")

    if not cloz.strip():
        if return_bool:
            return False
        else:
            return "EMPTY"

    if "#####" in cloz:  # multiple cards
        assert return_bool is False, "Unexpected return_bool True"
        splits = [cl.strip() for cl in cloz.split("#####") if cl.strip()]
        vals = await asyncio.gather(*[get_card_status(sp, True) for sp in splits])
        # vals = [get_card_status(sp, True) for sp in splits]

        n = len(vals)
        if all(vals) and all(isinstance(v, bool) for v in vals):
            if return_bool:
                return True
            else:
                return f"Added {n}/{n}"
        else:
            s = sum([bool(b) for b in vals])
            if return_bool:
                return False
            else:
                return f"MISSING {n-s}/{n}"

    else:
        cloz = re.sub(r"{{c\d+::.*?}}", "CLOZCONTENT", cloz).split("CLOZCONTENT")
        cloz = [cl.strip() for cl in cloz if cl.strip()]
        assert cloz
        query = ""
        for cl in cloz:
            query += f" body:\"*{cl}*\""
        query = query.strip()
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(
                None,
                partial(call_anki, action="findCards", query=query)
                )
        # state = call_anki(action="findCards", query=query)
        if state:
            if return_bool:
                return True
            else:
                return "Added"
        else:
            if return_bool:
                return False
            else:
                return "MISSING"


@trace
def sync_anki() -> None:
    "trigger anki synchronization"
    sync_output = call_anki(action="sync")
    assert sync_output is None or sync_output == "None", (
        f"Error during sync?: '{sync_output}'")
    # time.sleep(1)  # wait for sync to finish, just in case


@trace
def threaded_sync_anki() -> None:
    # trigger anki sync
    thread = threading.Thread(target=sync_anki)
    thread.start()


@trace
def mark_previous_note() -> None:
    "add the tag 'marked' to the latest added card."
    if not shared.added_note_ids:
        raise Exception(red("No card ids found."))
    pc = shared.added_note_ids[-1]
    call_anki(
            action="addTags",
            notes=pc,
            tags="marked",
            )


# @trace
def get_anki_tags() -> List[str]:
    try:
        return call_anki(
                action="getTags",
                )
    except Exception as err:
        return [red(f"Error when getting list of anki tags: {err}'")]


# @trace
def get_decks() -> List[str]:
    try:
        return call_anki(
                action="deckNames",
                )
    except Exception as err:
        return [red(f"Error when getting list of anki deck: {err}'")]


# load anki profile using ankipandas just to get the media folder
if shared.anki_media:
    db_path = Path(shared.anki_media)
    red(f"Using anki db_path found in argument: {db_path}")
else:
    try:
        db_path = akp.find_db(user="Main")
    except Exception as err:
        red(f"Exception when trying to find anki collection: '{err}'")
        db_path = akp.Collection().path

red(f"Voice2Anki will use anki collection found at {db_path}")

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
shared.anki_media = anki_media


if __name__ == "__main__":
    print(
            add_note_to_anki(
                body="{{c1::test}}",
                source="test",
                note_metadata="",
                source_extra="",
                source_audio="",
                tags=["test"],
                deck_name="Default"
                )
            )
    breakpoint()
