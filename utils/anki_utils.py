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

try:
    from .author_cloze_editor import cloze_editor
except Exception:
    def cloze_editor(cloze):
        return cloze


def _call_anki(action, **params):
    """ bridge between local python libraries and AnnA Companion addon
    (a fork from anki-connect) """

    if "async_mode" in params:
        use_async = True
        del params["async_mode"]
    else:
        use_async = False

    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)
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
def add_to_anki(
        body,
        source,
        note_metadata,
        tags,
        deck_name="Default",
        ):
    """create a new cloze directly in anki"""
    assert isinstance(tags, list), "tags have to be a list"
    model_name = False
    other_fields = {}

    body = cloze_editor(body)

    if "Clozolkor" in _call_anki(action="modelNames"):
        model_name = "Clozolkor"
        other_fields = {
                "header": "",
                "hint": "",
                "more": "",
                "source_extra": "",
                "teacher": "",
                "Nearest_neighbors": "",
                }
    else:
        if "WhisperToAnki" in _call_anki(action="modelNames"):
            model_name = "WhisperToAnki"
    if model_name:
        res = _call_anki(
                action="addNote",
                note={
                    "deckName": deck_name,
                    "modelName": model_name,
                    "fields": {
                        "body": body,
                        "source": source,
                        "GPToAnkiMetadata": note_metadata,
                        **other_fields,
                        },
                    "tags": tags,
                    "options": {"allowDuplicate": False}})
        return res
    else:
        # create note type model that has the right fields
        red("No notetype WhisperToAnki nor Clozolkor found, creating WhisperToAnki")
        res = _call_anki(
                action="createModel",
                modelName="WhisperToAnki",
                inOrderFields=["body", "source", "GPToAnkiMetadata"],
                isCloze=True,
                cardTemplates=[
                    {
                        "Front": "{{cloze:body}}",
                        "Back": "{{cloze:body}}<br><br>{{source}}",
                        },
                    ],
                )
        red("Done creating notetype")
        return add_to_anki(body, source, note_metadata, tags, deck_name)

@trace
async def anki_request_async(url, request):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=request) as response:
            data = await response.json()
            return data

@trace
def audio_to_anki(audio_mp3, queue):
    whi("Sending audio to anki")
    try:
        audio_mp3 = format_audio_component(audio_mp3)
        if not Path(audio_mp3).exists():
            red(f"File {audio_mp3} not found, looking for the right file")
            if (Path(audio_mp3).parent.parent / Path(audio_mp3).name).exists():
                audio_mp3 = (Path(audio_mp3).parent.parent / Path(audio_mp3).name).absolute()
                red(f"Right file found: '{audio_mp3}'")
            else:
                candidates = [str(p) for p in Path(audio_mp3).parent.iterdir()]
                if len(candidates) != 1:
                    raise Exception(f"Multiple candidate mp3 file: '{candidates}'")
                else:
                    audio_mp3 = candidates[0]
                    red(f"Right file found: '{audio_mp3}'")

        with open(audio_mp3, "rb") as audio_file:
            content = audio_file.read()
        audio_hash = hashlib.md5(content).hexdigest()[:10]
        audio_file_name = str(Path(audio_mp3).name).replace(" ", "_").replace("/", "").replace(".mp3", "")
        audio_path = anki_media / f"Voice2Anki_{audio_file_name}_{audio_hash}.mp3"
        if (audio_path).exists():
            red(f"Audio hash already exists! {audio_path}")
        else:
            shutil.copy2(audio_mp3, audio_path)
            assert (audio_path).exists(), "audio file not found in anki media!"

        html = f"</br>[sound:{audio_path.name}]"
        queue.put(html)
    except Exception as err:
        queue.put(red(f"\n\nError when copying audio to anki media: '{err}'"))


@trace
@Timeout(5)
async def get_card_status(txt_chatgpt_cloz, return_bool=False):
    """return depending on if the card written in
    txt_chatgpt_cloz is already in anki or not"""

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
                partial(_call_anki, action="findCards", query=query)
                )
        # state = _call_anki(action="findCards", query=query)
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
def sync_anki():
    "trigger anki synchronization"
    sync_output = _call_anki(action="sync")
    assert sync_output is None or sync_output == "None", (
        f"Error during sync?: '{sync_output}'")
    # time.sleep(1)  # wait for sync to finish, just in case


@trace
def threaded_sync_anki():
    # trigger anki sync
    thread = threading.Thread(target=sync_anki)
    thread.start()


@trace
def mark_previous_note():
    "add the tag 'marked' to the latest added card."
    if not shared.added_note_ids:
        raise Exception(red("No card ids found."))
    pc = shared.added_note_ids[-1]
    _call_anki(
            action="addTags",
            notes=pc,
            tags="marked",
            )


# @trace
def get_anki_tags():
    try:
        return _call_anki(
                action="getTags",
                )
    except Exception as err:
        return [red(f"Error when getting list of anki tags: {err}'")]


# @trace
def get_decks():
    try:
        return _call_anki(
                action="deckNames",
                )
    except Exception as err:
        return [red(f"Error when getting list of anki deck: {err}'")]


# load anki profile using ankipandas just to get the media folder
if shared.media_folder:
    db_path = Path(shared.media_folder)
    red(f"Using anki db_path found in argument: {db_path}")
else:
    try:
        db_path = akp.find_db(user="Main")
    except Exception as err:
        red(f"Exception when trying to find anki collection: '{err}'")
        db_path = akp.Collection().path

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
shared.anki_media = anki_media


if __name__ == "__main__":
    print(
            add_to_anki(
                body="{{c1::test}}",
                source="truc",
                note_metadata="",
                tags=["test"],
                deck_name="Default"
                )
            )
    breakpoint()
