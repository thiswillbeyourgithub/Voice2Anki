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
except:
    def cloze_editor(cloze):
        return cloze


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


def _call_anki(action, **params):
    """ bridge between local python libraries and AnnA Companion addon
    (a fork from anki-connect) """
    def request_wrapper(action, **params):
        return {'action': action, 'params': params, 'version': 6}

    requestJson = json.dumps(request_wrapper(action, **params)
                             ).encode('utf-8')

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
        audio_hash = hashlib.md5(content).hexdigest()
        audio_file_name = str(Path(audio_mp3).name).replace(" ", "_").replace("/", "").replace(".mp3", "")
        audio_path = anki_media / f"Voice2FormattedText_{audio_hash}_{audio_file_name}.mp3"
        if (audio_path).exists():
            red(f"Audio hash already exists! {audio_path}")
        else:
            shutil.copy2(audio_mp3, audio_path)
            assert (audio_path).exists(), "audio file not found in anki media!"

        html = f"</br>[sound:{audio_path.name}]"
        queue.put(html)
    except Exception as err:
        queue.put(red(f"\n\nError when copying audio to anki media: '{err}'"))


@Timeout(10)
def get_card_status(txt_chatgpt_cloz):
    """return True or False depending on if the card written in
    txt_chatgpt_cloz is already in anki or not"""

    cloz = cloze_editor(txt_chatgpt_cloz)
    cloz = re.sub(r"{{c\d+::.*?}}", "", cloz).strip()

    if not cloz:
        return gr.Button("EMPTY", variant="primary")

    if "#####" in cloz:  # multiple cards
        splits = [cl.strip() for cl in cloz.split("#####") if cl.strip()]
        vals = []
        for sp in splits:
            cloz = cloze_editor(sp)
            cloz = re.sub(r"{{c\d+::.*?}}", "", cloz).strip()
            cloz = cloz.replace("\"", "\\\"")
            val = _call_anki(action="findCards", query=f"added:7 body:\"*{cloz}*\"")
            vals.append(bool(val))

        if all(vals):
            return gr.Button("DONE")
        else:
            s = sum([bool(b) for b in vals])
            n = len(vals)
            return gr.Button(f"MISSING {n-s}/{n}", variant="primary")

    else:
        cloz = cloz.replace("\"", "\\\"")
        query = f"added:7 body:\"*{cloz}*\""
        state = _call_anki(
                action="findCards",
                query=query,
                )
        if state:
            return gr.Button("DONE")
        else:
            return gr.Button("MISSING", variant="primary")


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
