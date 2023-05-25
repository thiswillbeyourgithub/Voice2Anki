import hashlib
from pathlib import Path
import ankipandas as akp
import shutil
import time
import urllib.request
import json
from .logger import red, whi


def add_to_anki(
        body,
        source,
        note_metadata,
        tags,
        deck_name="Default",
        ):
    """create a new cloze directly in anki"""
    assert isinstance(tags, list), "tags have to be a list"
    return _call_anki(
            action="addNote",
            note={
                "deckName": deck_name,
                "modelName": "Clozolkor",
                "fields": {
                    "body": body,
                    "header": "",
                    "hint": "",
                    "more": "",
                    "source": source,
                    "source_extra": "",
                    "teacher": "",
                    "Nearest_neighbors": "",
                    "GPToAnkiMetadata": note_metadata,
                    },
                "tags": tags,
                "options": {"allowDuplicate": True},
               },
            )


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


def audio_to_anki(audio_path):
    whi("Sending audio to anki")
    try:
        with open(audio_path, "rb") as audio_file:
            audio_hash = hashlib.md5(audio_file.read()).hexdigest()
        shutil.copy(audio_path, anki_media / f"{audio_hash}.wav")
        html = f"</br>[sound:{audio_hash}.wav]"
        return html
    except Exception as err:
        return red(f"\n\nError when copying audio to anki media: '{err}'")


def sync_anki():
    "trigger anki synchronization"
    sync_output = _call_anki(action="sync")
    assert sync_output is None or sync_output == "None", (
        f"Error during sync?: '{sync_output}'")
    time.sleep(1)  # wait for sync to finish, just in case
    whi("Done syncing!")

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
