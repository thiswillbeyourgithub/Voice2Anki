from rapidfuzz.fuzz import ratio as levratio
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
from py_ankiconnect import PyAnkiconnect
from typing import Any

from .logger import red, whi, trace, Timeout
from .shared_module import shared
from .media import format_audio_component
from .typechecker import optional_typecheck


call_anki = PyAnkiconnect(async_mode=False)
async_call_anki = PyAnkiconnect(async_mode=True)

@optional_typecheck
def _request_wrapper(action: str, **params):
    return {'action': action, 'params': params, 'version': 6}

@optional_typecheck
async def anki_request_async(url: str, request: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=request) as response:
            return await response.json()

@optional_typecheck
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
        gr.Warning(red("Anki does not contain a notetype 'Voice2Anki' nor 'Clozolkor', creating it."))
        # create note type model that has the right fields
        call_anki(
                action="createModel",
                modelName="Voice2Anki",
                inOrderFields=["body", "source", "source_extra", "source_audio", "Voice2AnkiMetadata"],
                isCloze=True,
                cardTemplates=[
                    {
                        "Front": "{{cloze:body}}",
                        "Back": "{{cloze:body}}<br><br>{{source}}<br>{{source_extra}}<br>{{source_audio}}",
                        },
                    ],
                )
        gr.Warning(red("Done creating notetype 'Voice2Anki'"))
        shared.anki_notetype = "Voice2Anki"
    whi(f"Anki notetype detected: '{shared.anki_notetype}'")


@optional_typecheck
@trace
def add_note_to_anki(
        bodies: List[str],
        source: str,
        source_extra: str,
        source_audio: str,
        notes_metadata: List[dict],
        tags: List[str],
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
        red("No flashcard_editor.py found")
        cloze_editor = lambda x: x

    notes = [
            {
                "deckName": deck_name,
                "modelName": model_name,
                "fields": {
                    "body": cloze_editor(body.strip().replace("\n", "<br>")),
                    "source": source,
                    "source_extra": source_extra,
                    "source_audio": source_audio,
                    "Voice2AnkiMetadata": rtoml.dumps(note_metadata, pretty=True).replace("\n", "<br>"),
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


@optional_typecheck
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

        # create the right name
        audio_file_name = str(Path(audio_mp3).name).replace(" ", "_").replace("/", "").replace(".mp3", "")

        # # add hash to name only if missing
        # with open(audio_mp3, "rb") as audio_file:
        #     content = audio_file.read()
        # audio_hash = hashlib.md5(content).hexdigest()[:10]
        # if audio_hash in audio_file_name:
        #     red(f"Audio hash already in filename: {audio_file_name}")
        #     audio_file_name = f"Voice2Anki_{audio_file_name}.mp3"
        # else:
        #     audio_file_name = f"Voice2Anki_{audio_file_name}_{audio_hash}.mp3"
        audio_file_name = f"Voice2Anki_{audio_file_name}.mp3"

        # if the name contains a timestamp, move it as the end
        # as it can take a lot of length and anki crops the names sometimes
        sp = [s for s in audio_file_name.split("_") if s.isdigit()]
        for tstamp in sp:
            date = datetime.fromtimestamp(int(tstamp))
            if date.year > 2010 and date.year < 2030:
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
@optional_typecheck
async def get_card_status(txt_chatgpt_cloz: str) -> str:
    """return depending on if the card written in
    txt_chatgpt_cloz is already in anki or not"""
    assert shared.initialized, "Demo not yet launched!"
    if [f for f in shared.func_dir.iterdir() if f.name.endswith("flashcard_editor.py")]:
        spec = importlib.util.spec_from_file_location(
                "flashcard_editor.cloze_editor",
                (shared.func_dir / "flashcard_editor.py").absolute()
                )
        editor_module = importlib.util.module_from_spec(spec)
        sys.modules["editor_module"] = editor_module
        spec.loader.exec_module(editor_module)
        cloze_editor = editor_module.cloze_editor
    else:
        cloze_editor = lambda x: x

    cloz = cloze_editor(txt_chatgpt_cloz)
    cloz = cloz.replace("\"", "\\\"")

    if not cloz.strip():
        return "EMPTY"

    if "#####" in cloz:  # multiple cards
        splits = [cl.strip() for cl in cloz.split("#####") if cl.strip()]
        vals = [await get_card_status(sp) for sp in splits]
        assert "EMPTY" not in vals, f"Found EMPTY in {txt_chatgpt_cloz} that returned {vals}"

        n = len(vals)
        missing = len([v for v in vals if v == "MISSING"])
        present = len([v for v in vals if v == "Added"])
        assert missing + present == n, f"Unmatching length for {txt_chatgpt_cloz} that returned {vals}"
        if missing == 0:
            return f"Added {n}/{n}"
        else:
            return f"MISSING {n-missing}/{n}"
    else:
        cloz = re.sub(r"{{c\d+::.*?}}", "CLOZCONTENT", cloz).split("CLOZCONTENT")
        cloz = [cl.strip() for cl in cloz if cl.strip()]
        assert cloz
        query = "added:2"
        for cl in cloz:
            query += f" body:\"*{cl}*\""
        query = query.strip()
        state = await async_call_anki(action="findNotes", query=query)
        if state:
            return "Added"
        else:
            recent = await async_call_anki(action="findNotes", query="added:2")
            if not recent:
                return "MISSING"
            bodies = get_anki_content(nid=recent)
            if txt_chatgpt_cloz in bodies:
                return "Added"
            if all(c in bodies for c in cloz):
                return "Added"
            if all(c in [cloze_editor(b) for b in bodies] for c in cloz):
                return "Added"
            for b in bodies:
                if levratio(txt_chatgpt_cloz, b) > 95:
                    return "Added"
                if levratio(txt_chatgpt_cloz, cloze_editor(b)) > 95:
                    return "Added"
            return "MISSING"


@optional_typecheck
@trace
async def sync_anki() -> None:
    "trigger anki synchronization"
    sync_output = await async_call_anki(action="sync")
    assert sync_output is None or sync_output == "None", (
        f"Error during sync?: '{sync_output}'")
    # time.sleep(1)  # wait for sync to finish, just in case


@optional_typecheck
@trace
async def mark_previous_note() -> None:
    "add or remove the tag 'marked' to the latest added notes."
    if not shared.added_note_ids:
        raise Exception(red("No note ids found."))
    nids = shared.added_note_ids[-1]
    assert nids

    bodies = get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)

    current = [await async_call_anki(
        action="getNoteTags",
        note=int(n)) for n in nids]

    if all("marked" in t for t in current):
        await async_call_anki(
                action="removeTags",
                notes=nids,
                tags="marked",
                )
        gr.Warning(red(f"Note ids were already marked. Unmarking them.\nBodies:\n{bodies}"))
    else:
        await async_call_anki(
                action="addTags",
                notes=nids,
                tags="marked",
                )
        gr.Warning(red(f"Marked anki notes: {','.join([str(n) for n in nids])}\nBodies:\n{bodies}"))

@optional_typecheck
@trace
async def add_to_more_of_previous_note(more_content: str) -> None:
    "add or remove the tag 'marked' to the latest added notes."
    if not shared.added_note_ids:
        raise Exception(red("No note ids found."))
    nids = shared.added_note_ids[-1]
    assert nids
    more_content = more_content.strip()
    assert more_content

    bodies = get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)

    for nid in nids:
        await async_call_anki(
            action="updateNoteFields",
            note={
                "id": nid,
                "fields": {"more": more_content},
            }
        )
    gr.Warning(red(f"Edited anki notes 'more' field: {','.join([str(n) for n in nids])}\nBodies:\n{bodies}"))

@optional_typecheck
@trace
async def suspend_previous_notes() -> None:
    "suspend the latest added notes."
    if not shared.added_note_ids:
        raise Exception(red("No note ids found."))
    nids = shared.added_note_ids[-1]
    assert nids
    s_nids = [str(n) for n in nids]
    cids = await async_call_anki(
            action="findCards",
            query="nid:" + ",".join(s_nids),
            )
    assert cids, "No card ids found for the given note ids: {','.join(s_nids)}"
    status = await async_call_anki(
            action="areSuspended",
            cards=cids,
            )

    bodies = get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)
    if all(status):
        gr.Warning(red(f"Previous notes already suspended so UNsuspending them:\n{bodies}"))
        out = await async_call_anki(
                action="unsuspend",
                cards=cids)
        assert not any(
                await async_call_anki(
                    action="areSuspended",
                    cards=cids,
                    )
                ), f"Cards failed to unsuspend?"
    else:
        gr.Warning(red(f"Suspending notes nid:{','.join(s_nids)} with bodies:\n{bodies}"))
        out = await async_call_anki(
                action="suspend",
                cards=cids)
        assert out, f"Unexpected result from anki: {out} for nids {','.join(s_nids)}"
        assert all(
                await async_call_anki(
                    action="areSuspended",
                    cards=cids,
                    )
                ), "cards failed to suspend?"


@optional_typecheck
def get_anki_content(nid: List[Union[str, int]]) -> str:
    "retrieve the content of the body of an anki note"
    nid = [int(n) for n in nid]
    infos = call_anki(
            action="notesInfo",
            notes=nid,
            )
    if all("body" in inf["fields"] for inf in infos):
        return [inf["fields"]["body"]["value"] for inf in infos]
    else:
        fields = infos[0]["fields"]
        first_field = [f for f in fields if f["order"]==0]
        return [inf["fields"][first_field]["value"] for inf in infos]


# @trace
@optional_typecheck
def get_anki_tags() -> List[str]:
    try:
        return call_anki(
                action="getTags",
                )
    except Exception as err:
        return [red(f"Error when getting list of anki tags: {err}'")]


# @trace
@optional_typecheck
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
