import asyncio
from rapidfuzz.fuzz import ratio as levratio
import queue
from typing import List, Union, Optional, Callable, Coroutine
from datetime import datetime
import rtoml
import sys
import importlib.util
import aiohttp
import gradio as gr
import re
import shutil
from pathlib import Path, PosixPath
import ankipandas as akp
import time
from py_ankiconnect import PyAnkiconnect
from functools import cache
from cache import AsyncTTL

from dataclasses import MISSING

from .logger import red, whi, trace, Timeout
from .shared_module import shared
from .media import format_audio_component
from .typechecker import optional_typecheck
from .memory import split_thinking


# PyAnkiconnect automatically use async if called from async
call_anki = PyAnkiconnect(force_async_mode=False)


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
        ) -> Optional[List[Union[int, str]]]:
    """create a new cloze directly in anki"""
    assert isinstance(tags, list), "tags have to be a list"
    if not shared.anki_notetype:
        check_anki_models()
    model_name = shared.anki_notetype

    cloze_ed_file = shared.func_dir / "flashcard_editor.py"
    if cloze_ed_file.exists():
        ctime = cloze_ed_file.stat().st_ctime
        cloze_editor = cached_load_flashcard_editor(path=cloze_ed_file, ctime=ctime)
    else:
        red("No flashcard_editor.py found")
        def cloze_editor(x: str) -> str:
            return x

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
@cache
@optional_typecheck
def cached_load_flashcard_editor(path: PosixPath, ctime: float) -> Callable:
    spec = importlib.util.spec_from_file_location(
            "flashcard_editor.cloze_editor",
            (path).absolute()
            )
    editor_module = importlib.util.module_from_spec(spec)
    sys.modules["editor_module"] = editor_module
    spec.loader.exec_module(editor_module)
    cloze_editor = editor_module.cloze_editor
    return cloze_editor

@trace
@optional_typecheck
@AsyncTTL(maxsize=10000, time_to_live=60)
async def cached_get_anki_content(nid: Union[int, str]) -> Coroutine:
    "TTL cached that expire after n seconds to get the content of a single anki note"
    return await get_anki_content([nid])

def remove_markers(intext: str) -> str:
    return intext.replace("{{c", "").replace("}}", "").replace("::", "")

@trace
@Timeout(5)
@optional_typecheck
async def get_card_status(txt_chatgpt_cloz: str) -> str:
    """return depending on if the card written in
    txt_chatgpt_cloz is already in anki or not"""
    assert shared.initialized, "Demo not yet launched!"
    if txt_chatgpt_cloz.startswith("Too few words in txt_audio to be plausible "):
        return "null"
    if txt_chatgpt_cloz.startswith("Image change detected: '"):
        return "null"
    if "{{c1::" not in txt_chatgpt_cloz and "}}" not in txt_chatgpt_cloz:
        return "null"
    txt_chatgpt_cloz = split_thinking(txt_chatgpt_cloz)[0]
    cloze_ed_file = shared.func_dir / "flashcard_editor.py"
    if cloze_ed_file.exists():
        ctime = cloze_ed_file.stat().st_ctime
        cloze_editor = cached_load_flashcard_editor(path=cloze_ed_file, ctime=ctime)
    else:
        def cloze_editor(x: str) -> str:
            return x

    cloz = cloze_editor(txt_chatgpt_cloz)
    cloz = cloz.replace("\"", "\\\"")

    if not cloz.strip():
        return "EMPTY"

    if "#####" in cloz:  # multiple cards
        splits = [cl.strip() for cl in cloz.split("#####") if cl.strip()]
        # vals = [await get_card_status(sp) for sp in splits]
        tasks = [get_card_status(sp) for sp in splits]
        vals = await asyncio.gather(*tasks)
        assert "EMPTY" not in vals, f"Found EMPTY in {txt_chatgpt_cloz} that returned {vals}"

        n = len(vals)
        missing = len([v for v in vals if "missing" in v.lower()])
        present = len([v for v in vals if "added" in v.lower()])
        assert missing + present == n, f"Unmatching length for {txt_chatgpt_cloz} that returned {vals}"
        if missing == 0:
            return f"Added {n}/{n}"
        else:
            return f"MISSING {n-missing}/{n}"
    else:
        cloz = re.sub(r"{{c\d+::.*?}}", "CLOZCONTENT", cloz).split("CLOZCONTENT")
        cloz = [cl.strip() for cl in cloz if cl.strip()]
        assert cloz, f"cloz issue: {cloz}"
        query = "added:2"
        for cl in cloz:
            query += f" body:\"*{cl}*\""
        query = query.strip()
        state = await call_anki(action="findNotes", query=query)
        if state:
            return "Added"
        recent = await call_anki(action="findNotes", query="added:2")
        if not recent:
            return "MISSING"
        recent = [int(n) for n in recent]
        recent = sorted(recent, reverse=True)  # sort to get the most recent first
        # bodies = await get_anki_content(nid=recent)
        # tasks = [cached_get_anki_content(n) for n in recent]
        # bodies = await asyncio.gather(*tasks)

        semaphore = asyncio.Semaphore(10)
        async def limited_process(item):
            async with semaphore:
                return await cached_get_anki_content(item)
        tasks = {r: asyncio.create_task(limited_process(r)) for r in recent}
        bodies = [MISSING] * len(recent)
        should_return = False
        txt_nomarkers = remove_markers(txt_chatgpt_cloz)
        for r, task in tasks.items():
            if should_return:
                task.cancel()
                continue
            body = await task
            body = body[0]
            assert body is not MISSING
            assert bodies[recent.index(r)] is MISSING
            bodies[recent.index(r)] = body

            body2 = cloze_editor(body)
            body3 = remove_markers(body)
            body4 = remove_markers(body2)
            subbodies = [body, body2, body3, body4]

            for ib, b in enumerate(subbodies):
                if txt_chatgpt_cloz in b:
                    should_return = "Added"
                    break
                elif all(c in b for c in cloz):
                    should_return = "Added"
                    break
                elif all(c in b for c in cloz):
                    should_return = "Added"
                    break
                elif levratio(txt_chatgpt_cloz, b) > 90:
                    should_return = "Prob added " + (ib + 1)
                    break
                elif levratio(txt_nomarkers, b) > 90:
                    should_return = "Prob added " + (ib + 1 + len(subbodies))
                    break
        assert MISSING not in bodies
        if should_return:
            return should_return
        return "MISSING"


@optional_typecheck
@trace
async def sync_anki() -> None:
    "trigger anki synchronization"
    sync_output = await call_anki(action="sync")
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
    assert nids, "empty nids"

    bodies = await get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)

    current = [await call_anki(
        action="getNoteTags",
        note=int(n)) for n in nids]

    if all("marked" in t for t in current):
        await call_anki(
                action="removeTags",
                notes=nids,
                tags="marked",
                )
        gr.Warning(red(f"Note ids were already marked. Unmarking them.\nBodies:\n{bodies}"))
    else:
        await call_anki(
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
    assert nids, "empty nids"
    more_content = more_content.strip()
    assert more_content, "empty more_content"

    bodies = await get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)

    for nid in nids:
        await call_anki(
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
    assert nids, "empty nids"
    s_nids = [str(n) for n in nids]
    cids = await call_anki(
            action="findCards",
            query="nid:" + ",".join(s_nids),
            )
    assert cids, "No card ids found for the given note ids: {','.join(s_nids)}"
    status = await call_anki(
            action="areSuspended",
            cards=cids,
            )

    bodies = await get_anki_content(nid=nids)
    bodies = "* " + "\n* ".join(bodies)
    if all(status):
        gr.Warning(red(f"Previous notes already suspended so UNsuspending them:\n{bodies}"))
        out = await call_anki(
                action="unsuspend",
                cards=cids)
        assert not any(
                await call_anki(
                    action="areSuspended",
                    cards=cids,
                    )
                ), f"Cards failed to unsuspend?"
    else:
        gr.Warning(red(f"Suspending notes nid:{','.join(s_nids)} with bodies:\n{bodies}"))
        out = await call_anki(
                action="suspend",
                cards=cids)
        assert out, f"Unexpected result from anki: {out} for nids {','.join(s_nids)}"
        assert all(
                await call_anki(
                    action="areSuspended",
                    cards=cids,
                    )
                ), "cards failed to suspend?"


@optional_typecheck
async def get_anki_content(nid: List[Union[str, int]]) -> List[str]:
    "retrieve the content of the body of an anki note"
    nid = [int(n) for n in nid]
    infos = await call_anki(
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
    shared.anki_media = Path(shared.anki_media)
    assert shared.anki_media.is_dir(), f"shared.anki_media must be a dict. Path: {shared.anki_media}"
    if shared.anki_media.name != "collection.media":
        candidates = [f for f in shared.anki_media.iterdir() if f.is_dir() and f.name == "collection.media"]
        assert candidates, f"No collection.media folder found at {shared.anki_media}"
        assert len(candidates) == 1, f"Found multiple candidates for media folder: {candidates}"
        shared.anki_media = candidates[0]
        assert shared.anki_media, "anki_media not set?"

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
assert shared.anki_media.name == "collection.media"


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
