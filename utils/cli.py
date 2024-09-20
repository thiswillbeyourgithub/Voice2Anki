import gradio as gr
import pyclip
import time
import asyncio
import re
from pathlib import Path, PosixPath
from tqdm import tqdm
from dataclasses import MISSING
from typing import Union, Optional

from utils.profiles import ValueStorage
from utils.typechecker import optional_typecheck, beartype
from utils.logger import whi, yel, red, very_high_vis, high_vis
from utils.shared_module import shared
from utils.main import thread_whisp_then_llm, dirload_splitted, dirload_splitted_last, transcribe, alfred, to_anki
from utils.anki_utils import call_anki, get_decks, get_card_status, mark_previous_note, suspend_previous_notes, add_to_more_of_previous_note
from utils.media import get_image

vhv = very_high_vis
hv = high_vis


@optional_typecheck
class Cli:
    def __init__(
        self,
        audio_regex: str,
        txt_chatgpt_context: str,
        txt_deck: str,
        txt_tags: str,
        nb_audio_slots: Union[str, int],

        profile: Optional[str] = MISSING,
        txt_whisp_prompt: Optional[str] = MISSING,
        txt_whisp_lang: Optional[str] = MISSING,
        stt_choice: Optional[str] = MISSING,

        untouched_dir: Optional[Union[PosixPath, str]] = None,
        splitted_dir: Optional[Union[PosixPath, str]] = None,
        done_dir: Optional[Union[PosixPath, str]] = None,
        **kwargs,
    ) -> None:
        whi("Starting CLI")

        assert txt_deck in get_decks(), "Deck not found"
        audio_regex = re.compile(audio_regex)
        txt_tags = txt_tags.split(" ")
        assert txt_tags, "Missing txt_tags after splitting by a space"
        txt_tags += "VoiceToAnki::cli_mode"

        if profile is MISSING:
            profile = "latest"
        # shared.pv.__class__._instance = None  # otherwise it forbids creating new instances
        shared.reset(request=None)
        ValueStorage.__init__(shared.pv, profile)
        if txt_whisp_prompt:
            txt_whisp_prompt = shared.pv["txt_whisp_prompt"]
        if txt_whisp_lang:
            txt_whisp_lang = shared.pv["txt_whisp_lang"]
        if stt_choice:
            stt_choice = shared.pv["stt_choice"]

        # force preprocessing of sound
        red("Forcing sound preprocessing")
        shared.preprocess_sox_effects = shared.force_preprocess_sox_effects

        untouched_dir = shared.pv.p / "queues" / "audio_untouched"
        splitted_dir = shared.pv.p / "queues" / "audio_splits"
        done_dir = shared.pv.p / "queues" / "audio_done"

        assert untouched_dir.exists(), untouched_dir
        assert splitted_dir.exists(), splitted_dir
        assert done_dir.exists(), done_dir
        shared.unsplitted_dir = untouched_dir
        shared.done_dir = done_dir
        shared.splitted_dir = splitted_dir

        audio_todo = [f for f in splitted_dir.iterdir()]
        whi(f"Found {len(audio_todo)} audio splits before filtering")
        audio_todo = [f for f in audio_todo if audio_regex.match(f.name)]
        audio_todo = sorted(audio_todo, key=lambda f: f.stat().st_ctime)
        assert audio_todo, "No audio todo"
        whi(f"Found {len(audio_todo)} audio splits to do")

        if nb_audio_slots == "auto":
            nb_audio_slots = len(audio_todo)
            shared.audio_slot_nb = nb_audio_slots

        if "gallery_pdf" in kwargs:
            # load just the pathnames but might maybe cause issue with ocr caching?:
            # temp_gal = [f.absolute().resolve().__str__() for f in Path(kwargs["gallery_pdf"]).iterdir()]
            # load the image directly
            try:
                old = pyclip.paste()
            except Exception:
                old = MISSING
            pyclip.copy(kwargs["gallery_pdf"])
            temp_gal = get_image(None)
            if old is not MISSING:
                pyclip.copy(old)
            gallery = gr.Gallery(temp_gal).value
            assert gallery
            vhv("Loaded gallery")
        else:
            gallery = None

        audio_slots = dirload_splitted(True, [None] * nb_audio_slots)

        time.sleep(1)
        vhv("Done dirloading, press <enter> to continue")
        input()

        for audio in tqdm(audio_todo, unit="audio"):
            row = shared.dirload_queue.loc[audio.__str__(), :]
            assert not row.empty, f"Empty row: {row}"
            temp_path = row["temp_path"]
            hv(f"Audio file: {temp_path}")
            text = transcribe(temp_path)
            hv(f"Transcript:\n{text}")
            # for some reason using kwargs don't work
            # cloze = alfred(
            #     txt_audio=text,
            #     txt_chatgpt_context=shared.pv["txt_chatgpt_context"],
            #     profile=shared.pv.profile_name,
            #     max_token=shared.pv["sld_max_tkn"],
            #     temperature=shared.pv["sld_temp"],
            #     sld_buffer=shared.pv["sld_buffer"],
            #     llm_choice=shared.pv["llm_choice"],
            #     txt_keywords=shared.pv["txt_keywords"],
            #     prompt_manag=shared.pv["prompt_management"],
            #     cache_mode=False,
            # )
            cloze = alfred(
                text,
                shared.pv["txt_chatgpt_context"],
                shared.pv.profile_name,
                shared.pv["sld_max_tkn"],
                shared.pv["sld_temp"],
                shared.pv["sld_buffer"],
                shared.pv["llm_choice"],
                shared.pv["txt_keywords"],
                shared.pv["prompt_management"],
                False,
            )
            hv(f"Cloze:\n{cloze}")
            status = asyncio.run(get_card_status(cloze))
            vhv(f"Status:\n{status}")

            if status == "MISSING":
                vhv("Enter to proceed, 'debug' to breakpoint, anything else to quit")
                ans = input().lower()
                if ans.startswith("debug"):
                    breakpoint()
                if ans:
                    vhv("Quitting")

            try:
                out = to_anki(
                    audio_mp3_1=temp_path,
                    txt_audio=text,
                    txt_chatgpt_cloz=cloze,
                    txt_chatgpt_context=shared.pv["txt_chatgpt_context"],
                    txt_deck=shared.pv["txt_deck"],
                    txt_tags=shared.pv["txt_tags"],
                    gallery=gallery,
                    check_marked=False,
                    txt_extra_source=None,
                )
                if out is None:
                    out = "Success"
            except Exception as e:
                if "cannot create note because it is a duplicate" in str(e).lower():
                    out = "DUPLICATE"

            vhv(f"Output:\n{out}")

            audio_slots = dirload_splitted_last(True)

            while True:
                vhv("What next? [s(uspend previous) - m(ark previous) - a(add to more) - d(ebug)]\nEnter to roll to the next audio")
                ans = input().lower()
                if not ans:
                    vhv("Continuing to next audio")
                    break
                elif ans.startswith("d"):
                    vhv("Opening debugger then exiting")
                    breakpoint()
                    raise SystemExit(1)
                elif ans.startswith("s"):
                    try:
                        out = asyncio.run(suspend_previous_notes())
                        vhv(f"Suspended previous note: {out}")
                    except Exception as e:
                        vhv(f"Error when suspending: {e}")
                elif ans.startswith("m"):
                    try:
                        out = asyncio.run(mark_previous_note())
                        vhv(f"Marked previous note: {out}")
                    except Exception as e:
                        vhv(f"Error when marking: {e}")
                elif ans.startswith("a"):
                    more = input("Enter the content you want to add to the More field:\n>")
                    try:
                        asyncio.run(add_to_more_of_previous_note(more))
                        vhv("Added to 'More' field of the previous notes")
                    except Exception as e:
                        vhv(f"Error when adding to more: {e}")
                else:
                    vhv("Unexpected answer.")

        vhv("Done with that batch!\nOpening debugger just in case:")
        breakpoint()
