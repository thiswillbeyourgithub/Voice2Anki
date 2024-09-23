import gradio as gr
import pyclip
import time
import asyncio
import re
from pathlib import Path, PosixPath
from tqdm import tqdm
from dataclasses import MISSING
from typing import Union, Optional
from pydub import AudioSegment

from utils.profiles import ValueStorage
from utils.typechecker import optional_typecheck, beartype
from utils.logger import whi, yel, red, very_high_vis, high_vis
from utils.shared_module import shared
from utils.main import thread_whisp_then_llm, dirload_splitted, dirload_splitted_last, transcribe, alfred, to_anki, flag_audio
from utils.anki_utils import call_anki, get_decks, get_card_status, mark_previous_note, suspend_previous_notes, add_to_more_of_previous_note
from utils.media import get_image, roll_audio, force_sound_processing

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
        unsupervised: int = 0,
        **kwargs,
    ) -> None:
        whi("Starting CLI")

        assert txt_deck in get_decks(), "Deck not found"
        audio_regex = re.compile(audio_regex)
        txt_tags = txt_tags.split(" ")
        assert txt_tags, "Missing txt_tags after splitting by a space"
        txt_tags += ["VoiceToAnki::cli_mode"]

        if unsupervised:
            vhv(f"Activating 'unsupervised' mode: {unsupervised}")
            txt_tags += [f"VoiceToAnki::unsupervised::{unsupervised}"]

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

        shared.pv["sld_max_tkn"] = 2500

        audio_todo = [f for f in splitted_dir.iterdir()]
        whi(f"Found {len(audio_todo)} audio splits before filtering")
        audio_todo = [f for f in audio_todo if audio_regex.match(f.name)]
        # audio_todo = sorted(audio_todo, key=lambda f: f.stat().st_ctime)
        # fix: sort by name actualy because force processing would move the audio at the end
        audio_todo = sorted(audio_todo, key=lambda f:f.name)
        assert audio_todo, "No audio todo"
        whi(f"Found {len(audio_todo)} audio splits to do")

        if nb_audio_slots == "auto":
            nb_audio_slots = len(audio_todo)
            shared.audio_slot_nb = nb_audio_slots
        shared.pv["txt_deck"] = txt_deck
        shared.pv["txt_tags"] = txt_tags


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

        shared.pv["gallery"] = gallery

        audio_slots = dirload_splitted(True, [None] * nb_audio_slots)

        time.sleep(1)
        if not unsupervised:
            vhv("Done dirloading, press <enter> to continue")
            input()
        else:
            vhv("Done dirloading")

        for audio in tqdm(audio_todo, unit="audio"):
            shared.pv["txt_chatgpt_context"] = txt_chatgpt_context

            row = shared.dirload_queue.loc[audio.__str__(), :]
            assert not row.empty, f"Empty row: {row}"
            temp_path = row["temp_path"]
            hv(f"Audio file: {temp_path}")
            text = transcribe(temp_path)
            vhv("=" * 10)
            hv(f"Transcript:\n{text}")
            vhv("=" * 10)
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
                txt_chatgpt_context,
                shared.pv.profile_name,
                shared.pv["sld_max_tkn"],
                shared.pv["sld_temp"],
                shared.pv["sld_buffer"],
                shared.pv["llm_choice"],
                shared.pv["txt_keywords"],
                shared.pv["prompt_management"],
                False,
            )
            vhv("=" * 10)
            vhv("Cloze:")
            hv(cloze)
            trial = 1
            n_trial = 3
            while True:
                try:
                    status = asyncio.run(get_card_status(cloze))
                    break
                except Exception as e:
                    vhv(f"Error when getting status for {trial} time: {e}")
                    trial += 1
                    if trial >= n_trial:
                        raise e
                    loaded_audio = AudioSegment.from_mp3(audio)
                    duration = len(loaded_audio)/1000
                    if duration < 10:
                        raise Exception(f"Audio is surprisingly short: {duration}\nError: ") from e
                    vhv(f"Duration of the current audio: {duration:.2f}s")
                    audio2 = force_sound_processing(temp_path)
                    assert audio2.stat() != audio.stat()
                    assert audio2.absolute().resolve().__str__() == temp_path
                    loaded_audio = AudioSegment.from_mp3(audio2)
                    duration2 = len(loaded_audio)/1000
                    vhv(f"Duration of the processed audio: {duration2:.2f}s")
                    assert duration2 <= duration, f"Unexpected duration: {duration2} vs {duration}"
                    text2 = transcribe(audio2)
                    assert text2 != text, "Force processing did not change the text"
                    cloze2 = alfred(
                        text2,
                        txt_chatgpt_context,
                        shared.pv.profile_name,
                        shared.pv["sld_max_tkn"],
                        shared.pv["sld_temp"],
                        shared.pv["sld_buffer"],
                        shared.pv["llm_choice"],
                        shared.pv["txt_keywords"],
                        shared.pv["prompt_management"],
                        False,
                    )
                    assert cloze2 != cloze, "Force processing did not change the cloze"
                    cloze = cloze2

                vhv(f"Status:\n{status}")
                vhv("=" * 10)

            if not unsupervised:
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
                    txt_chatgpt_context=txt_chatgpt_context,
                    txt_deck=shared.pv["txt_deck"],
                    txt_tags=shared.pv["txt_tags"],
                    gallery=shared.pv["gallery"],
                    check_marked=False,
                    txt_extra_source=None,
                )
                if out is None:
                    out = "Success"
            except Exception as e:
                if "cannot create note because it is a duplicate" in str(e).lower():
                    out = "DUPLICATE"
                else:
                    out = str(e)

            vhv(f"Output:\n{out}")

            vhv("Rolling")
            if not isinstance(audio_slots[0], dict):
                vhv("Popping first element: ")
                hv(str(audio_slots.pop(0)))
            audio_slots = [gr.Audio(a["value"]).value for a in audio_slots]
            audio_slots = roll_audio(*audio_slots)

            vhv("Loading next audio")
            audio_slots[-1] = dirload_splitted_last(True)
            assert len(audio_slots) == shared.audio_slot_nb

            while True:
                vhv("What next?")
                if unsupervised:
                    hv("Unsupervised so continuing directly after 5s")
                    time.sleep(5)
                    break
                hv("[s(uspend previous) - m(ark previous) - a(dd to more) - d(ebug) - f(lag)]\nEnter to roll to the next audio")
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
                elif ans.startswith("f"):
                    try:
                        flag_audio(
                            txt_profile=shared.pv.profile_name,
                            txt_audio=text,
                            txt_whisp_lang=shared.pv["txt_whisp_lang"],
                            txt_whisp_prompt=shared.pv["txt_whisp_prompt"],
                            txt_chatgpt_cloz=cloze,
                            txt_chatgpt_context=txt_chatgpt_context,
                            gallery=shared.pv["gallery"],
                        )
                        vhv("Flagged audio")
                    except Exception as e:
                        vhv(f"Error flagging audio: {e}")
                else:
                    vhv("Unexpected answer.")

        vhv("Done with that batch!")
