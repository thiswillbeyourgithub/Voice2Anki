import re
from pathlib import Path, PosixPath
from tqdm import tqdm
from dataclasses import MISSING
from typing import Union, Optional

from utils.typechecker import optional_typecheck, beartype
from utils.logger import whi, yel, red
from utils.shared_module import shared
from utils.main import thread_whisp_then_llm, dirload_splitted, dirload_splitted_last, transcribe, alfred, to_anki
from utils.anki_utils import call_anki, get_decks, get_card_status

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
        if txt_whisp_prompt:
            txt_whisp_prompt = shared.pv["txt_whisp_prompt"]
        if txt_whisp_lang:
            txt_whisp_lang = shared.pv["txt_whisp_lang"]
        if stt_choice:
            stt_choice = shared.pv["stt_choice"]

        # force preprocessing of sound
        red("Forcing sound preprocessing")
        shared.preprocess_sox_effects = shared.force_preprocess_sox_effects

        shared.pv.__init__(profile=profile)

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

        audio_slots = dirload_splitted(True, [None] * nb_audio_slots)

        for audio in tqdm(total=audio_todo, unit="audio"):
            row = shared.dirload_queue.loc[audio.__str__(), :]
            assert row, f"Error with row: {row}"
            tmp_path = row["tmp_path"]
            text = transcribe(tmp_path)
            cloze = alfred(
                txt_audio=text,
                txt_chatgpt_context=shared.pv["txt_chatgpt_context"],
                profile=shared.pv.p.name,
                max_token=shared.pv["sld_max_tkn"],
                temperature=shared.pv["sld_temp"],
                sld_buffer=shared.pv["sld_buffer"],
                llm_choice=shared.pv["llm_choice"],
                txt_keywords=shared.pv["txt_keywords"],
                prompt_manag=shared.pv["prompt_manag"],
                cache_mode=False,
            )
            status = get_card_status(cloze)
            breakpoint()

        # transcribe -> alfred -> get_card_status
        # roll3:
            # transcribe -> alfred -> to_anki

        breakpoint()
        # for audio in tqdm(audio_todo, desc="Starting whisp then LLM on splits"):
        #     thread_whisp_then_llm(audio_mp3=audio)
        #     breakpoint()

        breakpoint()

