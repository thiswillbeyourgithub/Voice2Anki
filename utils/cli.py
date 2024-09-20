import re
from pathlib import Path, PosixPath
from tqdm import tqdm
from dataclasses import MISSING
from typing import Union, Optional

from utils.typechecker import optional_typecheck, beartype
from utils.logger import whi, yel, red
from utils.shared_module import shared
from utils.main import to_anki, thread_whisp_then_llm
from utils.anki_utils import call_anki, get_decks

@optional_typecheck
class Cli:
    def __init__(
        self,
        audio_regex: str,
        txt_chatgpt_context: str,
        txt_deck: str,
        txt_tags: str,

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

        self.profile = profile

        untouched_dir = Path(".") / "profile" / profile / "queues" / "audio_untouched"
        splitted_dir = Path(".") / "profile" / profile / "queues" / "audio_splits"
        done_dir = Path(".") / "profile" / profile / "queues" / "audio_done"

        assert untouched_dir.exists()
        assert splitted_dir.exists()
        assert done_dir.exists()

        audio_todo = [f for f in splitted_dir.iterdir()]
        whi(f"Found {len(audio_todo)} audio splits before filtering")
        audio_todo = [f for f in audio_todo if audio_regex.match(f.name)]
        assert audio_todo, "No audio todo"
        whi(f"Found {len(audio_todo)} audio splits to do")

        for audio in tqdm(audio_todo, desc="Starting whisp then LLM on splits"):
            thread_whisp_then_llm(audio_mp3=audio)
            breakpoint()

        breakpoint()

