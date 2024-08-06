import time
import os
import json
import multiprocessing
import queue
import pickle
from pathlib import Path
import sys
import importlib.util
import numpy as np
from typing import Any, List, Optional, Union, Tuple, Callable
from dataclasses import MISSING

import gradio as gr

try:
    from .logger import whi, yel, red, trace
    from .shared_module import shared
    from .typechecker import optional_typecheck
except Exception:
    from logger import whi, yel, red, trace
    from shared_module import shared
    from typechecker import optional_typecheck

profile_keys = {
    "enable_gallery": {"default": False, "type": bool},
    "enable_queued_gallery": {"default": False, "type": bool},
    "enable_flagging": {"default": False, "type": bool},
    "enable_dirload": {"default": False, "type": bool},
    "dirload_check": {"default": False},
    "sld_max_tkn": {"default": 1300},
    "sld_buffer": {"default": 0},
    "sld_temp": {"default": 0.0, "type": float},
    "txt_chatgpt_context": {},
    "txt_whisp_lang": {"default": "en"},
    "txt_whisp_prompt": {},
    "total_llm_cost": {"default": 0, "type": float},
    "prompt_management": {"default": "messages"},
    "llm_choice": {"default": [i for i in shared.llm_price.keys()][0]},
    "stt_choice": {"default": shared.stt_models[0], "type": str},
    "choice_embed": {"default": shared.embedding_models[0]},
    "sld_whisp_temp": {"default": 0, "type": float},
    "message_buffer": {"default": [], "type": list},
    "txt_keywords": {"default": "", "type": str},
    "sld_prio_weight": {"default": 1},
    "sld_keywords_weight": {"default": 1},
    "sld_timestamp_weight": {"default": 1},
    "sld_pick_weight": {"default": 1},
    "txt_extra_source": {},
    "txt_openai_api_key": {"default": ""},
    "txt_openrouter_api_key": {"default": ""},
    "txt_mistral_api_key": {"default": ""},
    "txt_deepgram_api_key": {"default": ""},
    "txt_deepgram_keyword_boosting": {"type": str},
    "gallery": {},
    "txt_deck": {},
    "txt_tags": {},
    "check_prompt_as_system": {"default": False, type: bool},
}
for i in range(1, shared.queued_gallery_slot_nb + 1):
    profile_keys[f"queued_gallery_{i:03d}"] = {}

for k in profile_keys:
    if "default" not in profile_keys[k]:
        profile_keys[k]["default"] = None

profile_path = Path("./profiles")

profile_path.mkdir(exist_ok=True)
(profile_path / "default").mkdir(exist_ok=True)

@optional_typecheck
class ValueStorage:
    _instance = None

    def __new__(cls):
        "make sure the instance will be a singleton"
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise Exception("Tried to create another instance of ValueStorage")

    @trace
    def __init__(self, profile: str = "latest") -> None:

        profile = profile.strip()

        if profile == "latest":
            try:
                with open(str(profile_path / "latest_profile.txt"), "r") as f:
                    profile = f.read().strip()
            except Exception as err:
                red(f"Error when loading profile '{profile}': '{err}'")
                profile = "default"
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.replace("_", "").replace("-", "").isalpha(), f"profile is not alphanumeric: '{profile}'"
        self.p = profile_path / profile

        # stored latest used profile
        with open(str(profile_path / "latest_profile.txt"), "w") as f:
            f.write(profile)

        self.in_queues = {k: queue.Queue() for k in profile_keys}
        self.out_queue = queue.Queue()
        self.worker = multiprocessing.Process(
                target=worker_setitem,
                kwargs={"in_queues": self.in_queues, "out_queue": self.out_queue},
                daemon=True,
                )
        self.worker.start()

        self.cache_values = {k: MISSING for k in profile_keys}

        self.profile_name = profile
        whi(f"Profile loaded: {self.p.name}")
        assert self.p.exists(), f"{self.p} not found!"

        # create methods like "save_gallery" to save the gallery to the profile
        for key in profile_keys:
            @optional_typecheck
            def create_save_method(key: str) -> Callable:
                #@trace
                @optional_typecheck
                def save_method(value) -> None:
                    self.__setitem__(key, value)
                return save_method

            setattr(self, f"save_{key}", create_save_method(key))

    def __check_equality(self, a: Any, b: Any) -> bool:
        assert a is not MISSING
        if b is MISSING:
            return False
        if not (isinstance(a, type(b)) and type(a) == type(b) and isinstance(b, type(a))):
            return False
        if isinstance(a, list):
            if not isinstance(b, list):
                return False
            if len(a) != len(b):
                return False
            for i in range(len(a)):
                if not self.__check_equality(a[i], b[i]):
                    return False
            return True
        if isinstance(a, dict):
            if not isinstance(b, dict):
                return False
            for k in b:
                if k not in a:
                    return False
            for k, v in a.items():
                if k not in b or self.__check_equality(b[k], v):
                    return False
        try:
            return (a == b).all()
        except Exception:
            return a == b

    def __check_message__(self) -> None:
        "read the message left in the out_queue by the worker"
        try:
            mess = self.out_queue.get_nowait()
        except queue.Empty:
            return
        red(mess)

    def __getitem__(self, key: str) -> Any:
        self.__check_message__()
        assert self.worker.is_alive(), "Saving worker appears to be dead!"
        if key not in profile_keys:
            raise Exception(f"Unexpected key was trying to be reload from profiles: '{key}'")

        if self.cache_values[key] is not MISSING:
            if "type" in profile_keys[key]:
                # cast as specific type
                return profile_keys[key]["type"](self.cache_values[key])
            else:
                return self.cache_values[key]

        kp = key + ".pickle"
        if key.startswith("queued_gallery_"):
            kf = self.p / "queues" / "galleries" / kp
        else:
            kf = self.p / kp

        if kf.exists():
            if key == "message_buffer":
                try:
                    with open(str(kf), "r") as f:
                        new = json.load(f)
                except Exception as err:
                    red(f"Error when loading message_buffer as json in pickle: '{err}'")

            else:
                try:
                    with open(str(kf), "r") as f:
                        new = pickle.load(f)
                except Exception:
                    try:
                        with open(str(kf), "rb") as f:
                            new = pickle.load(f)
                    except Exception as err:
                        raise Exception(f"Error when getting {kf}: '{err}'")
                if (key == "gallery" or key.startswith("queued_gallery_")) and new is not None:
                    if not isinstance(new, list):
                        new = [im.image.path for im in new.root]
            if "type" in profile_keys[key]:
                new = profile_keys[key]["type"](new)
            self.cache_values[key] = new
            return new
        else:
            self.cache_values[key] = profile_keys[key]["default"]
            return self.cache_values[key]

    def __setitem__(self, key: str, item: Any) -> None:
        # update the api key right away
        if "api_key" in key:
            if key == "txt_openai_api_key":
                os.environ["OPENAI_API_KEY"] = item
            elif key == "txt_deepgram_api_key":
                os.environ["DEEPGRAM_API_KEY"] = item
            elif key == "txt_mistral_api_key":
                os.environ["MISTRAL_API_KEY"] = item
            elif key == "txt_openrouter_api_key":
                os.environ["OPENROUTER_API_KEY"] = item

        self.__check_message__()
        assert self.worker.is_alive(), "Saver worker appears to be dead!"
        if key not in profile_keys:
            raise Exception(f"Unexpected key was trying to be set from profiles: '{key}'")

        if not self.__check_equality(item, self.cache_values[key]):

            if item != self.cache_values[key]:
                # cast as required type
                if "type" in profile_keys[key]:
                    item = profile_keys[key]["type"](item)

                # save to cache
                self.cache_values[key] = item

            # save to file
            self.in_queues[key].put((self.p, item))

        else:
            # item is the same as in the cache value
            # but if it's None etc, then the cache value must be destroyed
            try:
                if item is profile_keys[key]["default"]:
                    kp = key + ".pickle"
                    if key.startswith("queued_gallery_"):
                        kf = self.p / "queues" / "galleries" / kp
                    else:
                        kf = self.p / kp
                    if kf.exists():
                        red(f"Deleting file {kf}")
                        kf.unlink()
                    else:
                        whi(f"Couldn't delete file as it did not exist: {kf}")
            except Exception as err:
                red(f"Error when setting {key}=={item} being the same as in the cache dir: '{err}'")

@optional_typecheck
def worker_setitem(in_queues: dict, out_queue: queue.Queue) -> None:
    """continuously running worker that is used to save components value to
    the appropriate profile"""
    while True:
        profile = None
        while profile is None:
            for key, q in in_queues.items():
                try:
                    profile, item = q.get_nowait()
                    break
                except queue.Empty:
                    time.sleep(0.01)

        kp = key + ".pickle"
        if key.startswith("queued_gallery_"):
            kf = profile / "queues" / "galleries" / kp
        else:
            kf = profile / kp

        kf.unlink(missing_ok=True)

        if item is profile_keys[key]["default"]:
            # if it's the default value, simply delete it and don't create
            # the new file
            continue

        if key == "message_buffer":
            try:
                with open(str(kf), "w") as f:
                    json.dump(item, f, indent=4, ensure_ascii=False)
            except Exception as err:
                out_queue.put(f"Error when saving message_buffer as json for key {key} in pickle: '{err}'")
        else:
            try:
                with open(str(kf), "w") as f:
                    pickle.dump(item, f)
            except Exception:
                try:
                    # try as binary
                    with open(str(kf), "wb") as f:
                        pickle.dump(item, f)
                except Exception as err:
                    out_queue.put(f"Error when setting {kf}: '{err}'")


# @trace
@optional_typecheck
def get_profiles() -> List[str]:
    profiles = [str(p.name) for p in profile_path.iterdir()]
    if "latest_profile.txt" in profiles:
        profiles.remove("latest_profile.txt")
    assert profiles, "Empty list of profiles"
    return profiles


@optional_typecheck
@trace
def switch_profile(profile: str) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[Union[gr.Gallery, List[dict]]],
    None,
    None,
    None,
    str,
]:
    # output is [
    #         txt_deck,
    #         txt_tags,
    #         txt_chatgpt_context,
    #         txt_whisp_prompt,
    #         txt_whisp_lang,
    #         gallery,
    #         audio_slots[0],
    #         txt_audio,
    #         txt_chatgpt_cloz,
    #         txt_profile]

    if profile is None or profile.strip() == "" or "/" in profile or not profile.replace("_", "").replace("-", "").isalpha():
        red("Invalid profile name, must be alphanumeric (although it can include _ and -)")
        return [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                ]

    profile = profile.lower()

    if profile not in get_profiles():
        (profile_path / profile).mkdir(exist_ok=False)
        red(f"created {profile}.")
        return [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                profile,
                ]

    shared.pv = ValueStorage(profile)

    # reset the fields to the previous values of profile
    whi(f"Switch profile to '{profile}'")
    return [
            shared.pv["txt_deck"],
            shared.pv["txt_tags"],
            shared.pv["txt_chatgpt_context"],
            shared.pv["txt_whisp_prompt"],
            shared.pv["txt_whisp_lang"],
            shared.pv["gallery"],
            None,
            None,
            None,
            profile,
            ]


@optional_typecheck
def load_user_functions() -> Tuple[str, str]:
    """ If the user profile directory contains a directory called "functions"
    then functions can be loaded.
    Currently supported use case are:

    * If contains "flashcard_editor.py" then it will be loaded
      to edit on the fly flashcards when they are sent to anki.
      The idea is that it allows systematic change like removing accents or using
      acronyms for example as this is sometimes hard to follow by LLMs. The
      name of the function must be 'cloze_editor' take str as input and output
      a string.
    * If contains "chains.py" then must contain a dict 'chains' with as keys
      'name' and 'func'. The name is used to create buttons for the chain.
      The func must take as input a string and output a string.
      The goal is to enable complex formating like turning quickly a table into
      multiple flashcards.
      When called, the input will be the transcripted text and the output
      will be sent again to the transcript, so as to be an input to the
      "transcript to cloze" workflow.
    """
    flashcard_editor = """
# This contains the code in profile/USER/functions/flashcard_editor.py if
# found. It must contain a function called cloze_editor that takes as input
# a string and returns another string.
# The idea is that it allows systematic change like removing accents or using
# acronyms for example as this is sometimes hard to follow by LLMs
"""
    if [f
        for f in shared.func_dir.iterdir()
        if f.name.endswith("flashcard_editor.py")]:
        red("Found flashcard_editor.py")
        with open((shared.func_dir / "flashcard_editor.py").absolute(), "r") as f:
            flashcard_editor += f.read()
    chains = """
# This contains the code in profile/USER/functions/chains.py if
# found. It must contain a dict called chains with as keys 'name' and
# 'chain'. The name is used to create buttons for the chain.
# The chain must take as input a string and output a string.
# The goal is to enable complex formating like turning quickly a table into
# multiple flashcards.
# When called, the input will be the transcripted text and the output
# will be sent again to the transcript, so as to be an input to the
# "transcript to cloze" workflow.
"""
    if [f
        for f in shared.func_dir.iterdir()
        if f.name.endswith("chains.py")]:
        red("Found chains.py")
        with open((shared.func_dir / "chains.py").absolute(), "r") as f:
            chains += f.read()
    return flashcard_editor, chains


@optional_typecheck
def load_user_chain(*buttons) -> List:
    if shared.user_chains is not None:
        return buttons
    if not [f
            for f in shared.func_dir.iterdir()
            if f.name.endswith("chains.py")]:
        red("No chains.py found")
        return buttons
    # load chains if not already done
    buttons = list(buttons)
    yel("Loading chains.py")
    spec = importlib.util.spec_from_file_location(
            "chains.chains",
            (shared.func_dir / "chains.py").absolute()
            )
    chains = importlib.util.module_from_spec(spec)
    sys.modules["chains"] = chains
    spec.loader.exec_module(chains)

    shared.user_chains = []
    assert len(chains.chains) <= len(buttons)
    for i, chain in enumerate(chains.chains):
        upd = gr.update(
                visible=True,
                value=chain["name"],
                )
        shared.user_chains.append(chain)
        buttons[i] = upd

    return buttons

@optional_typecheck
def call_user_chain(txt_audio: str, evt: gr.EventData) -> str:
    i_ch = int(evt.target.elem_id.split("#")[1])
    chain = shared.user_chains[i_ch]
    assert chain is not None
    func = trace(chain["func"])
    txt_audio = func(txt_audio)
    assert isinstance(txt_audio, str), f"Output of user chain must be a string, not {type(txt_audio)}. Value: {txt_audio}"
    return txt_audio
