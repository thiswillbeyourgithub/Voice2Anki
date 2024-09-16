import time
import os
import json
import threading
import queue
import pickle
from pathlib import Path
import sys
import importlib.util
import numpy as np
import re
import zlib
from typing import Any, List, Optional, Union, Tuple, Callable, Iterator, Sequence, Any
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
    "llm_choice": {"default": [i for i in shared.llm_info.keys()][0]},
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
}
for i in range(1, shared.queued_gallery_slot_nb + 1):
    profile_keys[f"queued_gallery_{i:03d}"] = {}

for k in profile_keys:
    if "default" not in profile_keys[k]:
        profile_keys[k]["default"] = None

profile_path = Path("./profiles")

profile_path.mkdir(exist_ok=True)
(profile_path / "default").mkdir(exist_ok=True)

WORKER_TIMEOUT = 60 * 60  # 1 hour

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
        self.worker = threading.Thread(
                target=worker_setitem,
                kwargs={"in_queues": self.in_queues},
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

        self.__check_worker__()

        # create embeddings cache
        ec = self["choice_embed"]
        embed_cache = self.p / "embed_cache" / ec
        embed_cache.mkdir(exist_ok=True, parents=True)
        self.embed_cache = LocalFileStore(
            root_path=embed_cache,
            compress=False,
            update_atime=True,
        )

    def __check_worker__(self) -> None:
        if not self.worker.is_alive():
            red("Worker was not alive, restarting it")
            self.worker = threading.Thread(
                    target=worker_setitem,
                    kwargs={"in_queues": self.in_queues},
                    daemon=True,
                    )
            self.worker.start()
        assert self.worker.is_alive(), "Saver worker appears to be dead!"

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

    def __getitem__(self, key: str) -> Any:
        self.__check_worker__()
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

        self.__check_worker__()
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
def worker_setitem(in_queues: dict) -> None:
    """continuously running worker that is used to save components value to
    the appropriate profile"""
    # time since last
    t = None
    while True:
        time.sleep(0.5)
        if t and time.time() - t > WORKER_TIMEOUT:
            red("Worker: crashing because timeout")
            return

        for key, q in in_queues.items():
            if q.empty():
                continue
            size = q.qsize()
            if size > 100:
                red(f"Unexpected size of queue of {key}: {size}")
            if q.full():
                red(f"Queue of key {key} appears to be full. Size: {size}")
                raise Exception()

            profile, item = q.get_nowait()
            while not q.empty():  # always get the last value
                profile, item = q.get_nowait()
            # red(f"Saving {key}")  # for debugging
            t = time.time()

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
                    red(f"Error when saving message_buffer as json for key {key} in pickle: '{err}'")
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
                        red(f"Error when setting {kf}: '{err}'")


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
        shared.pv = None
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



class LocalFileStore:
    """
    source : https://api.python.langchain.com/en/latest/_modules/langchain/storage/file_system.html#LocalFileStore
    This is basically the exact same code but with added compression
    BaseStore interface that works on the local file system.


    Examples:
        Create a LocalFileStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import LocalFileStore

            # Instantiate the LocalFileStore with the root path
            file_store = LocalFileStore("/path/to/root")

            # Set values for keys
            file_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = file_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            file_store.mdelete(["key1"])

            # Iterate over keys
            for key in file_store.yield_keys():
                print(key)  # noqa: T201

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
        update_atime: bool = False,
        compress: Union[bool, int] = False,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
            update_atime: (optional, defaults to `False`) If `True`, updates the
                filesystem access time (but not the modified time) when a file is read.
                This allows MRU/LRU cache policies to be implemented for filesystems
                where access time updates are disabled.
            compress: (optional, defaults to `False`) If an int, compress
                stored data, reducing speed but lowering size. The int given
                is the level of compression of zlib, so between -1 and 9, both
                included. If `True`, defaults to -1, like in zlib.
        """
        self.root_path = Path(root_path).absolute()
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir
        self.update_atime = update_atime
        if compress is True:
            compress = -1
        if isinstance(compress, int):
            assert compress >= -1 and compress <= 9, (
                "compress arg as int must be between -1 and 9, both "
                f"included. Not {compress}"
            )
        self.compress = compress


    def _get_full_path(self, key: str) -> Path:
        """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
        if not re.match(r"^[a-zA-Z0-9_.\-/]+$", key):
            raise Exception(f"Invalid characters in key (the key should be a hash): {key}")
        full_path = os.path.abspath(self.root_path / key)
        common_path = os.path.commonpath([str(self.root_path), full_path])
        if common_path != str(self.root_path):
            raise Exception(
                f"Invalid key: {key}. Key should be relative to the full path."
                f"{self.root_path} vs. {common_path} and full path of {full_path}"
            )

        return Path(full_path)

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
            if self.chmod_dir is not None:
                os.chmod(dir, self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> List[Union[MISSING, Any]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be dataclasses.MISSING
        """
        values = []
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                with full_path.open("rb") as f:
                    value = pickle.load(f)
                assert value is not MISSING, f"Loaded a missing value for key '{key}'"
                if self.compress:
                    try:
                        value = zlib.decompress(value)
                    except zlib.error:
                        # trying to decompress data that wasn't compressed
                        # This happens if the user enables 'compress' without
                        # changing the root_dir, so overwriting the file to the
                        # compressed version
                        with full_path.open("wb") as f:
                            com_val = zlib.compress(value, level=self.compress)
                            pickle.dump(com_val, f)

                values.append(value)
                if self.update_atime:
                    # update access time only; preserve modified time
                    os.utime(full_path, (time.time(), os.stat(full_path).st_mtime))
            else:
                values.append(MISSING)
        return values


    def mset(self, key_value_pairs: Sequence[Tuple[str, Any]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            assert value is not MISSING, f"Trying to store a MISSING value for key '{key}'"
            full_path = self._get_full_path(key)
            self._mkdir_for_store(full_path.parent)
            if self.compress:
                com_val = zlib.compress(value, level=self.compress)
                with full_path.open("wb") as f:
                    pickle.dump(com_val, f)
            else:
                with full_path.open("wb") as f:
                    pickle.dump(value, f)
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)


    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for key in keys:
            full_path = self._get_full_path(key)
            if full_path.exists():
                full_path.unlink()


    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix_path = self._get_full_path(prefix) if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)

