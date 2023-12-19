import cv2
import threading
import pickle
from pathlib import Path
import numpy as np

from .logger import whi, red, trace
from .shared_module import shared
from .media import rgb_to_bgr

approved_keys_all = [
        "sld_max_tkn",
        "sld_buffer",
        "sld_temp",
        "txt_chatgpt_context",
        "txt_whisp_lang",
        "txt_whisp_prompt",
        "total_llm_cost",
        "dirload_check",
        "check_gpt4",
        "sld_whisp_temp",
        "message_buffer",
        "txt_keywords",
        "sld_prio_weight",
        "sld_keywords_weight",
        "sld_pick_weight",
        "txt_extra_source",
        "txt_openai_api_key",
        ]
approved_keys_all += [f"future_gallery_{i:03d}" for i in range(1, shared.future_gallery_slot_nb + 1)]
approved_keys_anki = approved_keys_all + ["gallery", "txt_deck", "txt_tags"]
approved_keys_md = approved_keys_all + ["txt_mdpath"]

profile_path = Path("./profiles")
anki_path = profile_path / "anki"
md_path = profile_path / "markdown"

profile_path.mkdir(exist_ok=True)
anki_path.mkdir(exist_ok=True)
md_path.mkdir(exist_ok=True)
(anki_path / "default").mkdir(exist_ok=True)
(md_path / "default").mkdir(exist_ok=True)

assert len([p for p in profile_path.iterdir() if str(p.name) not in ["anki", "markdown"]]) == 0, (
    "Directory profiles should only contains dir anki and markdown. Please move your profiles accordingly.")

class ValueStorage:
    @trace
    def __init__(self, profile="latest"):

        # determine the backend to know where to look for the profile
        if shared.backend == "anki":
            self.backend = "anki"
            self.approved_keys = approved_keys_anki
            self.backend_dir = anki_path
        elif shared.backend == "markdown":
            self.backend = "markdown"
            self.approved_keys = approved_keys_md
            self.backend_dir = md_path
        else:
            raise Exception(shared.backend)

        if profile == "latest":
            try:
                with open(str(self.backend_dir / "latest_profile.txt"), "r") as f:
                    profile = f.read()
            except Exception as err:
                red(f"Error when loading profile '{profile}': '{err}'")
                profile = "default"
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.replace("_", "").replace("-", "").isalpha(), f"profile is not alphanumeric: '{profile}'"
        self.p = self.backend_dir / profile

        # stored latest used profile
        with open(str(self.backend_dir / "latest_profile.txt"), "w") as f:
            f.write(profile)

        self.running_tasks = {k: None for k in self.approved_keys}
        self.cache_values = {k: None for k in self.approved_keys}
        self.profile_name = profile

        whi(f"Profile loaded: {self.p.name}")
        assert self.p.exists(), f"{self.p} not found!"

        # create methods like "save_gallery" to save the gallery to the profile
        for key in self.approved_keys:
            def create_save_method(key):
                @trace
                def save_method(value):
                    self.__setitem__(key, value)
                return save_method

            setattr(self, f"save_{key}", create_save_method(key))

    def __getitem__(self, key):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be reload from profiles: '{key}'")
        if self.running_tasks[key] is not None:
            whi(f"Waiting for task of {key} to finish.")
            self.running_tasks[key].join()
            whi(f"Done waiting for task {key}")

        if self.cache_values[key] is not None:
            return self.cache_values[key]

        kp = key + ".pickle"
        kf = self.p / kp

        if kf.exists():
            try:
                with open(str(kf), "r") as f:
                    new = pickle.load(f)
            except Exception:
                try:
                    with open(str(kf), "rb") as f:
                        new = pickle.load(f)
                except Exception as err:
                    raise Exception(f"Error when getting {kf}: '{err}'")
            if key.startswith("audio_mp3"):
                if not isinstance(new, (tuple, type(None))) and len(new) == 2 and isinstance(new[0], int) and isinstance(new[1], type(np.array(()))):
                    red(f"Error when loading {kf}: unexpected value for loaded value")
                    return None
            if (key == "gallery" or key.startswith("future_gallery_")) and new is not None:
                if not isinstance(new, list):
                    new = [im.image.path for im in new.root]
            self.cache_values[key] = new
            return new
        else:
            if not key.startswith("future_gallery_"):
                whi(f"No {kf} stored in profile dir, using appropriate default value for key {key}")
            if key == "sld_max_tkn":
                default = 3500
            elif key == "sld_buffer":
                default = 0
            elif key == "sld_temp":
                default = 0.2
            elif key == "txt_whisp_lang":
                default = "fr"
            elif key == "total_llm_cost":
                default = 0
            elif key == "dirload_check":
                default = False
            elif key == "check_gpt4":
                default = False
            elif key == "sld_whisp_temp":
                default = 0
            elif key == "message_buffer":
                default = []
            elif key == "sld_prio_weight":
                default = 1
            elif key == "sld_pick_weight":
                default = 1
            elif key == "sld_keywords_weight":
                default = 5
            else:
                default = None
            self.cache_values[key] = default
            return default

    def __check_equality(self, a, b):
        if not (isinstance(a, type(b)) and type(a) == type(b) and isinstance(b, type(a))):
            return False
        if isinstance(a, list):
            for i in range(len(a)):
                if not self.__check_equality(a[i], b[i]):
                    return False
            return True
        if isinstance(a, dict):
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

    def __setitem__(self, key, item):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be set from profiles: '{key}'")
        if not self.__check_equality(item, self.cache_values[key]):
            # make sure to wait for the previous setitem of the same key to finish
            if self.running_tasks[key] is not None:
                whi(f"Waiting for task of {key} to finish.")
                self.running_tasks[key].join()
                whi(f"Done waiting for task {key}")
                if item == self.cache_values[key]:  # value might
                    # have changed during the execution
                    return
            thread = threading.Thread(
                    target=self.__setitem__async,
                    kwargs={"key": key, "item": item, "lock": threading.Lock()})
            thread.start()
            self.running_tasks[key] = thread
        else:
            # item is the same as in the cache value
            # but if it's None etc, then the cache value must be destroyed
            try:
                if item is None or item is False or (bool(item) is False):
                    kp = key + ".pickle"
                    kf = self.p / kp
                    if kf.exists():
                        red(f"Deleting file {kf}")
                        kf.unlink()
            except Exception as err:
                red(f"Error when setting {key}=={item} being the same as in the cache dir: '{err}'")

    def __setitem__async(self, key, item, lock):
        kp = key + ".pickle"
        kf = self.p / kp

        kf.unlink(missing_ok=True)

        with lock:
            self.cache_values[key] = item

        try:
            with open(str(kf), "w") as f:
                return pickle.dump(item, f)
        except Exception:
            try:
                # try as binary
                with open(str(kf), "wb") as f:
                    return pickle.dump(item, f)
            except Exception as err:
                raise Exception(f"Error when setting {kf}: '{err}'")


@trace
def get_profiles():
    profiles = [str(p.name) for p in profile_path.iterdir()]
    if shared.backend == "anki":
        profiles = [str(p.name) for p in anki_path.iterdir()]
    elif shared.backend == "markdown":
        profiles = [str(p.name) for p in md_path.iterdir()]
    assert profiles, "Empty list of profiles"
    return profiles


@trace
def switch_profile(profile):
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
        if shared.backend == "anki":
            (anki_path / profile).mkdir(exist_ok=False)
        elif shared.backend == "markdown":
            (md_path / profile).mkdir(exist_ok=False)
        else:
            raise Exception(shared.backend)
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
    if shared.pv.backend == "anki":
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
    elif shared.pv.backend == "markdown":
        return [
                shared.pv["txt_mdpath"],
                shared.pv["txt_chatgpt_context"],
                shared.pv["txt_whisp_prompt"],
                shared.pv["txt_whisp_lang"],
                None,
                None,
                None,
                profile,
                ]


@trace
def save_path(txt_profile, txt_mdpath):
    if txt_mdpath:
        ValueStorage(txt_profile)["txt_mdpath"] = txt_mdpath
