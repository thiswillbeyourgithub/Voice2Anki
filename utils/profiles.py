import threading
import time
import pickle
from pathlib import Path
import numpy as np
import cv2

from .logger import whi, red
from .misc import rgb_to_bgr, backend_config

approved_keys_all = [
        "audio_mp3_1",
        "audio_mp3_2",
        "audio_mp3_3",
        "audio_mp3_4",
        "audio_mp3_5",
        "sld_max_tkn",
        "temperature",
        "txt_chatgpt_context",
        "txt_whisp_lang",
        "txt_whisp_prompt",
        "total_llm_cost",
        ]
approved_keys_anki = approved_keys_all + ["gallery", "txt_deck", "txt_tags"]
approved_keys_md = approved_keys_all + ["txt_mdpath"]

profile_path = Path("./profiles")
anki_path = profile_path / "anki"
md_path = profile_path / "markdown"

profile_path.mkdir(exist_ok=True)
anki_path.mkdir(exist_ok=True)
md_path.mkdir(exist_ok=True)

class previous_values:
    def __init__(self, profile="default"):
        assert len([p for p in profile_path.iterdir() if str(p.name) not in ["anki", "markdown"]]) == 0, (
            "Directory profiles should only contains dir anki and markdown. Please move your profiles accordingly.")
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.replace("_", "").replace("-", "").isalpha(), f"profile is not alphanumeric: '{profile}'"

        if backend_config.backend == "anki":
            self.backend = "anki"
            self.approved_keys = approved_keys_anki
            self.p = anki_path / profile
        elif backend_config.backend == "markdown":
            self.backend = "markdown"
            self.approved_keys = approved_keys_md
            self.p = md_path / profile
        else:
            raise Exception(backend_config.backend)
        with open(str(md_path / "latest_profile.pickle"), "wb") as f:
            pickle.dump(profile, f)

        self.running_tasks = {k: None for k in self.approved_keys}
        self.cache_values = {k: None for k in self.approved_keys}
        self.profile_name = profile

        whi(f"Profile loaded: {self.p.name}")
        assert self.p.exists(), f"{self.p} not found!"

    def __getitem__(self, key):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be reload from profiles: '{key}'")
        if self.running_tasks[key] is not None:
            red(f"Waiting for task of {key} to finish.")
            self.running_tasks[key].join()

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
            if key == "gallery":
                # when reloading gallery, the image has to be inverted again
                for i, im in enumerate(new):
                    new[i] = rgb_to_bgr(im)
            self.cache_values[key] = new
            return new
        else:
            whi(f"No {kf} stored in profile dir, using appropriate default value")
            if key == "sld_max_tkn":
                default = 2000
            if key == "temperature":
                default = 0.5
            if key == "txt_whisp_lang":
                default = "fr"
            if key == "total_llm_cost":
                default = 0
            else:
                default = None
            self.cache_values[key] = default
            return default

    def __setitem__(self, key, item):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be set from profiles: '{key}'")
        if item != self.cache_values[key]:
            # make sure to wait for the previous setitem of the same key to finish
            if self.running_tasks[key] is not None:
                red(f"Waiting for task of {key} to finish.")
                self.running_tasks[key].join()
                if item == self.cache_values[key]:  # value might
                    # have changed during the await
                    return
            thread = threading.Thread(
                    target=self.__setitem__async,
                    kwargs={"key": key, "item": item})
            thread.start()
            self.running_tasks[key] = thread


    def __setitem__async(self, key, item):
        kp = key + ".pickle"
        kf = self.p / kp

        try:
            self.cache_values[key] = item
            with open(str(kf), "w") as f:
                return pickle.dump(item, f)
        except Exception:
            try:
                # try as binary
                self.cache_values[key] = item
                with open(str(kf), "wb") as f:
                    return pickle.dump(item, f)
            except Exception as err:
                raise Exception(f"Error when setting {kf}: '{err}'")


def get_profiles():
    profiles = [str(p.name) for p in profile_path.iterdir()]
    if backend_config.backend == "anki":
        profiles = [str(p.name) for p in anki_path.iterdir()]
    elif backend_config.backend == "markdown":
        profiles = [str(p.name) for p in md_path.iterdir()]
    assert profiles, "Empty list of profiles"
    return profiles


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
        if backend_config.backend == "anki":
            (anki_path / profile).mkdir(exist_ok=False)
        elif backend_config.backend == "markdown":
            (md_path / profile).mkdir(exist_ok=False)
        else:
            raise Exception(backend_config.backend)
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

    pv = previous_values(profile)

    # reset the fields to the previous values of profile
    whi(f"Switch profile to '{profile}'")
    if pv.backend == "anki":
        return [
                pv["txt_deck"],
                pv["txt_tags"],
                pv["txt_chatgpt_context"],
                pv["txt_whisp_prompt"],
                pv["txt_whisp_lang"],
                pv["gallery"],
                pv["audio_mp3_1"],
                None,
                None,
                profile,
                ]
    elif pv.backend == "markdown":
        return [
                pv["txt_mdpath"],
                pv["txt_chatgpt_context"],
                pv["txt_whisp_prompt"],
                pv["txt_whisp_lang"],
                pv["audio_mp3_1"],
                None,
                None,
                profile,
                ]

def save_tags(txt_profile, txt_tags):
    if txt_tags:
        previous_values(txt_profile)["txt_tags"] = txt_tags

def save_deck(txt_profile, txt_deck):
    if txt_deck:
        previous_values(txt_profile)["txt_deck"] = txt_deck

def save_path(txt_profile, txt_mdpath):
    if txt_mdpath:
        previous_values(txt_profile)["txt_mdpath"] = txt_mdpath
