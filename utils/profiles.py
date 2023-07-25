import sys
import pickle
from pathlib import Path
import numpy as np
import cv2

from .logger import whi, red
from .misc import rgb_to_bgr

approved_keys = [
        "audio_numpy_1",
        "audio_numpy_2",
        "audio_numpy_3",
        "audio_numpy_4",
        "audio_numpy_5",
        "sld_max_tkn",
        "temperature",
        "txt_chatgpt_context",
        "txt_whisp_lang",
        "txt_whisp_prompt",
        "latest_profile",
        "total_llm_cost",
        ]
approved_keys_anki = ["gallery", "txt_deck", "txt_tags"]
approved_keys_md = ["txt_mdpath"]


class previous_values:
    def __init__(self, profile="default"):
        Path("./profiles").mkdir(exist_ok=True)
        Path("./profiles/anki").mkdir(exist_ok=True)
        Path("./profiles/markdown").mkdir(exist_ok=True)
        assert len([p for p in Path("./profiles").iterdir() if str(p.name) not in ["anki", "markdown"]]) == 0, (
            "Directory profiles should only contains dir anki and markdown. Please move your profiles accordingly.")
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.replace("_", "").replace("-", "").isalpha(), f"profile is not alphanumeric: '{profile}'"

        args = sys.argv[1:]
        if "--backend=anki" in args:
            self.backend = "anki"
            self.approved_keys = approved_keys + approved_keys_anki
            self.p = Path(f"./profiles/anki/{profile}")
        elif "--backend=markdown" in args:
            self.backend = "markdown"
            self.approved_keys = approved_keys + approved_keys_md
            self.p = Path(f"./profiles/markdown/{profile}")
        else:
            raise Exception

        if profile != "reload":
            self.p.mkdir(exist_ok=True)
        else:
            whi("Reloading latest profile")
            self.__init__(profile=self["latest_profile"])
        whi(f"Profile loaded: {self.p.name}")

        assert self.p.exists(), f"{self.p} not found!"

    def __getitem__(self, key):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be reload from profiles: '{key}'")
        kp = key + ".pickle"
        if key == "latest_profile":
            # latest_profile.pickle is stored in the root of the profile dir
            kf = self.p.parent / kp
        else:
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
            if key.startswith("audio_numpy"):
                if not isinstance(new, tuple) and len(new) == 2 and isinstance(new[0], int) and isinstance(new[1], type(np.array(()))):
                    red(f"Error when loading {kf}: unexpected value for loaded value")
                    return None
            if key == "gallery":
                # when reloading gallery, the image has to be inverted again
                for i, im in enumerate(new):
                    new[i] = rgb_to_bgr(im)
            return new
        else:
            whi(f"No {kf} stored in profile dir, using appropriate default value")
            if key == "sld_max_tkn":
                return 2000
            if key == "temperature":
                return 0.5
            if key == "txt_whisp_lang":
                return "fr"
            if key == "latest_profile":
                return "default"
            if key == "total_llm_cost":
                return 0
            return None
        return new

    def __setitem__(self, key, item):
        if key not in self.approved_keys:
            raise Exception(f"Unexpected key was trying to be set from profiles: '{key}'")
        kp = key + ".pickle"
        if key == "latest_profile":
            if item == "default":
                # don't store default as latest profile as it's already the default
                return None
            # the latest profile is stored in the root of the profile dir
            kf = self.p.parent / kp
        else:
            kf = self.p / kp

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


def get_profiles():
    profiles = [str(p.name) for p in Path("profiles").iterdir()]
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
        args = sys.argv[1:]
        if "--backend=anki" in args:
            Path(f"profiles/anki/{profile}").mkdir(exist_ok=False)
        elif "--backend=markdown" in args:
            Path(f"profiles/markdown/{profile}").mkdir(exist_ok=False)
        else:
            raise Exception
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
    pv["latest_profile"] = profile

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
                pv["audio_numpy_1"],
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
                pv["audio_numpy_1"],
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
