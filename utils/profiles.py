import pickle
from pathlib import Path
import numpy as np

from .logger import whi, red


class previous_values:
    def __init__(self, profile="default"):
        assert Path("./profiles").exists(), "profile folder not found"
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.isalpha(), f"profile is not alphanumeric: '{profile}'"
        self.p = Path(f"./profiles/{profile}")
        if profile == "default":
            self.p.mkdir(exist_ok=True)
        assert self.p.exists(), f"{self.p} not found!"

    def __getitem__(self, key):
        kp = key + ".pickle"
        if (self.p / kp).exists():
            try:
                with open(str(self.p / kp), "r") as f:
                    new = pickle.load(f)
            except Exception:
                try:
                    with open(str(self.p / kp), "rb") as f:
                        new = pickle.load(f)
                except Exception as err:
                    raise Exception(f"Error when getting {kp} from {self.p}: '{err}'")
            if "key".startswith("audio_numpy_"):
                if not isinstance(new, tuple) and len(new) == 2 and isinstance(new[0], int) and isinstance(new[1], type(np.array(()))):
                    red(f"Error when loading {kp} from {self.p}: unexpected value for loaded value")
                    return None
            return new
        else:
            whi(f"No {kp} in store for {self.p}")
            if key == "max_tkn":
                return 3500
            if key == "temperature":
                return 0
            if key == "txt_whisp_lang":
                return "fr"
            return None
        return new

    def __setitem__(self, key, item):
        try:
            with open(str(self.p / (key + ".pickle")), "w") as f:
                return pickle.dump(item, f)
        except Exception:
            try:
                # try as binary
                with open(str(self.p / (key + ".pickle")), "wb") as f:
                    return pickle.dump(item, f)
            except Exception as err:
                raise Exception(f"Error when setting {key} from {self.p}: '{err}'")


def get_profiles():
    profiles = [str(p.name) for p in Path("profiles").iterdir()]
    assert profiles, "Empty list of profiles"
    return profiles


def switch_profile(profile):
    if profile is None or profile.strip() == "" or "/" in profile or not profile.isalpha():
        red("Invalid profile name, must be alphanumeric")
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
        Path(f"profiles/{profile}").mkdir(exist_ok=False)
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
    return [
            pv["txt_deck"],
            pv["txt_tags"],
            pv["txt_chatgpt_context"],
            pv["txt_whisp_prompt"],
            pv["txt_whisp_lang"],
            pv["gallery"],
            pv["audio_numpy_1"],
            pv["txt_audio"],
            pv["txt_chatgpt_cloz"],
            profile,
            ]
