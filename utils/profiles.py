import pickle
import hashlib
import json
from pathlib import Path

from .logger import red, whi


class previous_values:
    def __init__(self, profile="default"):
        assert Path("./profiles").exists(), "profile folder not found"
        self.p = Path(f"./profiles/{profile}")
        if profile == "default":
            whi(f"Assuming profile 'default'")
            self.p.mkdir(exist_ok=True)
        assert self.p.exists(), f"{self.p} not found!"

    def __getitem__(self, key):
        kp = key + ".pickle"
        if (self.p / kp).exists():
            try:
                with open(str(self.p / kp), "r") as f:
                    return pickle.load(f)
            except Exception as err:
                try:
                    with open(str(self.p / kp), "rb") as f:
                        return pickle.load(f)
                except Exception as err:
                    raise Exception(f"Error when getting {kp} from {self.p}: '{err}'")
        else:
            whi(f"No {kp} in store for {self.p}")
            if key == "max_tkn":
                return 3500
            return None

    def __setitem__(self, key, item):
        try:
            with open(str(self.p / (key + ".pickle")), "w") as f:
                return pickle.dump(item, f)
        except Exception as err:
            try:
                # try as binary
                with open(str(self.p / (key + ".pickle")), "wb") as f:
                    return pickle.dump(item, f)
            except Exception as err:
                raise Exception(f"Error when setting {key} from {self.p}: '{err}'")


def get_profiles():
    profiles = [str(p.name) for p in Path(f"profiles").iterdir()]
    assert profiles, f"Empty list of profiles"
    return profiles

def switch_profile(profile, output):
    if profile not in get_profiles():
        Path(f"profiles/{profile}").mkdir(exist_ok=False)
        return [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                f"created {profile}.\n\n" + output
                ]

    pv = previous_values(profile)

    # reset the fields to the previous values of profile
    return [
            pv["txt_deck"],
            pv["txt_tags"],
            pv["txt_chatgpt_context"],
            pv["txt_whisp_prompt"],
            pv["gallery"],
            pv["audio_numpy"],
            pv["txt_audio"],
            pv["txt_chatgpt_cloz"],
            f"Switch profile to '{profile}'",
            ]
