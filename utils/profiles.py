import pickle
import hashlib
import json
from pathlib import Path

from .logger import red, whi


class previous_values:
    def __init__(self, profile="default"):
        self.p = Path(f"profiles/{profile}")
        if profile == "default":
            self.p.mkdir(exist_ok=True)
        assert self.p.exists(), "profile not found!"

    def __getitem__(self, key):
        if (self.p / key).exists():
            try:
                with open(str(self.p / key + ".pickle"), "r") as f:
                    return pickle.load(f)
            except Exception as err:
                try:
                    with open(str(self.p / key + ".pickle"), "rb") as f:
                        return pickle.load(f)
                except Exception as err:
                    raise Exception(f"Error when getting {key} from profile: '{err}'")
        else:
            whi(f"No {key} in store for profile")
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
                raise Exception(f"Error when setting {key} from profile: '{err}'")


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
            pv["audio_path"],
            pv["txt_audio"],
            pv["txt_chatgpt_cloz"],
            f"Switch profile to '{profile}'",
            ]
