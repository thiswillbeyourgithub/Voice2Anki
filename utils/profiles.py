import pickle
from pathlib import Path

from .logger import whi


class previous_values:
    def __init__(self, profile="default"):
        assert Path("./profiles").exists(), "profile folder not found"
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.isalpha(), f"profile is not alphanumeric: '{profile}'"
        self.p = Path(f"./profiles/{profile}")
        if profile == "default":
            whi("Assuming profile 'default'")
            self.p.mkdir(exist_ok=True)
        assert self.p.exists(), f"{self.p} not found!"

    def __getitem__(self, key):
        kp = key + ".pickle"
        if (self.p / kp).exists():
            try:
                with open(str(self.p / kp), "r") as f:
                    return pickle.load(f)
            except Exception:
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


def switch_profile(profile, output):
    if profile is None or profile.strip() == "" or "/" in profile or not profile.isalpha():
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
                f"Invalid profile name, must be alphanumeric" + output
                ]

    profile = profile.lower()

    # checks that there is no , in profile, which would be the default value
    if "," in profile:
        spl = [s for s in profile.strip().split(",") if s.strip()]
        if not len(spl) == 1:
            raise Exception(f"profile with invalid value: '{profile}'")
        else:
            profile = spl[0]

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
                None,
                profile,
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
            profile,
            f"Switch profile to '{profile}'",
            ]
