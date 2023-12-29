import json
import cv2
import threading
import queue
import pickle
from pathlib import Path
import numpy as np

from .logger import whi, red, trace
from .shared_module import shared
from .media import rgb_to_bgr

profile_keys = {
        "sld_max_tkn": {"default": 3500},
        "sld_buffer": {"default": 0},
        "sld_temp": {"default": 0.0, "type": float},
        "txt_chatgpt_context": {},
        "txt_whisp_lang": {"default": "en"},
        "txt_whisp_prompt": {},
        "total_llm_cost": {"default": 0, "type": float},
        "dirload_check": {"default": False},
        "llm_choice": {"default": [i for i in shared.llm_price.keys()][0]},
        "sld_whisp_temp": {"default": 0, "type": float},
        "message_buffer": {"default": [], "type": list},
        "txt_keywords": {"default": "", "type": str},
        "sld_prio_weight": {"default": 1},
        "sld_keywords_weight": {"default": 1},
        "sld_timestamp_weight": {"default": 1},
        "sld_pick_weight": {"default": 1},
        "txt_extra_source": {},
        "txt_openai_api_key": {},
        "txt_replicate_api_key": {},
        "gallery": {},
        "txt_deck": {},
        "txt_tags": {},
        }
for i in range(1, shared.future_gallery_slot_nb + 1):
    profile_keys[f"future_gallery_{i:03d}"] = {}

for k in profile_keys:
    if "default" not in profile_keys[k]:
        profile_keys[k]["default"] = None

profile_path = Path("./profiles")

profile_path.mkdir(exist_ok=True)
(profile_path / "default").mkdir(exist_ok=True)

class ValueStorage:
    @trace
    def __init__(self, profile="latest"):

        profile = profile.strip()
        self.profile_keys = profile_keys

        if profile == "latest":
            try:
                with open(str(profile_path / "latest_profile.txt"), "r") as f:
                    profile = f.read()
            except Exception as err:
                red(f"Error when loading profile '{profile}': '{err}'")
                profile = "default"
        assert isinstance(profile, str), f"profile is not a string: '{profile}'"
        assert profile.replace("_", "").replace("-", "").isalpha(), f"profile is not alphanumeric: '{profile}'"
        self.p = profile_path / profile

        # stored latest used profile
        with open(str(profile_path / "latest_profile.txt"), "w") as f:
            f.write(profile)

        self.in_queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread = threading.Thread(
                target=worker_setitem,
                args=[self.in_queue],
                daemon=False,
                )
        self.thread.start()

        self.running_tasks = {k: None for k in self.profile_keys}
        self.cache_values = {k: None for k in self.profile_keys}

        self.profile_name = profile
        whi(f"Profile loaded: {self.p.name}")
        assert self.p.exists(), f"{self.p} not found!"

        # create methods like "save_gallery" to save the gallery to the profile
        for key in self.profile_keys:
            def create_save_method(key):
                #@trace
                def save_method(value):
                    self.__setitem__(key, value)
                return save_method

            setattr(self, f"save_{key}", create_save_method(key))

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

    def __getitem__(self, key):
        assert self.thread.is_alive(), "Saving thread appears to be dead!"
        if key not in self.profile_keys:
            raise Exception(f"Unexpected key was trying to be reload from profiles: '{key}'")

        # make sure to wait for the previous setitem of the same key to finish
        prev_q = self.running_tasks[key]
        if prev_q is None:
            prev_q_val = True
        while prev_q is not None:
            prev_q = self.running_tasks[key]
            try:
                # Waits for X seconds, otherwise throws `Queue.Empty`
                prev_q_val = prev_q.get(True, 1)
                with self.lock:
                    self.running_tasks[key] = None
                whi(f"Done waiting for task {key}")
                break
            except queue.Empty:
                red(f"Waiting for {key} queue to output in getitem")
        if prev_q_val is not True:
            assert isinstance(prev_q_val, str), f"Unexpected prev_q_val: '{prev_q_val}'"
            raise Exception(f"Didn't save key {key} because previous saving went wrong: '{prev_q_val}'")

        if self.cache_values[key] is not None:
            if "type" in self.profile_keys[key]:
                # cast as specific type
                return self.profile_keys[key]["type"](self.cache_values[key])
            else:
                return self.cache_values[key]

        kp = key + ".pickle"
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
                if key.startswith("audio_mp3"):
                    if not isinstance(new, (tuple, type(None))) and len(new) == 2 and isinstance(new[0], int) and isinstance(new[1], type(np.array(()))):
                        red(f"Error when loading {kf}: unexpected value for loaded value")
                        return None
                if (key == "gallery" or key.startswith("future_gallery_")) and new is not None:
                    if not isinstance(new, list):
                        new = [im.image.path for im in new.root]
            if "type" in self.profile_keys[key]:
                new = self.profile_keys[key]["type"](new)
            self.cache_values[key] = new
            return new
        else:
            self.cache_values[key] = self.profile_keys[key]["default"]
            return self.cache_values[key]

    def __setitem__(self, key, item):
        assert self.thread.is_alive(), "Saver worker appears to be dead!"

        if key not in self.profile_keys:
            raise Exception(f"Unexpected key was trying to be set from profiles: '{key}'")

        if not self.__check_equality(item, self.cache_values[key]):
            # make sure to wait for the previous setitem of the same key to finish
            prev_q = self.running_tasks[key]
            if prev_q is None:
                prev_q_val = True
            while prev_q is not None:
                prev_q = self.running_tasks[key]
                try:
                    # Waits for X seconds, otherwise throws `Queue.Empty`
                    prev_q_val = prev_q.get(True, 1)
                    with self.lock:
                        self.running_tasks[key] = None
                    whi(f"Done waiting for task {key}")
                    break
                except queue.Empty:
                    red(f"Waiting for {key} queue to output in setitem")
            if prev_q_val is not True:
                assert isinstance(prev_q_val, str), f"Unexpected prev_q_val: '{prev_q_val}'"
                raise Exception(f"Didn't save key {key} because previous saving went wrong: '{prev_q_val}'")
            if item == self.cache_values[key]:
                # value might have changed during the execution
                return
            else:
                # cast as required type
                if "type" in self.profile_keys[key]:
                    item = self.profile_keys[key]["type"](item)

                # save to cache
                self.cache_values[key] = item

                # save to file
                out_q = queue.Queue()
                self.in_queue.put([self.p, key, item, out_q])
                self.running_tasks[key] = out_q
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

def worker_setitem(in_queue):
    """continuously running worker that is used to save components value to
    the appropriate profile"""
    while True:
        profile, key, item, out_queue = in_queue.get()
        kp = key + ".pickle"
        kf = profile / kp

        kf.unlink(missing_ok=True)

        if key == "message_buffer":
            try:
                with open(str(kf), "w") as f:
                    json.dump(item, f, indent=4, ensure_ascii=False)
                out_queue.put(True)
            except Exception as err:
                out_queue.put(red(f"Error when saving message_buffer as json in pickle: '{err}'"))
        else:
            try:
                with open(str(kf), "w") as f:
                    pickle.dump(item, f)
                out_queue.put(True)
            except Exception:
                try:
                    # try as binary
                    with open(str(kf), "wb") as f:
                        pickle.dump(item, f)
                    out_queue.put(True)
                except Exception as err:
                    out_queue.put(f"Error when setting {kf}: '{err}'")
                    raise Exception(f"Error when setting {kf}: '{err}'")


@trace
def get_profiles():
    profiles = [str(p.name) for p in profile_path.iterdir()]
    if "latest_profile.txt" in profiles:
        profiles.remove("latest_profile.txt")
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
