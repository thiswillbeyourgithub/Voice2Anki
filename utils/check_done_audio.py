import shutil
import hashlib
from tqdm import tqdm
import time
from pathlib import Path, PosixPath
import ankipandas as akp
import fire
from pydub import AudioSegment
from joblib import Memory
from typing import Optional, List

from logger import red, whi, cache_dir

from typechecker import optional_typecheck

@optional_typecheck
def hasher(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


audio_length_cache = Memory(cache_dir / "audio_length_checker", verbose=0)

@audio_length_cache.cache(ignore=["path"])
def get_audio_length(path, filehash):
    audio = AudioSegment.from_mp3(path)
    return len(audio)

@optional_typecheck
class DoneAudioChecker:
    def __init__(
        self,
        profile: PosixPath,
        anki_profile: str,
        exclude_list: Optional[List] = None,
        ):
        profile = Path("./profiles/" + profile)
        self.unsp_dir = profile / "queues/audio_untouched"
        self.sp_dir = profile / "queues/audio_splits"
        self.done_dir = profile / "queues/audio_done"
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        try:
            db_path = akp.find_db(user=anki_profile)
        except Exception as err:
            red(f"Exception when trying to find anki collection for profile {anki_profile}: '{err}'")
            db_path = akp.Collection(anki_profile).path
        red(f"Voice2Anki will use anki collection found at {db_path}")

        # check that akp will not go in trash
        if "trash" in str(db_path).lower():
            red("Ankipandas seems to have "
                "found a collection that might be located in "
                "the trash folder. If that is not your intention "
                "cancel now. Waiting 10s for you to see this "
                "message before proceeding.")
            time.sleep(1)
        anki_media = Path(db_path).parent / "collection.media"
        assert anki_media.exists(), "Media folder not found!"

        done_list = sorted([p for p in self.done_dir.iterdir()], key=lambda x: x.stat().st_ctime, reverse=True)
        whi(f"Found {len(done_list)} files in {self.done_dir}")

        # get the list of audio files in anki collection that were
        # created by Voice2Anki
        media_list = sorted(
                [
                    p
                    for p in anki_media.iterdir()
                    if (
                        # older names of Voice2Anki
                        p.name.startswith("Voice2FormattedText")
                        or p.name.startswith("WhisperToAnki")
                        or p.name.startswith("Voice2Anki")
                        )
                    ], key=lambda x: x.stat().st_ctime, reverse=True)

        # get the suffix list of each file in anki media
        suffix_list = set(p.suffix for p in anki_media.iterdir())

        # remove some old wrongly formatted files
        if exclude_list is not None:
            assert isinstance(exclude_list, list) and exclude_list, "exclude_list must be a list"

            to_remove = []
            for exc in exclude_list:
                for file in done_list:
                    if exc in file.name:
                        to_remove.append(file)
            done_list = [d for d in done_list if d not in to_remove]

        # restrict found anki media to those that have the right suffix
        media_dict = {}
        for m in media_list:
            for suffix in suffix_list:
                if suffix == m.suffix:
                    name = "_".join(m.name.split("_")[2:])
                    name = name.replace("-0-100", "").replace(".mp3_", "_").replace("_processed", "")

                    if not name.strip():
                        red(f"Ignored media file as irrelevant: {m.name}")
                    else:
                        if suffix in media_dict:
                            media_dict[suffix].append(name)
                        else:
                            media_dict[suffix] = [name]
                        # whi(f"Keeping {name}")
                    break

        # get the list of timestamp in done folder
        timestamps_p = {}
        p_timestamps = {}
        to_ignore = []
        for p in tqdm(done_list, desc="Getting timestamps"):
            sp = p.name.split("_")
            stamps = [t for t in sp if t.isdigit() and (t.startswith("16") or t.startswith("17")) and len(t) == 10]
            if not stamps:
                red(f"Ignored done file as no timestamp: {p}")
                to_ignore.append(p)
                continue
            assert len(stamps) == 1

            stamp = stamps[0]
            if stamp in timestamps_p:
                timestamps_p[stamp].append(p)
            else:
                timestamps_p[stamp] = [p]
            p_timestamps[p.name] = stamp

        done_list = [d for d in done_list if d not in to_ignore]

        # get the list of timestamp in media folder
        timestamps_m = {}
        m_timestamps = {}
        red("Getting timestamps of media folder")
        for suffix in media_dict:
            to_ignore = []
            for p in media_dict[suffix]:
                sp = p.split("_")
                stamps = [t for t in sp if t.isdigit() and (t.startswith("16") or t.startswith("17")) and len(t) == 10]
                if not stamps:
                    red(f"Ignored media file as no timestamp: {p}")
                    to_ignore.append(p)
                    continue
                assert len(stamps) == 1

                stamp = stamps[0]
                if stamp in timestamps_m:
                    timestamps_m[stamp].append(p.replace(stamp, ""))
                else:
                    timestamps_m[stamp] = [p.replace(stamp, "")]
                m_timestamps[p] = stamp
            for ig in to_ignore:
                media_dict[suffix].remove(ig)

        # check for each file of done if it's among the files with the
        # same timestamp in the media folder
        missing = []
        for stamp, done_files in tqdm(timestamps_p.items(), desc="Checking timestamps"):
            for f in done_files:
                name = f.name
                if name.count("mp3") > 1:
                    name = name.replace(".mp3_", "_")
                name = name.replace("_processed", "")
                name = name.replace(stamp, "")
                name = name.replace(" ", "_")
                really_missing = True
                for t in range(-10, 11, 1):
                    nt = str(int(stamp) + t)
                    if nt in timestamps_m and name in timestamps_m[nt]:
                        really_missing = False
                        break
                if really_missing:
                    red(f"Not found in media: {f.name}")
                    missing.append(f)

        # for each missing, check if it's very short
        missing_long = []
        for m in tqdm(missing, desc="Checking length"):
            with open(m, "rb") as audio_file:
                content = audio_file.read()
            audio_hash = hashlib.md5(content).hexdigest()
            le = get_audio_length(m.absolute(), audio_hash)
            if le <= 2500:
                whi(f"Ignored {m.name} (too short)")
            else:
                red(f"Long missing ({le//1000}s): {m.name}")
                missing_long.append(m)

        # copy the weird files for further manual inspection
        weirds = Path("weirds")
        weirds.mkdir(exist_ok=True)
        for m in missing_long:
            shutil.copy2(m.absolute(), weirds / m.name)
            red(f"Copied {m}")


if __name__ == "__main__":
    fire.Fire(DoneAudioChecker)
