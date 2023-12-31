from tqdm import tqdm
import time
from pathlib import Path
import ankipandas as akp
import fire
from pydub import AudioSegment

from logger import red, whi, trace

try:
    db_path = akp.find_db(user="Main")
except Exception as err:
    red(f"Exception when trying to find anki collection: '{err}'")
    db_path = akp.Collection().path
red(f"WhisperToAnki will use anki collection found at {db_path}")

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

class DoneAudioChecker:
    def __init__(
            self,
            profile,
            ):
        profile = Path("./profiles/" + profile)
        self.unsp_dir = profile / "queues/audio_untouched"
        self.sp_dir = profile / "queues/audio_splits"
        self.done_dir = profile / "queues/audio_done"
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        done_list = sorted([p for p in self.done_dir.iterdir()])
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
                    ])

        # get the suffix list of each file in anki media
        suffix_list = set(p.suffix for p in anki_media.iterdir())

        # restrict found anki media to those that have the right suffix
        media_dict = {}
        for m in media_list:
            for suffix in suffix_list:
                if suffix == m.suffix:
                    name = "_".join(m.name.split("_")[2:])
                    name = name.replace("-0-100", "").replace(".mp3_", "_").replace("_processed", "")

                    if not name.strip():
                        red(f"Ignored media file as irrelevant: {m}")
                    else:
                        if suffix in media_dict:
                            media_dict[suffix].append(name)
                        else:
                            media_dict[suffix] = [name]
                        whi(f"Keeping {name}")
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
        red(f"Getting timestamps of media folder")
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
                    timestamps_m[stamp].append(p)
                else:
                    timestamps_m[stamp] = [p]
                m_timestamps[p] = stamp
            for ig in to_ignore:
                media_dict[suffix].remove(ig)

        for stamp, done_files in tqdm(timestamps_p.items(), desc="Checking timestamps"):
            if stamp in timestamps_m:
                for f in done_files:
                    name = f.name
                    if name.count("mp3") > 1:
                        name = name.replace(".mp3_", "_")
                    name = name.replace("_processed", "")
                    if name not in timestamps_m[stamp]:
                        red(f"Not found in media: {f.name} : {timestamps_m[stamp]}")
            else:
                red(f"No timestamp in media: {done_files}")

        breakpoint()

        # for each audio found in the done folder, check that it is found
        # in the media folder
        whi("\n\n")
        issues = {}
        for p in tqdm(done_list, desc="Checking"):
            suffix = p.suffix
            name = p.name.replace("-0-100", "")
            name = name.replace(" ", "_")
            # if name.count(".mp3") > 1:
            #     name = name.replace(".mp3", "", 1)
            if name not in media_dict[suffix]:
                stamp = p_timestamps[p.name]

                # todo: gather the timestamps of the files in media

                audio = AudioSegment.from_mp3(p)
                if len(audio) <= 1000:
                    whi(f"Ignored {p.name} (too short)")
                else:
                    red(f"ISSUE {p.name}")
                    issues[p.name] = audio
            else:
                whi(f"Ignored {p.name} (found)")
        breakpoint()


if __name__ == "__main__":
    fire.Fire(DoneAudioChecker)
