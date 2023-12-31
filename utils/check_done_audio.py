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

        media_list = sorted([p for p in anki_media.iterdir() if p.name.startswith("Voice2FormattedText")])
        suffix_list = set(p.suffix for p in anki_media.iterdir())
        media_dict = {}
        for m in media_list:
            for suffix in suffix_list:
                if suffix == m.suffix:
                    name = "_".join(m.name.split("_")[2:])
                    name = name.replace("-0-100", "")
                    if suffix in media_dict:
                        media_dict[suffix].append(name)
                    else:
                        media_dict[suffix] = [name]
                    print(media_dict[suffix][-1])
                    break


        print("\n\n")
        issues = {}
        for p in tqdm(done_list, desc="Checking"):
            suffix = p.suffix
            name = p.name.replace("-0-100", "")
            name = name.replace(" ", "_")
            if name.count(".mp3") > 1:
                name = name.replace(".mp3", "", 1)
            if name not in media_dict[suffix]:
                audio = AudioSegment.from_mp3(p)
                if len(audio) <= 5000:
                    whi(f"Ignored {p.name} (too short)")
                else:
                    red(f"ISSUE {p.name}")
                    issues[p.name] = audio
            else:
                whi(f"Ignored {p.name} (found)")
        breakpoint()


if __name__ == "__main__":
    fire.Fire(DoneAudioChecker)
