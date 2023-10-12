from datetime import datetime
import replicate
import shutil
import hashlib
import re
import joblib
import time
import exiftool
from tqdm import tqdm
import fire
from pathlib import Path
import os
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from logger import whi, yel, red

# TODO fix that
assert Path("REPLICATE_API_KEY.txt").exists(), "No api key found. Create a file REPLICATE8API_KEY.txt and paste your openai API key inside"
os.environ["REPLICATE_API_TOKEN"] = str(Path("REPLICATE_API_KEY.txt").read_text()).strip()

stt_cache = joblib.Memory("transcript_cache", verbose=0)

d = datetime.today()
today = f"{d.day:02d}_{d.month:02d}"

class AudioSplitter:
    def __init__(
            self,
            prompt=None,
            stop_list=["stop", "nouvelles? cartes?", "nouvelles? questions?"],
            language="fr",
            n_todo=1,
            unsplitted_dir="./user_directory/unsplitted",
            splitted_dir="./user_directory/splitted",
            done_dir = "./user_directory/done",
            ):
        self.unsp_dir = Path(unsplitted_dir)
        self.sp_dir = Path(splitted_dir)
        self.done_dir = Path(done_dir)
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        assert isinstance(prompt, str), "prompt argument should be string"
        assert isinstance(n_todo, (float, int)) and n_todo > 0, "n_todo should be a number greater than 0"

        self.prompt = prompt
        self.n_todo = n_todo
        self.language = language
        self.stop_list = [
                re.compile(s, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
                for s in stop_list]

        self.to_split = self.gather_todos()

        for file in tqdm(self.to_split, unit="file"):
            self.split_one_file(file)

    def gather_todos(self):
        to_split = [p for p in self.unsp_dir.rglob("*.mp3")]
        assert to_split, f"no mp3 found in {self.unsp_dir}"
        to_split = sorted(to_split, key=lambda x: x.stat().st_ctime)
        whi(f"Total number of files to split: {len(to_split)}")

        to_split = to_split[:self.n_todo]
        return to_split

    def split_one_file(self, file_path):
        transcript = self.run_whisperx(file_path)

        # find where stop is pronounced
        duration = transcript["segments"][-1]["end"]
        whi(f"Duration of {file_path}: {duration}")
        # note: duration is not the total recording duration but rather the
        # time of the end of the last pronounced word

        full_text = transcript["transcription"]
        whi(f"Full text of {file_path}:\n'''\n{full_text}\n'''")

        # verbose_json
        text_segments = [""]
        times_to_keep = [[0, duration]]
        for segment in tqdm(transcript["segments"], unit="segment", desc="parsing"):
            st = segment["start"]
            ed = segment["end"]
            text = segment["text"]

            if not [re.search(stop, text) for stop in self.stop_list]:
                # not stopping
                text_segments[-1] += f" {text}"
                times_to_keep[-1][1] = ed
                continue

            for w in segment["words"]:
                word = w["word"]
                not_matched = True
                for stop in self.stop_list:
                    if re.search(stop, word):
                        whi(f"Found {stop} in {text} ({st}->{ed})")
                        times_to_keep[-1][1] = w["start"]
                        times_to_keep.append([w["end"], duration])
                        text_segments.append("")
                        not_matched = False
                        break
                if not_matched:
                    text_segments[-1] += f" {word}"
                    times_to_keep[-1][1] = duration

        n = len(text_segments)
        whi(f"Found {n} audio segments in {file_path}")

        for i, (start, end) in enumerate(times_to_keep):
            if end - start < 3:
                times_to_keep[i] = None
        times_to_keep = [t for t in times_to_keep if t is not None]

        n = len(text_segments)
        whi(f"Kept {n} audio segments when removing <3s in {file_path}")

        text_segments = [t.strip() for t in text_segments]

        assert len(times_to_keep) + 1 == len(text_segments), "invalid lengths"

        if len(times_to_keep) == 1:
            whi(f"Stopping there for {file_path} as there is no cutting to do")
            shutil.move(file_path, self.sp_dir / f"{file_path.name}_too_small.mp3")
            return

        audio = AudioSegment.from_mp3(file_path)

        for i, (start_cut, end_cut) in tqdm(enumerate(times_to_keep), unit="segment", desc="cutting"):
            sliced = audio[start_cut*1000:end_cut*1000]
            out_file = self.sp_dir / f"{file_path.name}_{today}_{i+1:03d}.mp3"
            assert not out_file.exists(), f"file {out_file} already exists!"
            trimmed = self.trim_silences(sliced)
            if len(trimmed) < 3 * 1000:
                red(f"Audio too short so ignored: {out_file} of length {len(trimmed)/1000:.1f}f")
                continue
            trimmed.export(out_file, format="mp3")
            whi(f"Saved sliced to {out_file}")

            # TODO fix metadata setting
            # for each file, keep the relevant transcript
            # whi(f"Setting metadata for {out_file}")
            # with exiftool.ExifToolHelper() as et:
            #     et.execute(b"-whisperx_transcript='" + bytes(text_segments[i].replace(" ", "\ ")) + b"'", str(out_file))
            #     et.execute(b"-transcription_date=" + bytes(int(time.time())), str(out_file))
            #     et.execute(b"-chunk_i=" + bytes(i), str(out_file))
            #     et.execute(b"-chunk_ntotal=" + bytes(n), str(out_file))

        whi(f"Moving {file_path} to {self.done_dir} dir")
        shutil.move(file_path, self.done_dir / file_path.name)

    def run_whisperx(self, audio_path):
        whi(f"Running whisperx on {audio_path}")
        with open(audio_path, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).hexdigest()

        whisperx_cached = stt_cache.cache(
                func=whisperx_splitter,
                ignore=["audio_path"])

        try:
            transcript = whisperx_cached(
                    audio_path=audio_path,
                    audio_hash=audio_hash,
                    prompt=self.prompt,
                    language=self.language,
                    )
            # TODO handle case where sound too long, must be cut
        except Exception as err:
            red(f"Exception when running whisperx: '{err}'")
            raise

        return transcript

    def trim_silences(self, audio):
        whi(f"Audio length before trimming silence: {len(audio)}ms")
        trimmed = audio[detect_leading_silence(audio):-detect_leading_silence(audio.reverse())]
        whi(f"Audio length after trimming silence: {len(trimmed)}ms")
        return trimmed


def whisperx_splitter(audio_path, audio_hash, prompt, language):
    whi("Starting replicate")
    start = time.time()
    transcript = replicate.run(
            "hnesk/whisper-wordtimestamps:4a60104c44dd709fc08a03dfeca6c6906257633dd03fd58663ec896a4eeba30e",
            input={
                "audio": open(audio_path, "rb"),
                "model": "large-v2",
                #"model": "medium",
                "language": language,
                "temperature": 0,
                "initial_prompt": prompt,
                "condition_on_previous_text": False,
                "word_timestamps": True,
                },
            )
    whi(f"Finished with replicate in {int(time.time()-start)} second")
    return transcript

if __name__ == "__main__":
    fire.Fire(AudioSplitter)
