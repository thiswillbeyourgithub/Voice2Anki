import shutil
import hashlib
import re
import joblib
import time
import pyexiftool
from tqdm import tqdm
import openai
import fire
from pathlib import Path
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

from .logger import whi, yel, red

assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()

stt_cache = joblib.Memory("transcript_cache", verbose=0)


def whisper_splitter(audio_path, audio_hash, prompt, language):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            prompt=prompt,
            language=language,
            response_format="verbose_json",
            )
    return transcript


class AudioSplitter:
    def __init__(
            self,
            prompt=None,
            stop_list=["stop", "nouvelles? cartes?", "nouvelles? questions?"],
            language="fr",
            n_todo=1,
            unsplitted_dir="./user_directory/unsplitted",
            splitted_dir="../user_directory/splitted",
            done_dir = "../user_directory/done",
            ):
        self.unsp_dir = Path(unsplitted_dir)
        self.sp_dir = Path(splitted_dir)
        self.done_dir = Path(done_dir)
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        assert isinstance(prompt, str), "prompt argument should be string"
        assert isinstance(n_todo, (float, int)) and n > 0, "n_todo should be a number greater than 0"

        self.prompt = prompt
        self.n_todo = n_todo
        self.language = language
        self.stop_list = [
                re.compile(s, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
                for s in stop_list]

        self.to_split = self.gather_todos()

        for file in tqdm(self.to_split, unit="file"):
            self.split_one_file(file)
            shutil.move(file, self.done_dir / file.name)

    def gather_todos(self):
        to_split = [p for p in self.unsp_dir.iterdir("*.mp3")]
        assert to_split, f"no mp3 found in {self.unsp_dir}"
        to_split = sorted(to_split, key=lambda x: x.stat().st_ctime)
        whi(f"Total number of files to split: {len(to_split)}")

        to_split = to_split[:self.n_todo]
        return to_split

    def split_one_file(self, file_path):
        # run whisper
        transcript = self.run_whisper(file_path)

        duration = transcript["duration"]
        whi(f"Duration of {file_path}: {duration}")

        full_text = transcript["text"]
        whi(f"Full text of {file_path}:\n'''\n{full_text}\n'''")

        # find where stop is pronounced
        text_segments = [""]
        cut_time = [0]
        for segment in tqdm(transcript["segments"], unit="segment", desc="parsing"):
            st = segment["start"]
            ed = segment["end"]
            text = segment["text"]

            text_segments[-1] += text
            if not [re.search(stop, text) for stop in self.stop_list]:
                whi(f"Found stop in {text} ({st}->{ed})")
                text_segments.append("")
                cut_time[-1] = ed
                cut_time.append(0)

        assert 0 not in cut_time, "0 found in cut_time"
        assert len(cut_time) == len(text_segments), "invalid lengths"

        n = len(text_segments)
        whi(f"Found {n} audio segments in {file_path}")

        audio = AudioSegment.from_mp3(file_path)

        prev_cut = 0
        for i, cut in tqdm(enumerate(cut_time), unit="segment", desc="cutting"):
            sliced = audio[prev_cut:cut]
            out_file = self.sp_dir / f"{file_path.name}_{i:03d}.mp3"
            assert not out_file.exists(), f"file {out_file} already exists!"
            sliced.export(out_file, format="mp3")
            whi(f"Sliced to {out_file}")
            prev_cut = cut

            # for each file, keep the relevant transcript and 
            metadata = {
                    "whisper_transcript": text_segments[i],
                    "transcription_date": time.time(),
                    "chunk nb": i,
                    "chunk total": n,
                    }
            whi(f"Setting metadata for {out_file}")
            with pyexiftool.ExifTool() as et:
                et.set_metadata(
                        out_file,
                        metadata)

    def run_whisper(self, audio_path):
        whi(f"Running whisper on {audio_path}")
        with open(audio_path, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).hexdigest()

        whisper_cached = stt_cache.cache(
                func=whisper_splitter,
                ignore=["audio_path"])

        try:
            transcript = whisper_cached(
                    audio_path=audio_path,
                    audio_hash=audio_hash,
                    prompt=self.prompt,
                    language=self.language,
                    )
            # TODO handle case where sound too long, must be cut
        except Exception as err:
            red(f"Exception when running whisper: '{err}'")

        return transcript

    def detect_silences(self, file_path):
        """if the audio file is too long, chunk it by cutting when there are
        silences"""
        # Set the minimum duration of silence (in milliseconds) to detect for cutting
        min_silence_duration = 2000

        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Split the audio based on silence periods
        chunks = split_on_silence(audio, min_silence_duration, silence_thresh=-40)

        return chunks

if __name__ == "__main__":
    fire.Fire(AudioSplitter)
