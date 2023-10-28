import soundfile as sf
import tempfile
import pyrubberband as pyrb
from datetime import datetime
import shutil
import hashlib
import re
import joblib
import time
# import exiftool
from tqdm import tqdm
import fire
from pathlib import Path
import os
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence

from logger import whi, yel, red

# replicate has to be imported after the api is loader
assert Path("REPLICATE_API_KEY.txt").exists(), "No api key found. Create a file REPLICATE8API_KEY.txt and paste your openai API key inside"
os.environ["REPLICATE_API_TOKEN"] = str(Path("REPLICATE_API_KEY.txt").read_text()).strip()
import replicate

stt_cache = joblib.Memory("transcript_cache", verbose=0)

d = datetime.today()
today = f"{d.day:02d}_{d.month:02d}"

class AudioSplitter:
    def __init__(
            self,
            prompt="Lecture de mes notes de cours separées par 'STOP' : ",
            stop_list=["stop", "nouvelles? cartes?", "nouvelles? questions?"],
            language="fr",
            n_todo=1,
            unsplitted_dir="./user_directory/unsplitted",
            splitted_dir="./user_directory/splitted",
            done_dir = "./user_directory/done",
            remove_silence=True,
            silence_method="sox",
            ):
        self.unsp_dir = Path(unsplitted_dir)
        self.sp_dir = Path(splitted_dir)
        self.done_dir = Path(done_dir)
        self.silence_method = silence_method
        assert silence_method in ["sox", "pydub"], "invalid silence_method"
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        assert isinstance(prompt, str), "prompt argument should be string"
        assert isinstance(n_todo, (float, int)) and n_todo > 0, "n_todo should be a number greater than 0"

        self.prompt = prompt
        self.n_todo = n_todo
        self.language = language
        self.remove_silence = remove_silence
        self.stop_list = [
                re.compile(s, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
                for s in stop_list]

        self.to_split = self.gather_todos()

        # removing silences
        if self.remove_silence:
            for i, file in tqdm(enumerate(self.to_split), unit="file"):
                if "unsilenced_" not in str(file):
                    new_filename = self.unsilence_audio(file)
                    assert "unsilenced_" in str(new_filename), "error"
                    self.to_split[i] = new_filename

        for file in tqdm(self.to_split, unit="file"):
            whi(f"Splitting file {file}")
            transcript = self.run_whisperx(file)
            times_to_keep, text_segments = self.split_one_transcript(transcript)

            if len(times_to_keep) == 1:
                whi(f"Stopping there for {file} as there is no cutting to do")
                shutil.move(file, self.sp_dir / f"{file.name}_too_small.mp3")
                return

            audio = AudioSegment.from_mp3(file)

            alterations = {}
            spf = 0.7  # speed factor
            for i, (t0, t1) in enumerate(times_to_keep):
                dur = t1 - t0
                if dur > 45:
                    red(f"Audio #{i} has too long duration: {dur}s.")
                    red(f"Text content: {text_segments[i]}\n")

                    # take the suspicious segment, slow it down and
                    # re analyse it
                    sub_audio = audio[t0 * 1000:t1 * 1000]
                    tempf = tempfile.NamedTemporaryFile(delete=False)
                    whi(f"Saving segment to {tempf.name} as wav")
                    # we need to use sf and pyrb because
                    # pydub is buggingly slow to change the speedup
                    sub_audio.export(tempf.name, format="wav")
                    whi("Stretching time")
                    y, sr = sf.read(tempf.name)
                    y2 = pyrb.time_stretch(y, sr, spf)
                    whi("Saving as wav")
                    sf.write(tempf.name, y2, sr, format='wav')
                    sub_audio = AudioSegment.from_wav(tempf.name)
                    whi("Resaving as mp3")
                    sub_audio.export(tempf.name, format="mp3")
                    transcript = self.run_whisperx(tempf.name)
                    sub_ttk, sub_ts = self.split_one_transcript(transcript)
                    new_times = [ [t0 + k / spf, t0 + v / spf] for k, v in sub_ttk]
                    alterations[i] = [new_times, sub_ts]
                    # Path(tempf.name).unlink()

            red(f"Found {len(alterations)} segments that needed slower analysis")
            for i, vals in tqdm(alterations.items(), desc="Fixing previously long audio"):
                new_times = vals[0]
                sub_ts = vals[1]
                old_len = len(times_to_keep)
                assert old_len == len(text_segments), "unexpected length"
                times_to_keep[i:i+1] = new_times
                text_segments[i:i+1] = sub_ts
                assert old_len + len(new_times) - 1 == len(times_to_keep), (
                    "Unexpected new length when altering audio")
                assert len(times_to_keep) == len(text_segments), "unexpected length"

            prev_t0 = 0
            prev_t1 = 0
            for i, (t0, t1) in enumerate(times_to_keep):
                dur = t1 - t0
                assert t0 > prev_t0 and t1 >= prev_t1, "overlapping audio!"
                if dur > 45:
                    red(f"Audio #{i} has too long duration even after correction! {dur}s.")
                    red(f"Text content: {text_segments[i]}\n")
                prev_t0 = t0
                prev_t1 = t1

            for i, (start_cut, end_cut) in tqdm(enumerate(times_to_keep), unit="segment", desc="cutting"):
                sliced = audio[start_cut*1000:end_cut*1000]
                out_file = self.sp_dir / f"{int(time.time())}_{today}_{file.name}_{i+1:03d}.mp3"
                assert not out_file.exists(), f"file {out_file} already exists!"
                if self.remove_silence:
                    sliced = self.trim_silences(sliced, 20)
                if len(sliced) < 1000:
                    red(f"Audio too short so ignored: {out_file} of length {len(sliced)/1000:.1f}s")
                    continue
                sliced.export(out_file, format="mp3")
                whi(f"Saved sliced to {out_file}")

                # TODO fix metadata setting
                # for each file, keep the relevant transcript
                # whi(f"Setting metadata for {out_file}")
                # with exiftool.ExifToolHelper() as et:
                #     et.execute(b"-whisperx_transcript='" + bytes(text_segments[i].replace(" ", "\ ")) + b"'", str(out_file))
                #     et.execute(b"-transcription_date=" + bytes(int(time.time())), str(out_file))
                #     et.execute(b"-chunk_i=" + bytes(i), str(out_file))
                #     et.execute(b"-chunk_ntotal=" + bytes(n), str(out_file))

            whi(f"Moving {file} to {self.done_dir} dir")
            shutil.move(file, self.done_dir / file.name)

    def gather_todos(self):
        to_split = [p for p in self.unsp_dir.rglob("*.mp3")]
        assert to_split, f"no mp3 found in {self.unsp_dir}"
        to_split = sorted(to_split, key=lambda x: x.stat().st_ctime)
        to_split = to_split[:self.n_todo]
        whi(f"Total number of files to split: {len(to_split)}")

        return to_split

    def split_one_transcript(self, transcript):
        duration = transcript["segments"][-1]["end"]
        whi(f"Duration: {duration}")
        # note: duration is not the total recording duration but rather the
        # time of the end of the last pronounced word

        full_text = transcript["transcription"]
        whi(f"Full text:\n'''\n{full_text}\n'''")

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
                # whi(f"Word: {word}")
                not_matched = True
                for stop in self.stop_list:
                    if re.search(stop, word):
                        # whi(f"Found {stop.pattern} in '{text}' ({st}->{ed})")
                        times_to_keep[-1][1] = w["start"]
                        times_to_keep.append([w["end"], duration])
                        text_segments.append("")
                        not_matched = False
                        break
                if not_matched:
                    text_segments[-1] += f" {word}"
                    times_to_keep[-1][1] = duration

        n = len(text_segments)
        whi(f"Found {n} audio segments")

        # remove too short
        for i, (start, end) in enumerate(times_to_keep):
            if end - start < 1:
                times_to_keep[i] = None
                text_segments[i] = None
            else:
                while "  " in text_segments[i]:
                    text_segments[i] = text_segments[i].replace("  ", " ").strip()
        text_segments = [t for t in text_segments if t is not None]
        times_to_keep = [t for t in times_to_keep if t is not None]
        n = len(text_segments)
        whi(f"Kept {n} audio segments when removing <1s")

        # remove almost no words
        for i, te in enumerate(text_segments):
            if len(te.split(" ")) <= 4 and "alfred" not in te.lower() and "image" not in te.lower():
                text_segments[i] = None
                times_to_keep[i] = None
        text_segments = [t for t in text_segments if t is not None]
        times_to_keep = [t for t in times_to_keep if t is not None]
        n = len(text_segments)
        whi(f"Kept {n} audio segments with > 4 words")

        text_segments = [t.strip() for t in text_segments]

        whi("Text segments found:")
        for i, t in enumerate(text_segments):
            whi(f"* {i:03d}: {t}")

        assert len(times_to_keep) == len(text_segments), "invalid lengths"

        return times_to_keep, text_segments

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

    def trim_silences(self, audio, db_threshold):
        whi(f"Audio length before trimming silence: {len(audio)}ms")
        threshold = - 10 ** (db_threshold / 20)  # in dFBS
        trimmed = audio[detect_leading_silence(audio, threshold):-detect_leading_silence(audio.reverse(), threshold)]
        whi(f"Audio length after trimming silence: {len(trimmed)}ms")
        return trimmed

    def unsilence_audio(self, file):
        whi(f"Removing silence from {file}")
        audio = AudioSegment.from_mp3(file)
        previous_len = len(audio) // 1000

        new_filename = file.parent / ("unsilenced_" + file.name)

        # pydub's way (very slow)
        if self.silence_method == "pydub":
            splitted = split_on_silence(
                    audio,
                    min_silence_len=500,
                    silence_thresh=-20,
                    seek_step=1,
                    keep_silence=500,
                    )
            new_audio = splitted[0]
            for chunk in splitted[1:]:
                new_audio += chunk
            new_audio.export(file.parent / ("unsilenced_" + file.name), format="mp3")
        elif self.silence_method == "sox":
            # sox way, fast but needs linux
            f1 = "\"" + str(file.name) + "\""
            f2 = "\"" + str(new_filename.name) + "\""
            d = "\"" + str(file.parent.absolute()) + "\""

            sox_cmd = f"cd {d} && rm tmpoutput*.mp3 ; sox {f1} tmpoutput.mp3 silence 1 1 0.1% 1 1 0.1% : newfile : restart && cat tmpoutput*.mp3 > {f2} && rm -v tmpout*.mp3"
            self.exec(sox_cmd)
            assert new_filename.exists(), f"new file not found: '{new_filename}'"
            new_audio = AudioSegment.from_mp3(new_filename)
        else:
            raise ValueError(self.silence_method)

        new_len = len(new_audio) // 1000
        red(f"Removed silence of {file} from {previous_len}s to {new_len}s")

        assert new_len >= 10, red("Suspiciously show new audio file, exiting.")

        whi(f"Moving {file} to {self.done_dir} dir")
        shutil.move(file, self.done_dir / file.name)

        return new_filename

    def exec(self, cmd):
        whi(f"Shell command: {cmd}")
        os.system(cmd)

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
                "no_speech_threshold": 1,
                },
            )
    whi(f"Finished with replicate in {int(time.time()-start)} second")
    return transcript

if __name__ == "__main__":
    fire.Fire(AudioSplitter)
