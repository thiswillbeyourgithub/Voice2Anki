import torchaudio
import copy
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

stt_cache = joblib.Memory("transcript_cache", verbose=1)

d = datetime.today()
today = f"{d.day:02d}_{d.month:02d}"

class AudioSplitter:
    def __init__(
            self,
            prompt,

            stop_list=["stop"],
            language="fr",
            n_todo=1,

            stop_source="replicate",

            unsplitted_dir="./user_directory/unsplitted",
            splitted_dir="./user_directory/splitted",
            done_dir="./user_directory/done",

            trim_splitted_silence=False,
            global_slowdown_factor=1.0,

            remove_silence=True,
            silence_method="torchaudio",
            ):
        self.unsp_dir = Path(unsplitted_dir)
        self.sp_dir = Path(splitted_dir)
        self.done_dir = Path(done_dir)
        assert silence_method in ["sox", "pydub", "torchaudio"], "invalid silence_method"
        assert self.unsp_dir.exists(), "missing unsplitted dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        assert isinstance(prompt, str), "prompt argument should be string"
        assert isinstance(n_todo, (float, int)) and n_todo > 0, "n_todo should be a number greater than 0"

        self.prompt = prompt
        self.n_todo = n_todo
        self.language = language
        self.stop_source = stop_source
        self.remove_silence = remove_silence
        self.trim_splitted_silence = trim_splitted_silence
        self.silence_method = silence_method
        assert global_slowdown_factor <= 1 and global_slowdown_factor > 0, (
                "invalid value for global_slowdown_factor")
        self.spf = global_slowdown_factor
        self.stop_list = [
                re.compile(s, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
                for s in stop_list]

        self.to_split = self.gather_todos()

        # removing silences
        if self.remove_silence:
            assert self.stop_source != "local_json", (
                "can't use local_json stop source and remove_silence")
            for i, file in tqdm(enumerate(self.to_split), unit="file"):
                if "unsilenced_" not in str(file):
                    new_filename = self.unsilence_audio(file)
                    assert "unsilenced_" in str(new_filename), "error"
                    self.to_split[i] = new_filename

        # contains the original file path, while self.to_split will contain
        # the path to the slowed down /tmp versions
        self.to_split_original = copy.deepcopy(self.to_split)

        # slow down a bit each audio
        if self.spf != 1.0:
            red(f"Global slowdown factor is '{self.spf}' so will slow down each audio file")
            assert self.stop_source != "local_json", (
                "can't use local_json stop source and slowdown")
            for i, file in enumerate(tqdm(self.to_split, unit="file", desc="Slowing down")):
                audio = AudioSegment.from_mp3(file)
                tempf = tempfile.NamedTemporaryFile(delete=False, prefix=file.stem + "__")
                whi(f"Saving slowed down {file} to {tempf.name} as wav")
                # we need to use sf and pyrb because
                # pydub is buggingly slow to change the speedup
                audio.export(tempf.name, format="wav")
                whi("  Stretching time of wav")
                y, sr = sf.read(tempf.name)
                y2 = pyrb.time_stretch(y, sr, self.spf)
                whi("  Saving streched wav")
                sf.write(tempf.name, y2, sr, format='wav')
                sub_audio = AudioSegment.from_wav(tempf.name)
                speed_ratio = len(sub_audio) / len(audio)
                assert abs(1 - speed_ratio / self.spf) <= 0.0001, (
                    f"The slowdown factor is different than asked: '{speed_ratio}'")
                whi("  Resaving as mp3")
                sub_audio.export(tempf.name, format="mp3")
                self.to_split[i] = tempf.name
        else:
            self.spf = 1

        # splitting the long audio
        for ii, file in tqdm(enumerate(self.to_split), unit="file"):
            whi(f"Splitting file {file}")
            if self.stop_source == "replicate":
                transcript = self.run_whisperx(file)
                times_to_keep, text_segments = self.split_one_transcript(transcript)
            elif self.stop_source == "local_json":
                raise NotImplementedError
            else:
                raise ValueError(self.stop_source)

            fileo = self.to_split_original[ii]

            if len(times_to_keep) == 1:
                whi(f"Stopping there for {fileo} as there is no cutting to do")
                shutil.move(fileo, self.sp_dir / f"{fileo.stem}_too_small.{fileo.suffix}")
                continue

            audio = AudioSegment.from_mp3(file)

            whi("\nChecking if some splits are too long")
            alterations = {}
            spf = 0.7  # speed factor
            n = len(times_to_keep)
            for i, (t0, t1) in enumerate(times_to_keep):
                dur = t1 - t0
                if dur > 45:
                    red(f"Split #{i}/{n} has too long duration: {dur}s.")
                    red(f"Text content: {text_segments[i]}\n")

                    # take the suspicious segment, slow it down and
                    # re analyse it
                    sub_audio = audio[t0 * 1000:t1 * 1000]
                    tempf = tempfile.NamedTemporaryFile(delete=False, prefix=fileo.stem + "__")

                    # sf and pyrb way:
                    # we need to use sf and pyrb because
                    # pydub is buggingly slow to change the speedup
                    whi(f"Saving segment to {tempf.name} as wav")
                    sub_audio.export(tempf.name, format="wav")
                    whi("Stretching time")
                    y, sr = sf.read(tempf.name)
                    y2 = pyrb.time_stretch(y, sr, spf)
                    whi("Saving as wav")
                    sf.write(tempf.name, y2, sr, format='wav')
                    sub_audio = AudioSegment.from_wav(tempf.name)
                    whi("Resaving as mp3")
                    sub_audio.export(tempf.name, format="mp3")

                    # # pydub way:
                    # whi(f"Saving segment to {tempf.name} as mp3")
                    # sub_audio.speedup(spf, chunk_size=300).export(tempf.name, format="mp3")
                    # whi("Saved")

                    transcript = self.run_whisperx(tempf.name)
                    sub_ttk, sub_ts = self.split_one_transcript(transcript)
                    new_times = [[t0 + k * spf, t0 + v * spf] for k, v in sub_ttk]
                    alterations[i] = [new_times, sub_ts]
                    assert new_times[-1][-1] <= t1, "unexpected split timeline"
                    # Path(tempf.name).unlink()

            red(f"Found {len(alterations)} segments that needed slower analysis")
            for i, vals in tqdm(alterations.items(), desc="Fixing previously long split"):
                new_times = vals[0]
                sub_ts = vals[1]

                # find the corresponding segment: it's when the start
                # time is very close
                old_times = None
                for j, old_vals in enumerate(times_to_keep):
                    if abs(old_vals[0] - new_times[0][0]) <= 0.1:
                        old_times = times_to_keep[j]
                        break
                assert old_times

                old_len = len(times_to_keep)
                assert old_len == len(text_segments), "unexpected length"
                assert abs(old_vals[0] - new_times[0][0]) <= 0.1, "start time are different!"

                if len(new_times) == 1:
                    whi(f"The slowed down split #{i} is not split "
                        "differently than the original so keeping the "
                        f"original: {old_times} vs {new_times}")
                    assert abs(1 - old_vals[1] / new_times[0][1]) <= 0.05, "end times are different!"
                else:
                    whi(f"Found {len(new_times)} new splits inside split #{i}/{n}")

                    times_to_keep[j:j+1] = new_times
                    text_segments[j:j+1] = sub_ts
                    assert old_len + len(new_times) - 1 == len(times_to_keep), (
                        "Unexpected new length when altering audio")
                    assert len(times_to_keep) == len(text_segments), "unexpected length"

            prev_t0 = -1
            prev_t1 = -1
            n = len(times_to_keep)
            whi("\nChecking if some splits are still too long")
            for i, (t0, t1) in enumerate(times_to_keep):
                dur = t1 - t0
                assert t0 > prev_t0 and t1 >= prev_t1, "overlapping splits!"
                if dur > 45:
                    red(f"Split #{i}/{n} has too long duration even after correction! {dur}s.")
                    red(f"Text content: {text_segments[i]}\n")
                prev_t0 = t0
                prev_t1 = t1

            audio_o = AudioSegment.from_mp3(fileo)
            assert abs(1 - (times_to_keep[-1][1] * 1000 * self.spf) / len(audio_o)) <= 0.01
            for i, (start_cut, end_cut) in tqdm(enumerate(times_to_keep), unit="segment", desc="cutting"):
                sliced = audio_o[start_cut*1000 * self.spf:end_cut*1000 * self.spf]
                out_file = self.sp_dir / f"{int(time.time())}_{today}_{fileo.stem}_{i+1:03d}.mp3"
                assert not out_file.exists(), f"file {out_file} already exists!"
                if self.trim_splitted_silence:
                    sliced = self.trim_silences(sliced)
                if len(sliced) < 1000:
                    red(f"Split too short so ignored: {out_file} of length {len(sliced)/1000:.1f}s")
                    continue
                whi(f"Saving sliced to {out_file}")
                sliced.export(out_file, format="mp3")

                # TODO fix metadata setting
                # for each file, keep the relevant transcript
                # whi(f"Setting metadata for {out_file}")
                # with exiftool.ExifToolHelper() as et:
                #     et.execute(b"-whisperx_transcript='" + bytes(text_segments[i].replace(" ", "\ ")) + b"'", str(out_file))
                #     et.execute(b"-transcription_date=" + bytes(int(time.time())), str(out_file))
                #     et.execute(b"-chunk_i=" + bytes(i), str(out_file))
                #     et.execute(b"-chunk_ntotal=" + bytes(n), str(out_file))

            whi(f"Moving {fileo} to {self.done_dir} dir")
            shutil.move(fileo, self.done_dir / fileo.name)

    def gather_todos(self):
        to_split = [p for p in self.unsp_dir.iterdir() if "mp3" in p.suffix or "wav" in p.suffix]
        assert to_split, f"no mp3/wav found in {self.unsp_dir}"
        # to_split = sorted(to_split, key=lambda x: x.stat().st_mtime)
        to_split = sorted(to_split, key=lambda x: x.name)
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
                        times_to_keep[-1][1] = (w["start"] + w["end"]) / 2
                        times_to_keep.append([w["end"], duration])
                        text_segments.append("")
                        not_matched = False
                        break
                if not_matched:
                    text_segments[-1] += f" {word}"
                    times_to_keep[-1][1] = duration

        n = len(text_segments)
        whi(f"Found {n} splits")

        # # remove too short
        # latest_kept_i = 0
        # for i, (start, end) in enumerate(times_to_keep):
        #     assert end - start >= 0
        #     if end - start < 1:

        #         assert times_to_keep[latest_kept_i][1] <= end
        #         times_to_keep[latest_kept_i][1] = end

        #         times_to_keep[i] = None
        #         text_segments[i] = None
        #     else:
        #         latest_kept_i = i
        #         while "  " in text_segments[i]:
        #             text_segments[i] = text_segments[i].replace("  ", " ").strip()
        # text_segments = [t for t in text_segments if t is not None]
        # times_to_keep = [t for t in times_to_keep if t is not None]
        # n = len(text_segments)
        # whi(f"Kept {n} splits when removing <1s")

        # # remove almost no words
        # for i, te in enumerate(text_segments):
        #     if len(te.split(" ")) <= 4 and "alfred" not in te.lower() and "image" not in te.lower():
        #         text_segments[i] = None
        #         times_to_keep[i] = None
        # text_segments = [t for t in text_segments if t is not None]
        # times_to_keep = [t for t in times_to_keep if t is not None]
        # n = len(text_segments)
        # whi(f"Kept {n} splits with > 4 words")

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

        try:
            transcript = whisperx_splitter(
                    audio_path=str(audio_path),
                    audio_hash=audio_hash,
                    prompt=self.prompt,
                    language=self.language,
                    model="medium",
                    )
            # TODO handle case where sound too long, must be cut
        except Exception as err:
            red(f"Exception when running whisperx: '{err}'")
            raise

        return transcript

    def trim_silences(self, audio, dbfs_threshold=-50, depth=0):
        if depth >= 10:
            red("Recursion limit of self.trim_silences reached, not trimming this split.")
            return audio
        # pydub's default DBFs default is -50
        whi(f"Audio length before trimming silence: {len(audio)}ms")

        # trim only the beginning
        trimmed = audio[detect_leading_silence(audio, dbfs_threshold):]

        # trim the end
        # trimmed = trimmed[:-detect_leading_silence(trimmed.reverse(), dbfs_threshold)]

        ln = len(trimmed)
        whi(f"Audio length after trimming silence: {ln}ms (depth={depth}, threshold={dbfs_threshold})")
        if ln == 0:
            red("Trimming silence is way too harsch on this file, changing threshold a lot")
            return self.trim_silences(audio, dbfs_threshold=dbfs_threshold - 10, depth=depth + 1)
        if ln <= 1000 or len(audio) / ln >= 3:
            red("Trimming silence of audio would be too harsh so reducing threshold")
            return self.trim_silences(audio, dbfs_threshold=dbfs_threshold - 5, depth=depth + 1)
        else:
            return trimmed

    def unsilence_audio(self, file):
        whi(f"Removing silence from {file}")

        audio = AudioSegment.from_mp3(file)
        new_filename = file.parent / ("unsilenced_" + file.name)
        previous_len = len(audio) // 1000

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
            new_audio.export(new_filename, format="mp3")

        elif self.silence_method == "sox":
            # sox way, fast but needs linux
            f1 = "\"" + str(file.name) + "\""
            f2 = "\"" + str(new_filename.name) + "\""
            d = "\"" + str(file.parent.absolute()) + "\""

            sox_cmd = f"cd {d} && rm tmpoutput*.mp3 ; sox {f1} tmpoutput.mp3 silence 1 3 1.0% 1 3 1.0% : newfile : restart && cat tmpoutput*.mp3 > {f2} && rm -v tmpout*.mp3"
            self.exec(sox_cmd)
            assert new_filename.exists(), f"new file not found: '{new_filename}'"
            new_audio = AudioSegment.from_mp3(new_filename)

        elif self.silence_method == "torchaudio":
            # load from file
            waveform, sample_rate = torchaudio.load(file)

            # alternative vad using torchaudio sox:
            sox_effects = [
                    # max silence should be 2s
                    ["silence", "-l", "1", "0.1", "0.01%", "-1", "1.0", "0.01%"],
                    ]

            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                    waveform,
                    sample_rate,
                    sox_effects,
                    )

            # write to wav, then convert to mp3
            sf.write(str(new_filename), waveform.numpy().T, sample_rate, format='wav')
            temp = AudioSegment.from_wav(new_filename)
            temp.export(new_filename, format="mp3")

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


@stt_cache.cache(ignore=["audio_path"])
def whisperx_splitter(audio_path, audio_hash, prompt, language, model="large-v2"):
    whi("Starting replicate (meaning cache is not used)")
    start = time.time()
    transcript = replicate.run(
            "hnesk/whisper-wordtimestamps:4a60104c44dd709fc08a03dfeca6c6906257633dd03fd58663ec896a4eeba30e",
            input={
                "audio": open(audio_path, "rb"),
                "model": model,
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
