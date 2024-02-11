import uuid
import textwrap
import json
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
import replicate

from logger import whi, yel, red, shared
from profiles import ValueStorage

stt_cache = joblib.Memory("cache/audio_splitter_cache", verbose=1)

d = datetime.today()
today = f"{d.day:02d}_{d.month:02d}"

class AudioSplitter:
    def __init__(
            self,
            prompt=None,
            profile=None,

            stop_list=[" stop", " top"],
            language="fr",
            n_todo=1,

            stop_source="replicate",

            untouched_dir=None,
            splitted_dir=None,
            done_dir=None,

            trim_splitted_silence=False,
            global_slowdown_factor=1.0,
            second_pass_slowdown_factor=1.0,

            remove_silence=True,
            silence_method="torchaudio",
            h=False,
            help=False,
            ):
        """
        prompt: str, default None
            prompt used to guide whisper. None to disable

        profile: str, default None
            name of the profile to use when looking for the audio to split
            as well as the replicate API key.
            if None, will use the latest_profile used by Voice2Anki

        stop_list: list, default [' stop', ' top']
            list of strings that when found will trigger the audio splitting.

        language: str, default fr

        n_todo: int, default 1
            number of files to split by default. Can be used to only process
            a certain batch.

        stop_source: str, default 'replicate'
            if 'local_json', then an output from whispercpp is expected. This
            is not yet implemented. The idea is to be able to do the audio
            splitting locally using whispercpp instead of relying on replicate.

        untouched_dir: str or Path, default  to profiles/PROFILE/queues/audio_untouched

        splitted_dir: see untouched_dir

        done_dir: see untouched_dir

        trim_splitted_silence: bool, default False
            if True, will try to remove leading and ending silence of each
            audio split before exporting them. This can cause problems with
            words being cut so I disabled it by default. If the file after
            trimming is unexpectedly short, the trimming will be ignored. This
            is so that loud splits that don't contain silence are not botched.

        global_slowdown_factor float, default 1.0
            if lower than 1.0, then the long audio will be slowed down before
            processing. This can help whisper.

        second_pass_slowdown_factor float, default 1.0
            like global_slowdown_factor but for the second pass

        silence_method, str, default 'torchaudio'
            can be any of 'torchaudio', 'pydub' or 'sox_cli'
            * 'torchaudio' works using the sox filters present in utils/shared_module.py
            * 'sox_cli' only works on linux and needs sox installed
            * 'pydub' is excrutiatingly slow
        """
        if h or help:
            return help(self)

        if profile is None:
            shared.pv = ValueStorage()
            profile = shared.pv.profile_name
        else:
            shared.pv = ValueStorage(profile)
        red(f"Will use profile {shared.pv.profile_name}")

        # replicate has to be imported after the api is loader
        assert shared.pv["txt_replicate_api_key"].strip(), f"Missing replicate api key for profile {shared.pv.profile_name}. You must open Voice2Anki.py and set it in the settings."
        os.environ["REPLICATE_API_TOKEN"] = shared.pv["txt_replicate_api_key"].strip()

        if untouched_dir is None:
            untouched_dir = f"profiles/{profile}/queues/audio_untouched"
        if splitted_dir is None:
            splitted_dir = f"profiles/{profile}/queues/audio_splits"
        if done_dir is None:
            done_dir = f"profiles/{profile}/queues/audio_done"
        self.unsp_dir = Path(untouched_dir)
        self.sp_dir = Path(splitted_dir)
        self.done_dir = Path(done_dir)
        self.metadata_file = self.sp_dir / "metadata.txt"
        assert silence_method in ["sox_cli", "pydub", "torchaudio"], "invalid silence_method"
        assert self.unsp_dir.exists(), "missing untouched dir"
        assert self.sp_dir.exists(), "missing splitted dir"
        assert self.done_dir.exists(), "missing done dir"

        assert isinstance(prompt, (str, type(None))), "prompt argument should be string"
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
        assert second_pass_slowdown_factor <= 1 and second_pass_slowdown_factor > 0, (
                "invalid value for second_pass_slowdown_factor")
        self.g_spf = global_slowdown_factor
        self.l_spf = second_pass_slowdown_factor
        self.stop_list = [
                re.compile(s, flags=re.DOTALL | re.MULTILINE | re.IGNORECASE)
                for s in stop_list]

        self.to_split = self.gather_todos()

        # removing silences
        if self.remove_silence:
            assert self.stop_source != "local_json", (
                "can't use local_json stop source and remove_silence")
            for i, file in enumerate(tqdm(self.to_split, unit="file")):
                if "_unsilenced" not in str(file):
                    new_filename = self.unsilence_audio(file)
                    assert "_unsilenced" in str(new_filename), "error"
                    self.to_split[i] = new_filename

        # contains the original file path, while self.to_split will contain
        # the path to the slowed down versions in /tmp
        self.to_split_original = copy.deepcopy(self.to_split)

        # slow down a bit each audio
        if self.g_spf != 1.0:
            red(f"Global slowdown factor is '{self.g_spf}' so will slow down each audio file")
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
                y2 = pyrb.time_stretch(y, sr, self.g_spf)
                whi("  Saving streched wav")
                sf.write(tempf.name, y2, sr, format='wav')
                sub_audio = AudioSegment.from_wav(tempf.name)
                speed_ratio = len(sub_audio) / len(audio)
                assert abs(1 - speed_ratio / self.g_spf) <= 0.0001, (
                    f"The slowdown factor is different than asked: '{speed_ratio}'")
                whi("  Resaving as mp3")
                sub_audio.export(tempf.name, format="mp3")
                self.to_split[i] = tempf.name

        # splitting the long audio
        for iter_file, file in enumerate(tqdm(self.to_split, unit="file", desc="Splitting file", disable=not bool(len(self.to_split)-1))):
            whi(f"Splitting file {file}")
            if self.stop_source == "replicate":

                transcript = self.run_whisper(file, second_pass=False)
                times_to_keep, metadata = self.split_one_transcript(transcript, False)
                whi("Text segments metadata:")
                for i, t in enumerate(metadata):
                    whi(f"* {i:03d}:")
                    for k, v in t.items():
                        whi(textwrap.indent(f"{k}: {v}", "    "))

            elif self.stop_source == "local_json":
                raise NotImplementedError
            else:
                raise ValueError(self.stop_source)

            audio = AudioSegment.from_mp3(file)
            fileo = self.to_split_original[iter_file]  # original file
            audio_o = AudioSegment.from_mp3(fileo)  # original audio, without slowing down

            if len(times_to_keep) == 1:
                whi(f"Stopping there for {fileo} as there is no cutting to do")
                shutil.move(fileo, self.sp_dir / f"{fileo.stem}_too_small.{fileo.suffix}")
                continue

            # check overlap
            prev_t0 = -1
            prev_t1 = -1
            n = len(times_to_keep)
            for iter_ttk, val in enumerate(times_to_keep):
                if val is None:
                    continue
                t0, t1 = val
                assert t0 > prev_t0 and t1 >= prev_t1, "overlapping splits!"
                prev_t0 = t0
                prev_t1 = t1

            whi("\nSecond pass")
            alterations = {}
            n = len(times_to_keep)

            # load all sub_audios and send them to whisper directly
            # so that it is already cached for the loop just afterwards
            sub_audios = [
                    audio_o[val[0] * 1000 * self.g_spf:val[1] * 1000 * self.g_spf]
                    if val is not None
                    else None
                    for val in times_to_keep
                    ]
            tempfiles = [
                tempfile.NamedTemporaryFile(delete=False, prefix=fileo.stem + f"__{i}_").name
                if a is not None
                else None
                for i, a in enumerate(sub_audios)
                ]

            def threaded_export(audio, path):
                if audio is None or path is None:
                    assert audio is None and path is None
                    return
                if self.l_spf != 1.0:
                    tempf = tempfile.NamedTemporaryFile(delete=False, prefix=fileo.stem + "__")
                    # sf and pyrb way:
                    # we need to use sf and pyrb because
                    # pydub is buggingly slow to change the speedup
                    audio.export(tempf.name, format="wav")
                    # Stretching time
                    y, sr = sf.read(tempf.name)
                    y2 = pyrb.time_stretch(y, sr, self.l_spf)
                    # Saving as wav
                    sf.write(tempf.name, y2, sr, format='wav')
                    audio = AudioSegment.from_wav(tempf.name)

                    # # pydub way:
                    # whi(f"Saving segment to {tempf.name} as mp3")
                    # sub_audio.speedup(spf, chunk_size=300).export(tempf.name, format="mp3")
                    # whi("Saved")
                audio.export(path, format="mp3")
            joblib.Parallel(
                    n_jobs=10,
                    backend="threading",
                    )(joblib.delayed(threaded_export)(sub, f)
                        for sub, f in zip(
                            tqdm(
                                sub_audios,
                                desc="Exporting before second pass",
                                unit="mp3",
                                ),
                            tempfiles,
                            ))

            # run whisper on each split
            def threaded_whisper(path):
                if path is None:
                    return None
                transcript = self.run_whisper(audio_path=path, second_pass=True)
                Path(path).unlink()
                return transcript
            split_transcripts = joblib.Parallel(
                    n_jobs=-1,
                    backend="threading",
                    )(
                            joblib.delayed(threaded_whisper)(tf)
                            for tf in tqdm(
                                tempfiles,
                                unit="mp3",
                                desc="Transcribing for second pass",
                                )
                            )
            assert len(times_to_keep) == len(sub_audios), "Error when caching in advance sub audio"
            for iter_ttk, val in enumerate(tqdm(times_to_keep, desc="Second pass", unit="mp3")):
                if val is None:
                    continue
                iter_print = f"* {iter_ttk}/{n} "
                t0, t1 = val
                dur = t1 - t0
                whi(f"{iter_print}Text content before second pass: {metadata[iter_ttk]['text']}")

                transcript = split_transcripts[iter_ttk]
                assert transcript is not None
                sub_ttk, sub_meta = self.split_one_transcript(transcript, True)
                if not sub_ttk and not sub_meta:
                    red(f"{iter_print}Audio between {t0} and {t1} seems empty after second pass. Keeping results from first pass.")
                    continue

                # adjusting times so that the second pass is synced with the original
                # also check for overlap
                prev_t0 = -1
                prev_t1 = -1
                new_times = []
                for val, met in zip(sub_ttk, sub_meta):
                    met["start"] = t0 + met["start"] * self.l_spf * self.g_spf
                    met["end"] = t0 + met["end"] * self.l_spf * self.g_spf
                    assert met["start"] > prev_t0 and met["end"] >= prev_t1, "overlap"
                    if val is None:
                        new_times.append(val)
                    else:
                        new_times.append([t0 + val[0] * self.l_spf * self.g_spf, t0 + val[1] * self.l_spf * self.g_spf])
                        assert new_times[-1][0] > prev_t0 and new_times[-1][1] >= prev_t1, "overlap"
                        assert new_times[-1][0] == met["start"], "Inconsistency between metadata and times_to_keep"
                        assert new_times[-1][1] == met["end"], "Inconsistency between metadata and times_to_keep"

                        prev_t0 = met["start"]
                        prev_t1 = met["end"]

                alterations[iter_ttk] = [new_times, sub_meta]
                if [nt for nt in new_times if nt]:
                    assert [nt for nt in new_times if nt][-1][-1] <= t1, "unexpected split timeline"

                if len(sub_meta) > 1:
                    red(f"{iter_print}Segment was rescinded in those texts. Metadata:")
                    for meta in sub_meta:
                        red(f"* '{meta}'")
                elif sub_meta[0]["text"] != metadata[iter_ttk]["text"]:
                    red(f"{iter_print}Text segment after second pass is: '{sub_meta[0]['text']}'")
                else:
                    whi(f"{iter_print}No change after second pass")

            red(f"Resplitting after second pass")
            for iter_alt, vals in tqdm(alterations.items(), desc="Resplitting"):
                new_times = vals[0]
                sub_meta = vals[1]
                new_times_real = [val for val in new_times if val is not None]
                if not new_times_real:
                    metadata[iter_alt]["status"] += "2nd pass considered it empty"
                    metadata[iter_alt]["2nd_pass_metadata"] = sub_meta
                    continue

                # find the corresponding segment: it's when the start
                # time is very close
                diffs = []
                for j, old_vals in enumerate(times_to_keep):
                    if old_vals is None:
                        diffs.append(None)
                    else:
                        diffs.append(abs(old_vals[0] - new_times_real[0][0]))
                min_diff = min([d for d in diffs if d is not None])
                assert len(diffs) == len(times_to_keep)
                i_good_seg = diffs.index(min_diff)
                iter_print = f"* {i_good_seg}/{len(times_to_keep)} "
                old_times = times_to_keep[i_good_seg]
                dur_old = old_times[1] - old_times[0]
                dur_new = new_times_real[-1][1] - new_times_real[0][0]
                diff_dur = abs(1 - dur_old / dur_new)
                if not (min_diff <= 2 or diff_dur <= 0.15):
                    red(f"{iter_print}Suspiciously big difference: min_diff: {min_diff}; diff_dur: {diff_dur}; old_times: {old_times}; new_times: {new_times}")

                assert len(new_times) == len(sub_meta)
                old_len_ttk = len(times_to_keep)
                assert old_len_ttk == len(metadata), "unexpected length"

                if len(new_times_real) == 1:
                    whi(f"{iter_print}The split is not split "
                        "differently than the first pass so keeping the "
                        f"original: {old_times} vs {new_times[0]}")
                    metadata[i_good_seg]["2nd_pass_metadata"] = sub_meta
                    metadata[i_good_seg]["status"] += "2nd split contained only one split"
                else:
                    whi(f"{iter_print}Found {len(new_times)} new splits inside split")

                    times_to_keep[i_good_seg] = None
                    metadata[i_good_seg]["status"] += "Replaced by 2nd pass"
                    times_to_keep[i_good_seg:i_good_seg+1] = new_times
                    for sm in sub_meta:
                        sm["1st_pass_metadata"] = metadata[i_good_seg:i_good_seg+1]
                    metadata[i_good_seg:i_good_seg+1] = sub_meta
                    assert old_len_ttk + len(new_times) - 1 == len(times_to_keep), (
                        "Unexpected new length when resplitting audio")
                    assert len(times_to_keep) == len(metadata), "unexpected length"

            # check values
            prev_t0 = -1
            prev_t1 = -1
            n = len(times_to_keep)
            whi("\nChecking if some splits are too long")
            for iter_ttk, val in enumerate(times_to_keep):
                if val is None:
                    continue
                t0, t1 = val
                dur = t1 - t0
                assert t0 > prev_t0 and t1 >= prev_t1, "overlapping splits!"
                if dur > 45:
                    red(f"Split #{iter_ttk}/{n} has too long duration even after second pass: {dur:.2f}s.")
                    red(f"metadata: {metadata[iter_ttk]}")
                prev_t0 = t0
                prev_t1 = t1

            assert [t for t in times_to_keep if t][-1][1] * 1000 * self.g_spf <= len(audio_o)
            # # make sure to start at 0 and end at the end. Even though if
            # # it was removed from times_to_keep means that it
            # # contained no words
            # times_to_keep[-1][1] = len(audio_o) / 1000 / self.g_spf
            # times_to_keep[0][0] = 0

            ignored = AudioSegment.empty()
            prev_end = 0
            for iter_ttk, val in enumerate(tqdm(times_to_keep, unit="segment", desc="cutting")):
                if val is not None:
                    time_markers = f"_{int(val[0])}s_to_{int(val[1])}s"
                else:
                    time_markers = ""

                begin_time = time.time()
                out_file = self.sp_dir / f"{int(time.time())}_HASH_{fileo.stem}_{iter_ttk+1:03d}{time_markers}.mp3"
                assert not out_file.exists(), f"File {out_file} already exists!"

                with self.metadata_file.open("a") as mf:
                    mf.write("\n")
                    metadata[iter_ttk]["file_path"] = str(out_file)
                    mf.write(json.dumps(metadata[iter_ttk], ensure_ascii=False))

                if val is None:
                    continue

                start_cut, end_cut = val

                # assemble all ignored audios as one single slice too
                ignored += audio_o[prev_end*1000: start_cut*1000]
                prev_end = end_cut

                sliced = audio_o[start_cut*1000 * self.g_spf:end_cut*1000 * self.g_spf]
                if self.trim_splitted_silence:
                    sliced = self.trim_silences(sliced)
                # if len(sliced) < 1000:
                #     red(f"Split too short so ignored: {out_file} of length {len(sliced)/1000:.1f}s")
                #     continue
                whi(f"Saving sliced to {out_file}")
                sliced.export(out_file, format="mp3")

                # rename to replace HASH by its hash
                with open(out_file, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()[:10]
                assert out_file.count("HASH") == 1, f"Unexpected name: {out_file}"
                shutil.move(out_file, out_file.replace("HASH", h))

                # make sure to wait at least 1.1s otherwise the order of
                # the audio can be wrong because the timestamps are to
                # the second
                time.sleep(max(0, 1.1 - (time.time() - begin_time)))

            whi(f"Length of ignored sections before trimming silences: '{len(ignored)//1000}s'")
            ignored = self.trim_silences(ignored)
            whi(f"Length of ignored sections after trimming silences: '{len(ignored)//1000}s'")
            out_file = self.sp_dir / f"{int(time.time())}_HASH_{fileo.stem}_{iter_ttk+2:03d}_IGNORED.mp3"
            assert not out_file.exists(), f"File {out_file} already exists!"
            ignored.export(out_file, format="mp3")

            # rename to replace HASH by its hash
            with open(out_file, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()[:10]
            assert out_file.count("HASH") == 1, f"Unexpected name: {out_file}"
            shutil.move(out_file, out_file.replace("HASH", h))

            whi(f"Moving {fileo} to {self.done_dir} dir")
            whi("Copying")
            shutil.copy2(fileo, self.done_dir / (fileo.name + "temp"))
            whi("Renaming")
            shutil.move(
                    self.done_dir / (fileo.name + "temp"),
                    self.done_dir / fileo.name)
            whi("Removing")
            fileo.unlink(missing_ok=False)
            whi("Done!")

        red("All done!")
        return

    def gather_todos(self):
        to_split = [p for p in self.unsp_dir.iterdir() if "mp3" in p.suffix or "wav" in p.suffix]
        assert to_split, f"no mp3/wav found in {self.unsp_dir}"
        # to_split = sorted(to_split, key=lambda x: x.stat().st_mtime)
        to_split = sorted(to_split, key=lambda x: x.name)
        to_split = to_split[:self.n_todo]
        whi(f"Total number of files to split: {len(to_split)}")

        return to_split

    def split_one_transcript(self, transcript, second_pass):
        duration = transcript["segments"][-1]["end"]
        full_text = transcript["transcription"]
        if not second_pass:
            whi(f"Duration: {duration}")
            # note: duration is not the total recording duration but rather the
            # time of the end of the last pronounced word
            whi(f"Full text:\n'''\n{full_text}\n'''")

        # verbose_json
        times_to_keep = [[0, duration]]
        previous_start = -1
        previous_end = -1
        metadata = [{"text": "", "start": 0, "end": duration, "status": "", "repo": transcript["repo"], "modelname": transcript["modelname"]}]
        for iter_seg, segment in enumerate(tqdm(transcript["segments"], unit="segment", desc="parsing", disable=True if second_pass else False)):

            st = segment["start"]
            ed = segment["end"]

            text = segment["text"]
            if not second_pass:
                whi(f"* {iter_seg:03d} Text segment: {text}")
                metadata[-1]["n_pass"] = 1
            else:
                metadata[-1]["n_pass"] = 2

            if transcript["repo"] != "fast":
                # impossibly short token
                if ed - st <= 0.05:
                    red(f"Too short segment is ignored: {ed-st}s (text was '{text}')")
                    metadata[-1]["status"] = "Too short"
                    continue

                # store whisper metadata
                metadata[-1]["no_speech_prob"] = segment["no_speech_prob"]
                metadata[-1]["avg_logprob"] = segment["avg_logprob"]
                metadata[-1]["compression_ratio"] = segment["compression_ratio"]
                metadata[-1]["temperature"] = segment["temperature"]

                # low speech probability
                nsprob = segment["no_speech_prob"]
                if nsprob >= 0.9:
                    red(f"No speech probability is {nsprob}%>90% so ignored. Text was '{text}'")
                    metadata[-1]["status"] = "No speech"
                    continue

                if segment["temperature"] == 1:
                    red(f"Temperature at 1 and no speech probability at {nsprob}%>40% so ignored. Text was '{text}'")
                    metadata[-1]["status"] = "No speech at high temp"
                    continue

            assert st >= previous_start, "Output from whisper contains overlapping segments"
            assert ed >= previous_end, "Output from whisper contains overlapping segments"
            assert ed >= previous_start, "Output from whisper contains overlapping segments"
            previous_start = st
            previous_end = ed

            if not [re.search(stop, text) for stop in self.stop_list]:
                # not stopping
                metadata[-1]["text"] += f" {text}"
                metadata[-1]["end"] = ed
                times_to_keep[-1][1] = ed
                continue

            for w in segment["words"]:
                word = w["word"]
                # whi(f"Word: {word}")
                not_matched = True
                for stop in self.stop_list:
                    if re.search(stop, word):
                        # whi(f"Found {stop.pattern} in '{text}' ({st}->{ed})")
                        times_to_keep[-1][1] = w["end"]
                        metadata[-1]["end"] = times_to_keep[-1][1]
                        metadata[-1]["status"] = "Kept"

                        times_to_keep.append([w["end"], duration])
                        metadata.append({"text": "", "start": w["end"], "end": duration, "status": "", "repo": transcript["repo"], "modelname": transcript["modelname"]})
                        not_matched = False
                        break
                if not_matched:
                    metadata[-1]["text"] += f" {word}"
                    times_to_keep[-1][1] = duration
                    metadata[-1]["end"] = duration

        if not metadata[-1]["status"]:
            metadata[-1]["status"] = "Kept"
        n = len(metadata)
        if not second_pass:
            whi(f"Found {n} splits")

        # remove too short audio
        latest_kept_i = 0
        time_limit = 1
        for iter_ttk, (start, end) in enumerate(times_to_keep):
            metadata[iter_ttk]["duration"] = end - start
            if end - start < time_limit:
                # sometimes for very short segments, start and end are so
                # close that they overlap
                assert times_to_keep[latest_kept_i][1] - end <= 0.1, "overlapping audio"
                times_to_keep[latest_kept_i][1] = end
                metadata[latest_kept_i]["end"] = end

                times_to_keep[iter_ttk] = None
                metadata[iter_ttk]["status"] += " Too short"
            else:
                assert end - start >= 0, "End before start"
                latest_kept_i = iter_ttk
                while "  " in metadata[iter_ttk]["text"]:
                    metadata[iter_ttk]["text"] = metadata[iter_ttk]["text"].replace("  ", " ").strip()
        nbefore = len(times_to_keep)
        nafter = len([t for t in times_to_keep if t is not None])
        if not second_pass:
            whi(f"Kept {nafter}/{nbefore} splits when removing those <{time_limit}s")

        # remove almost no words if large model was used
        if second_pass:
            latest_kept_i = 0
            word_limit = 3
            for iter_meta, met in enumerate(metadata):
                if times_to_keep[iter_meta] is None:
                    continue
                start, end = times_to_keep[iter_meta]
                metadata[iter_meta]["n_words"] = len(met["text"].split(" "))
                if metadata[iter_meta]["n_words"] <= word_limit:

                    times_to_keep[latest_kept_i][1] = end
                    metadata[latest_kept_i]["end"] = end

                    times_to_keep[iter_meta] = None
                    metadata[iter_meta]["status"] += "Low nwords"
                else:
                    latest_kept_i = iter_meta
            nbefore = len(times_to_keep)
            nafter = len([t for t in times_to_keep if t is not None])
            whi(f"    Removed {nafter-nbefore}/{nbefore} splits with less than {word_limit} words")

        assert len(times_to_keep) == len(metadata), "invalid lengths"

        return times_to_keep, metadata

    def run_whisper(self, audio_path, second_pass):
        audio_path = str(audio_path)
        if not second_pass:
            whi(f"Running whisper on {audio_path}")

            audio = AudioSegment.from_mp3(audio_path)
            min_limit = 10
            minutes = len(audio) / 1000 / 60
            if minutes > min_limit:
                red(f"Audio is longer than {min_limit} minutes ({minutes:.1f}min) so will split then merge.")
                return self.run_whisper_long(audio_path, audio, min_limit)


        # hash used for the caching so that it does not depend on the path
        with open(audio_path, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).hexdigest()

        if second_pass:
            n_retry = 1
        else:
            n_retry = 3
        failed = True
        trial_dict = [
                {
                    "model": "large-v2",
                    "repo": "hnesk",
                    "batch_size": None,
                    },
                # {
                #     "model": "large-v1",
                #     "repo": "hnesk",
                #     "batch_size": None,
                #     },
                # {
                #     "model": "medium",
                #     "repo": "hnesk",
                #     "batch_size": None,
                #     },
                # {
                #     "model": "large-v3",
                #     "repo": "fast",
                #     "batch_size": 1,
                #     },
                ]
        for iparam, params in enumerate(trial_dict):
            for iter_retry in range(n_retry):
                try:
                    transcript = whisper_splitter(
                            audio_path=audio_path,
                            audio_hash=audio_hash,
                            prompt=self.prompt,
                            language=self.language,
                            **params,
                            )
                    failed = False
                    if iter_retry > 0 or iparam > 0:
                        red(f"Successfuly translated using parameters: {json.dumps(params)}")
                    break
                except Exception as err:
                    red(f"[{iparam+1}/{len(trial_dict)}] #{iter_retry + 1}/{n_retry}: Error when calling whisper_splitter with parameters: {json.dumps(params)}\nError: '{err}'")
            if not failed:
                break

        assert not failed, "Failed to get transcript."

        return transcript

    def run_whisper_long(self, audio_path, audio, min_limit):
        """for audio longer than some threshold (say 10 minutes) then in the
        first pass it makes sense to split the audio, use multithreading to
        transcribe it, then merge the transcripts (taking care of the
        transition and offsets)
        """
        ms_limit = min_limit * 60 * 1000

        # split the audio
        splits = [
                audio[i * ms_limit:(i+1) * ms_limit]
                for i in tqdm(range(len(audio) // ms_limit + 2), desc="splitting")
                if len(audio) >= i * ms_limit
                ]
        min_last = len(splits[-1]) / 1000 / 60
        if min_last < 1:
            # less than one minute, merge it with latest
            red(f"Last audio split lasts {min_last:.2f}min so merging with previous")
            splits[-2] += splits[-1]
            splits = splits[:-1]

        # creating temporary files
        tempfiles = [
                tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix=Path(audio_path).stem + f"__{i+1}_").name
                for i in range(len(splits))]

        # multithreaded export as mp3
        def threaded_export(audio, path):
            audio.export(path, format="mp3")
        joblib.Parallel(
                n_jobs=-1,
                backend="threading",
                )(
                        joblib.delayed(threaded_export)(split, temp)
                        for split, temp in zip(
                            splits,
                            tqdm(
                                tempfiles,
                                unit=f"{min_limit}min file",
                                desc="Exporting splits"
                                )
                            )
                        )
        assert all(Path(t).exists() for t in tempfiles), "missing temp files"

        # run whisper on each split
        def threaded_whisper(path):
            return self.run_whisper(audio_path=path, second_pass=False)
        tscripts = joblib.Parallel(
                n_jobs=-1,
                backend="threading",
                )(
                        joblib.delayed(threaded_whisper)(t)
                        for t in tqdm(
                            tempfiles,
                            unit=f"{min_limit}min file",
                            desc="Treating long audio as splits"
                            )
                        )

        # merge the transcripts
        transcript = tscripts[0]
        offset = 0
        for it, t in enumerate(tscripts[1:]):
            transcript["transcription"] += " " + t["transcription"]
            offset = (it + 1) * min_limit * 60

            # offset the timestamps
            segs = t["segments"]
            for iseg, seg in enumerate(segs):
                segs[iseg]["start"] += offset
                segs[iseg]["end"] += offset
                for iw, w in enumerate(segs[iseg]["words"]):
                    segs[iseg]["words"][iw]["start"] += offset
                    segs[iseg]["words"][iw]["end"] += offset

            # make sure to merge the transition
            transcript["segments"][-1]["end"] = segs[0]["end"]
            transcript["segments"][-1]["text"] += " " + segs[0]["text"]
            transcript["segments"][-1]["words"].extend(segs[0]["words"])

            # edit metadata
            if transcript["repo"] != "fast":
                transcript["segments"][-1]["no_speech_prob"] = min(segs[0]["no_speech_prob"], transcript["segments"][-1]["no_speech_prob"])
                transcript["segments"][-1]["avg_logprob"] = min(segs[0]["avg_logprob"], transcript["segments"][-1]["avg_logprob"])
                transcript["segments"][-1]["compression_ratio"] = min(segs[0]["compression_ratio"], transcript["segments"][-1]["compression_ratio"])
                transcript["segments"][-1]["temperature"] = min(segs[0]["temperature"], transcript["segments"][-1]["temperature"])

            segs.pop(0)
            transcript["segments"].extend(segs)

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
        new_filename = file.parent / (file.stem + "_unsilenced" + file.suffix)
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

        elif self.silence_method == "sox_cli":
            # sox way, fast but needs linux
            f1 = "\"" + str(file.name) + "\""
            f2 = "\"" + str(new_filename.name) + "\""
            d = "\"" + str(file.parent.absolute()) + "\""

            sox_oneliner = " ".join([" ".join(effect).strip() for effect in shared.splitter_sox_effects]).strip()
            sox_cmd = f"cd {d} && rm tmpoutput*.mp3 ; sox {f1} tmpoutput.mp3 {sox_oneliner} : newfile : restart && cat tmpoutput*.mp3 > {f2} && rm -v tmpout*.mp3"
            self.exec(sox_cmd)
            assert new_filename.exists(), f"new file not found: '{new_filename}'"
            new_audio = AudioSegment.from_mp3(new_filename)

        elif self.silence_method == "torchaudio":
            # load from file
            shutil.copy2(file, new_filename)

            # reuse the same code form media
            waveform, sample_rate = torchaudio.load(new_filename)

            waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                    waveform,
                    sample_rate,
                    shared.splitter_sox_effects,
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
        whi("Copying")
        shutil.copy2(file, self.done_dir / (file.name + "temp"))
        whi("Renaming")
        shutil.move(
                self.done_dir / (file.name + "temp"),
                self.done_dir / file.name)
        whi("Removing")
        file.unlink(missing_ok=False)
        whi("Done!")

        return new_filename

    def exec(self, cmd):
        whi(f"Shell command: {cmd}")
        os.system(cmd)


@stt_cache.cache(ignore=["audio_path", "batch_size"])
def whisper_splitter(audio_path, audio_hash, prompt, language, repo, model, batch_size):
    whi(f"Starting replicate (meaning cache is not used). Model={model} Repo={repo} Batch_size={batch_size}")
    if not audio_path.startswith("http"):
        audio_path = open(audio_path, "rb")
    start = time.time()
    if repo == "fast":
        raise NotImplementedError("Fast repo is disabled because it seems to produce overlapping segments.")
        # https://replicate.com/vaibhavs10/incredibly-fast-whisper/
        # https://github.com/chenxwh/insanely-fast-whisper
        transcript = replicate.run(
                "vaibhavs10/incredibly-fast-whisper:c6433aab18b7318bbae316495f1a097fc067deef7d59dc2f33e45077ae5956c7",
                input={
                    "audio": audio_path,
                    "task": "transcribe",
                    "model": model,
                    "language": language,
                    "timestamp": "word",
                    "diarise_audio": False,
                    "batch_size": batch_size,
                    },
                )

        # fix the format to have the same type as hnesk and collectiveai
        transcript["segments"] = transcript.pop("chunks")
        transcript["transcription"] = transcript.pop("text")
        # del transcript["chunks"], transcript["text"]
        for iter_chunk, chunk in enumerate(transcript["segments"]):
            transcript["segments"][iter_chunk]["start"] = chunk["timestamp"][0]
            transcript["segments"][iter_chunk]["end"] = chunk["timestamp"][1]
            # del transcript["segments"][iter_chunk]["timestamp"]

            transcript["segments"][iter_chunk]["words"] = [
                    {
                        "word": chunk["text"],
                        "start": chunk["timestamp"][0],
                        "end": chunk["timestamp"][1],
                        }
                    ]

    elif repo == "collectiveai":
        # https://replicate.com/collectiveai-team/whisper-wordtimestamps/
        # https://github.com/collectiveai-team/whisper-wordtimestamps/
        # fork from hnesk's repo. Allows larger file to be sent apparently.
        transcript = replicate.run(
                "collectiveai-team/whisper-wordtimestamps:781317565f264090bf5831cceb3ea6b794ed402e746fde1cdec103a8951b52df",
                input={
                    "audio": audio_path,
                    "model": model,
                    "language": language,
                    "temperature": 0,
                    "initial_prompt": prompt,
                    "condition_on_previous_text": False,
                    "word_timestamps": True,
                    "no_speech_threshold": 1,
                    },
                )
    elif repo == "hnesk":
        # https://replicate.com/hnesk/whisper-wordtimestamps/
        # https://github.com/hnesk/whisper-wordtimestamps
        transcript = replicate.run(
                "hnesk/whisper-wordtimestamps:4a60104c44dd709fc08a03dfeca6c6906257633dd03fd58663ec896a4eeba30e",
                input={
                    "audio": audio_path,
                    "model": model,
                    "language": language,
                    "temperature": 0,
                    "initial_prompt": prompt,
                    "condition_on_previous_text": False,
                    "word_timestamps": True,
                    "no_speech_threshold": 1,
                    },
                )
    else:
        raise ValueError(repo)

    transcript["modelname"] = model
    transcript["repo"] = repo

    whi(f"Finished with replicate in {int(time.time()-start)} second")
    return transcript

if __name__ == "__main__":
    fire.Fire(AudioSplitter)
