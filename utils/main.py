import numpy as np
from typing import Union, List, Callable, Optional
import os
import asyncio
import pickle
from pprint import pprint
import re
from rapidfuzz.fuzz import ratio as levratio
import shutil
import queue
import threading
import hashlib
import base64
import json
from textwrap import dedent
from functools import wraps
import time
from datetime import datetime
from pathlib import Path, PosixPath

import cv2
import gradio as gr
import joblib
from pydub import AudioSegment

import litellm
import openai
from deepgram import DeepgramClient, PrerecordedOptions

from .anki_utils import add_note_to_anki, add_audio_to_anki
from .shared_module import shared
from .logger import red, whi, yel, store_to_db, trace, Timeout, smartcache, Critical
from .memory import prompt_filter, load_prev_prompts, tkn_len, transcript_template, default_system_prompt, split_thinking
from .media import sound_preprocessing, get_img_source, format_audio_component, rgb_to_bgr
from .profiles import ValueStorage
from .typechecker import optional_typecheck

litellm.set_verbose = False  #shared.debug
shared.pv = ValueStorage()
shared.message_buffer = shared.pv["message_buffer"]

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

stt_cache = joblib.Memory("cache/transcript_cache", verbose=0)
llm_cache = joblib.Memory("cache/llm_cache", verbose=0)

# store in a dict to avoid recreating an instance each time
deepgram_clients = {}

@trace
@optional_typecheck
def clear_cache() -> None:
    # reset the llm and stt cache to make sure shared.llm_to_db_buffer is up to date
    llm_cache.clear()
    stt_cache.clear()


@trace
@optional_typecheck
def pop_buffer() -> None:
    "remove the latest card from message buffer"
    removed = shared.message_buffer.pop(-1)
    shared.pv["message_buffer"] = shared.message_buffer
    red(f"Message buffer size is now {len(shared.message_buffer)} after removing '{removed}'")

@optional_typecheck
def floatizer(func: Callable) -> Callable:
    "used to cast the ints as float to make sure the cache is used"
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [float(ar) if (isinstance(ar, int) and not isinstance(ar, bool)) else ar for ar in args]
        kwargs = {k: float(v) if (isinstance(v, int) and not isinstance(v, bool)) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

@optional_typecheck
def stripizer(func: Callable) -> Callable:
    """wrapper for alfred to make sure to strip the txt_audio"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "txt_audio" in kwargs:
            kwargs["txt_audio"] = kwargs["txt_audio"].strip()
        else:
            assert isinstance(args[0], str), f"expected string but found {args[0]}"
            args = list(args)
            args[0] = args[0].strip()
        return func(*args, **kwargs)
    return wrapper

@optional_typecheck
def split_txt_audio(txt_audio: str) -> str:
    """if the txt audio contains "STOP" then it must be replaced by \n\n so
    that alfred treats them as separate notes"""
    txt_audio = re.sub(r"(\W|^)s?top ?(\W|$)", "\n\n", txt_audio, flags=re.IGNORECASE)

    # remove leading space etc
    sp = txt_audio.split("\n\n")
    txt_audio = "\n\n".join([s.strip() for s in sp])
    txt_audio = txt_audio.strip()

    if not txt_audio:
        txt_audio = "Empty"

    return txt_audio


@floatizer
@optional_typecheck
@trace
@smartcache
@stt_cache.cache(ignore=["audio_path"])
def whisper_cached(
        audio_path: Union[str, PosixPath],
        audio_hash: str,
        txt_whisp_prompt: Optional[str],
        txt_whisp_lang: Optional[str],
        sld_whisp_temp: Union[float, int],
        stt_model: str,
        ) -> dict:
    """this is a call to whisper. It's called as soon as the
    recording is done to begin caching. The audio_path can change so a hash
    of the content is used instead."""
    red(f"Calling whisper because not in cache: {audio_path}")

    assert shared.pv["txt_openai_api_key"] or shared.pv["txt_deepgram_api_key"], "Missing OpenAI or deepgram API key, needed for Whisper"
    if "openai" in stt_model:
        assert shared.pv["txt_openai_api_key"], "Missing OpenAI API key, needed for Whisper"
        if shared.openai_client is None:
            shared.openai_client = openai.OpenAI(api_key=shared.pv["txt_openai_api_key"].strip())
    elif "deepgram" in stt_model:
        assert shared.pv["txt_deepgram_api_key"], "Missing Deepgram API key, needed for Whisper"
        os.environ["DEEPGRAM_API_TOKEN"] = shared.pv["txt_deepgram_api_key"].strip()

    if txt_whisp_prompt.strip() == "":
        txt_whisp_prompt = None
    if txt_whisp_lang.strip() == "":
        txt_whisp_lang = None

    try:
        len_audio = len(AudioSegment.from_mp3(audio_path)) // 1000
        if len_audio <= 1:
            red(f"Very short audio under 1s sent to whisper! (length={len_audio:.2f}s)")
    except Exception as err:
        red(f"Error when trying to see if audio is too short but will continue nonetheless with whisper.\nError: {err}")
    try:
        cnt = 0
        while True:
            try:
                cnt += 1

                with open(audio_path, "rb") as audio_file:
                    if stt_model == "openai:whisper-1":
                        transcript = shared.openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            prompt=txt_whisp_prompt,
                            language=txt_whisp_lang,
                            temperature=sld_whisp_temp,
                            response_format="verbose_json",
                            )
                        transcript = json.loads(transcript.json())
                        if not transcript["segments"]:
                            gr.Warning(red(f"No audio segment found in {audio_path}"))
                        if sld_whisp_temp == 0:
                            temps = [seg["temperature"] for seg in transcript["segments"]]
                            if not temps:
                                gr.Warning(red(f"No audio temperature found in {audio_path}"))
                            elif sum(temps) / len(temps) == 1:
                                raise Exception(red(f"Whisper increased temperature to maximum, probably because no words could be heard."))


                    elif stt_model == "deepgram:nova-2":

                        if os.environ["DEEPGRAM_API_TOKEN"] not in deepgram_clients:
                            while deepgram_clients:
                                k = next(deepgram_clients.keys())
                                del deepgram_clients[k]
                            deepgram_clients[os.environ["DEEPGRAM_API_TOKEN"]] = DeepgramClient()
                        assert len(deepgram_clients) == 1, f"found multiple deepgram_client: {deepgram_clients}"
                        client = deepgram_clients[os.environ["DEEPGRAM_API_TOKEN"]]
                        if shared.pv["txt_deepgram_keyword_boosting"]:
                            keywords = shared.pv["txt_deepgram_keyword_boosting"].strip().splitlines()
                        else:
                            keywords = None
                        keywords="&keywords=".join(shared.pv["txt_deepgram_keyword_boosting"])
                        # set options
                        options = dict(
                            # docs: https://playground.deepgram.com/?endpoint=listen&smart_format=true&language=en&model=nova-2
                            model="nova-2",
                            language=txt_whisp_lang,

                            detect_language=False,
                            # not all features below are available for all languages

                            # transcription
                            smart_format=True,
                            punctuate=True,
                            paragraphs=False,
                            utterances=True,
                            diarize=False,
                            # dictation=False,
                            detect_entities=False,
                            detect_topics=False,
                            keywords=keywords,
                            numerals=True,
                        )
                        if txt_whisp_lang == "en":
                            options.update(dict(
                                # intelligence
                                summarize=False,
                                topics=False,
                                intents=False,
                                sentiment=False,

                                filler_words=True,
                                measurements=True,
                                )
                            )

                        options = PrerecordedOptions(**options)
                        payload = {"buffer": audio_file.read()}
                        transcript = client.listen.prerecorded.v("1").transcribe_file(
                            payload,
                            options,
                        ).to_dict()
                        assert isinstance(transcript, dict), f"transcript is not dict but {transcript}"
                        assert len(transcript["results"]["channels"]) == 1, "unexpected deepgram output"
                        assert len(transcript["results"]["channels"][0]["alternatives"]) == 1, "unexpected deepgram output"
                        text = transcript["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
                        assert text, "Empty text from deepgram transcription"
                        transcript["duration"] = transcript["metadata"]["duration"]
                        transcript["text"] = text
                    else:
                        raise ValueError(stt_model)

                # update the df
                if shared.pv["enable_dirload"]:
                    try:
                        if not shared.dirload_queue.empty:
                            tmp_df = shared.dirload_queue.reset_index().set_index("temp_path")
                            try:
                                orig_path = tmp_df.loc[str(audio_path), "path"]
                            except Exception:
                                orig_path = tmp_df.loc[str(audio_path).replace("_proc", ""), "path"]
                            assert orig_path in tmp_df.index, f"Not in tmp_df index: {orig_path}"
                            with shared.dirload_lock:
                                tmp_df.loc[orig_path, "transcribed"] = transcript["text"]
                    except Exception as err:
                        red(f"Couldn't update df to set transcript of {audio_path}: {err}")

                transcript["text"] = transcript["text"].replace("Sous-titres réalisés par la communauté d'Amara.org", "")
                transcript["text"] = transcript["text"].replace("Abonnez-vous à la chaine !", "")
                return transcript
            except openai.RateLimitError as err:
                if cnt >= 5:
                    raise Exception(red(f"Cached whisper: RateLimitError >5: '{err}'"))
                else:
                    gr.Warning(red(f"Cached whisper: RateLimitError #{cnt}/5 from cached whisper: '{err}'"))
                    time.sleep(2 * cnt)
    except Exception as err:
        raise Exception(red(f"Error when cache transcribing audio: '{err}'"))


@optional_typecheck
@trace
def thread_whisp_then_llm(audio_mp3: Optional[Union[PosixPath, str]]) -> None:
    """run whisper on the audio and return nothing. This is used to cache in
    advance and in parallel the transcription."""
    if audio_mp3 is None:
        return

    audio_mp3 = format_audio_component(audio_mp3)
    whi(f"Transcribing audio for the cache: {audio_mp3}")

    if not (shared.pv["txt_openai_api_key"] or shared.pv["txt_deepgram_api_key"] or shared.pv["txt_mistral_api_key"] or shared.pv["txt_openrouter_api_key"]):
        raise Exception(red("No API key provided for any LLM. Do it in the settings."))

    with open(audio_mp3, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    tmp_df = shared.dirload_queue.reset_index().set_index("temp_path")
    assert str(audio_mp3) in tmp_df.index, f"Missing {audio_mp3} from shared.dirload_queue"
    assert tmp_df.index.tolist().count(str(audio_mp3)) == 1, f"Duplicate temp_path in shared.dirload_queue: {audio_mp3}"

    orig_path = tmp_df.loc[str(audio_mp3), "path"]
    with shared.dirload_lock:
        shared.dirload_queue.loc[orig_path, "transcribed"] = "started"

    transcript = whisper_cached(
            audio_mp3,
            audio_hash,
            txt_whisp_prompt=shared.pv["txt_whisp_prompt"],
            txt_whisp_lang=shared.pv["txt_whisp_lang"],
            sld_whisp_temp=shared.pv["sld_whisp_temp"],
            stt_model=shared.pv["stt_choice"],
            )
    txt_audio = transcript["text"]

    if transcript["duration"] <= 1:
        txt_audio = f"Very short audio, so unreliable transcript: {txt_audio}"

    # if contains stop, split it
    txt_audio = split_txt_audio(txt_audio)

    with shared.dirload_lock:
        shared.dirload_queue.loc[orig_path, "transcribed"] = txt_audio
        shared.dirload_queue.loc[orig_path, "alfreded"] = "started"

    try:
        cloze = alfred(
                txt_audio=txt_audio,
                txt_chatgpt_context=shared.pv["txt_chatgpt_context"],
                profile=shared.pv.profile_name,
                max_token=int(shared.pv["sld_max_tkn"]),
                temperature=shared.pv["sld_temp"],
                sld_buffer=shared.pv["sld_buffer"],
                llm_choice=shared.pv["llm_choice"],
                txt_keywords=shared.pv["txt_keywords"],
                prompt_management=shared.pv["prompt_management"],
                cache_mode=True)
        with shared.dirload_lock:
            shared.dirload_queue.loc[orig_path, "alfreded"] = cloze
    except Exception as err:
        with shared.dirload_lock:
            shared.dirload_queue.loc[orig_path, "alfreded"] = f"Failed: {err}"
        gr.Warning(red(f"Failed anticipating Alfred call: {err}"))


@optional_typecheck
@trace
def transcribe(audio_mp3_1: Optional[Union[List[Union[str, dict]], str, dict]]) -> str:
    "turn the 1st audio track into text"
    whi("Transcribing audio")

    if isinstance(audio_mp3_1, list):
        assert len(audio_mp3_1) == 1, f"Got an unexpected list as audio_mp3_1: {audio_mp3_1}"
        audio_mp3_1 = audio_mp3_1[0]

    if audio_mp3_1 is None:
        raise Exception(red("Error: None audio_mp3_1"))

    if shared.pv["txt_whisp_lang"] is None:
        red("Warning: None whisper language")

    audio_mp3_1 = format_audio_component(audio_mp3_1)

    with open(audio_mp3_1, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        whi(f"Asking Cached Whisper for {audio_mp3_1}")
        transcript = whisper_cached(
                audio_mp3_1,
                audio_hash,
                txt_whisp_prompt=shared.pv["txt_whisp_prompt"],
                txt_whisp_lang=shared.pv["txt_whisp_lang"],
                sld_whisp_temp=shared.pv["sld_whisp_temp"],
                stt_model=shared.pv["stt_choice"],
                )
        with open(audio_mp3_1, "rb") as audio_file:
            mp3_content = audio_file.read()
        txt_audio = transcript["text"]
        if transcript["duration"] <= 1:
            txt_audio = f"Very short audio, so unreliable transcript: {txt_audio}"
        yel(f"\nWhisper transcript: {txt_audio}")

        with shared.thread_lock:
            if shared.running_threads["saving_whisper"]:
                shared.running_threads["saving_whisper"][-1].join()
            while shared.running_threads["saving_whisper"]:
                shared.running_threads["saving_whisper"].pop()
        thread = threading.Thread(
                target=store_to_db,
                name="saving_whisper",
                kwargs={
                    "dictionnary": {
                        "type": "whisper_transcription",
                        "timestamp": time.time(),
                        "whisper_language": shared.pv["txt_whisp_lang"],
                        "whisper_context": shared.pv["txt_whisp_prompt"],
                        "whisper_temperature": shared.pv["sld_whisp_temp"],
                        "Voice2Anki_profile": shared.pv.profile_name,
                        "transcribed_input": txt_audio,
                        "full_whisper_output": transcript,
                        "model_name": shared.pv["stt_choice"],
                        "audio_mp3": base64.b64encode(mp3_content).decode(),
                        "Voice2Anki_version": shared.VERSION,
                        "request_information": shared.request,
                        },
                    "db_name": "whisper"
                    })
        thread.start()
        with shared.thread_lock:
            shared.running_threads["saving_whisper"].append(thread)

        # if contains stop, split it
        txt_audio = split_txt_audio(txt_audio)

        return txt_audio
    except Exception as err:
        raise Exception(red(f"Error when transcribing audio: '{err}'"))


@optional_typecheck
@trace
def flag_audio(
        txt_profile: str,
        txt_audio: str,
        txt_whisp_lang: str,
        txt_whisp_prompt: str,
        txt_chatgpt_cloz: str,
        txt_chatgpt_context: str,
        gallery: Union[None, List, gr.Gallery],
        ) -> None:
    """copy audio in slot #1 to the "flagged" folder in the profile folder"""
    assert shared.pv["enable_flagging"], "Incoherent UI"
    # move audio file
    if not (shared.dirload_queue["loaded"] == True).any():
        raise Exception(red("No loaded files in shared.dirload_queue"))

    # create the appropriate dir
    if shared.splitted_dir is None:
        flag_dir = "./flagged"
    else:
        flag_dir = shared.splitted_dir.parent / "flagged"
        assert flag_dir.parent.exists(), f"Couldn't find flag_dir parent at {flag_dir.parent}"
    if flag_dir.exists():
        assert flag_dir.is_dir(), f"flag_dir is not a directory: {flag_dir}"
    else:
        flag_dir.mkdir(parents=False, exist_ok=False)

    aud = Path(shared.dirload_queue[shared.dirload_queue["loaded"] == True].iloc[0].name)
    assert aud.exists(), f"Original audio file not found: {aud}"
    new_filename = flag_dir / aud.name
    if Path(new_filename).exists():
        raise Exception(red(f"Audio you're trying to flag already exists: {new_filename}"))
    shutil.copy2(aud, new_filename)
    red(f"Flagged: copied '{aud}' to '{new_filename}'")

    # make sure the gallery is saved as image and not as path
    if gallery is not None:
        try:
            new_val = []
            if hasattr(gallery, "root"):
                gallery = gallery.root
            if isinstance(gallery, list):
                new_gal = []
                for im in gallery:
                    if isinstance(im, tuple):
                        assert Path(im[0]).exists(), f"Missing image from tuple {im}"
                        assert im[1] is None, f"Unexpected tupe: {im}"
                        new_gal.append(
                                rgb_to_bgr(
                                    cv2.imread(
                                        im[0],
                                        flags=1)
                                    )
                                )
                    else:
                        new_gal.append(
                                rgb_to_bgr(
                                    cv2.imread(
                                        im.image.path,
                                        flags=1)
                                    )
                                )
            gallery = new_gal
            pickle.dumps(gallery)
        except Exception as err:
            gr.Warning(red(f"Failed to get the gallery to flag: '{err}'"))

    # move the other component's data
    to_save = {
            "txt_profile": txt_profile,
            "txt_audio": txt_audio,
            "txt_whisp_lang": txt_whisp_lang,
            "txt_whisp_prompt": txt_whisp_prompt,
            "txt_chatgpt_cloz": txt_chatgpt_cloz,
            "txt_chatgpt_context": txt_chatgpt_context,
            "gallery": gallery,
            "request_information": shared.request,
            }
    with (new_filename.parent / (new_filename.name + ".pickle")).open("wb") as f:
        pickle.dump(to_save, f)
    gr.Warning(red(f"Done flagging audio and metadata at {new_filename}"))


@stripizer
@optional_typecheck
@trace
def pre_alfred(
        txt_audio: str,
        txt_chatgpt_context: str,
        profile: str,
        max_token: Union[int, float],
        temperature: Union[float, int],
        sld_buffer: Union[int, float],
        txt_keywords: str,
        prompt_management: str,
        cache_mode: bool,
        ) -> List[dict]:
    """used to prepare the prompts for alfred call. This is a distinct
    function to make it callable by the cached function too."""
    # don't print when using cache
    assert isinstance(cache_mode, bool), f"Invalid cache_mode: {cache_mode}"
    if cache_mode:
        whi = lambda x: None
        yel = lambda x: None
    else:
        from .logger import whi, yel

    if "," in txt_keywords:
        keywords = [re.compile(kw.strip(), flags=re.DOTALL|re.MULTILINE|re.IGNORECASE) for kw in txt_keywords.split(",")]
        keywords = [kw for kw in keywords if re.search(kw, txt_audio)]
    else:
        keywords = []

    # load prompts from memory.json
    prev_prompts = load_prev_prompts(profile).copy()

    # format the new prompt
    new_prompt = {
            "role": "user",
            "content": dedent(
                transcript_template.replace("CONTEXT", txt_chatgpt_context
                    ).replace("TRANSCRIPT", txt_audio))
                }
    # the last few transcript/answer pair is always saved in message_buffer
    # even if it will not be saved to memory.
    buffer_to_add = []
    if sld_buffer and shared.message_buffer:
        whi(f"Length of message_buffer: {len(shared.message_buffer)}")

        for mb in shared.message_buffer[::-1]:
            if len(buffer_to_add) / 2 >= sld_buffer:
                break
            if levratio(txt_audio, mb["unformatted_txt_audio"]) >= 95:
                # skip this one
                red(f"Skipped buffer: {mb}")
                continue
            if levratio(txt_audio, mb["question"]) >= 95:
                # skip this one
                red(f"Skipped buffer: {mb}")
                continue
            buffer_to_add.extend(
                    [
                        {
                            "role": "user",
                            "content": mb["question"],
                            },
                        {
                            "role": "assistant",
                            "content": mb["answer"],
                            }
                        ]
                    )
            whi(f"Added message_buffer to the prompt: {mb}")
    else:
        whi("Ignored message buffer")

    prev_prompts = prompt_filter(
            prev_prompts,
            max_token,
            prompt_messages=buffer_to_add + [new_prompt],
            keywords=keywords,
            )

    # add all prompts together to format the messages list
    formatted_messages = []
    # add system prompt
    formatted_messages.insert(
            0,
            {"role": "system", "content": default_system_prompt["content"]}
            )

    if prompt_management == "1 per mess":
        # first way: add the examples one after the other
        # add the selected prompts
        for m in prev_prompts:
            formatted_messages.append(
                    {
                        "role": m["role"],
                        "content": m["content"],
                        }
                    )
            formatted_messages.append({
                "role": "assistant",
                "content": m["answer"]})
        # add message buffer
        formatted_messages.extend(buffer_to_add)
        formatted_messages.append(new_prompt)

    elif prompt_management == "Stuff as XML in sys":
        # second way: add the messages as example in the system prompt
        inputs = """

Here are examples of input (me) and appropriate outputs (you):
<examples>
"""
        for m in prev_prompts:
            inputs += f"<ex>\n<input>{m['content']}</input>\n<output>{m['answer']}</output>\n</ex>"
        for m in buffer_to_add:
            if m["role"] == "user":
                inputs += f"<ex>\n<input>{m['content']}</input>"
            elif m["role"] == "assistant":
                inputs += f"\n<output>{m['content']}</output>\n</ex>"
            else:
                raise ValueError(m["role"])
        assert inputs.endswith("</output>\n</ex>"), f"Unexpected end of inputs:\n{inputs}"
        inputs += "\n</examples>"
        assert len(formatted_messages) == 1
        formatted_messages[-1]["content"] += inputs
        formatted_messages.append(new_prompt)

    elif prompt_management == "Stuff as XML in user":
        inputs = """

Here are examples of input (me) and appropriate outputs (you):
<examples>
"""
        for m in prev_prompts:
            inputs += f"<ex>\n<input>{m['content']}</input>\n<output>{m['answer']}</output>\n</ex>"
        for m in buffer_to_add:
            if m["role"] == "user":
                inputs += f"<ex>\n<input>{m['content']}</input>"
            elif m["role"] == "assistant":
                inputs += f"\n<output>{m['content']}</output>\n</ex>"
            else:
                raise ValueError(m["role"])
        assert inputs.endswith("</output>\n</ex>"), f"Unexpected end of inputs:\n{inputs}"
        inputs += "\n</examples>"
        assert len(formatted_messages) == 1
        formatted_messages.append(
            {"role": "user", "content": inputs}
        )
        formatted_messages[-1]["content"] += f"\n\n<user_input>\n{new_prompt['content']}\n</user_input>\n\nNow answer the user_input."

    else:
        raise ValueError(f"Invalid prompt managment: {prompt_management}")

    # check the number of token is fine and format the previous
    # prompts in chatgpt format
    tkns = 0
    for mess in formatted_messages:
        tkns += tkn_len(mess["content"])

    yel(f"Number of messages that will be sent to ChatGPT: {len(formatted_messages)} (representing {tkns} tokens)")

    if shared.pv['llm_choice'] in shared.llm_info:
        modelinfo = litellm.model_cost[shared.pv['llm_choice']]
    elif shared.pv['llm_choice'].split("/", 1)[1] in shared.llm_info:
        modelinfo = litellm.model_cost[shared.pv['llm_choice'].split("/", 1)[1]]
    else:
        raise ValueError(f"Couldn't find model info about '{shared.pv['llm_choice']}' in litellm")
    if "max_input_tokens" in modelinfo:
        input_token_limit = modelinfo["max_input_tokens"]
    elif "max_tokens" in modelinfo:
        input_token_limit = modelinfo["max_tokens"]
    else:
        raise ValueError(f"Couldn't find neither max_input_tokens nor max_tokens in modelinfo:\n{modelinfo}")

    assert tkns <= input_token_limit, f"Too many input tokens for model: {tkns} > {input_token_limit}"
    if tkns >= input_token_limit:
        red(f"More than {input_token_limit} tokens before calling LLM. Bypassing to ask "
            "with fewer tokens to make sure you have room for the answer")
        return pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature, sld_buffer, txt_keywords, prompt_management, cache_mode)

    # # print prompts used for the call:
    # n = len(formatted_messages)
    # whi("LLM prompt:")
    # for i, p in enumerate(formatted_messages):
    #     whi(f"* {i+1}/{n}: {p['role']}")
    #     whi(indent(p['content'], " " * 5))

    return formatted_messages


async def async_alfred(*args, **kwargs):
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, alfred, *args, **kwargs)
    except Exception as err:
        return err


async def async_parallel_alfred(splits, *args, **kwargs):
    tasks = [async_alfred(sp, *args, **kwargs) for sp in splits]
    return await asyncio.gather(*tasks)


@stripizer
@floatizer
@optional_typecheck
@trace
@smartcache
@Timeout(60)
@llm_cache.cache(ignore=["cache_mode"])
def alfred(
        txt_audio: str,
        txt_chatgpt_context: Optional[str],
        profile: str,
        max_token: Union[int, float],
        temperature: Union[int, float],
        sld_buffer: Union[int, float],
        llm_choice: str,
        txt_keywords: Optional[str],
        prompt_management: str,
        cache_mode: bool = False,
        ) -> str:
    "send the previous prompt and transcribed speech to the LLM"
    red(f"Calling Alfred in cache_mode={cache_mode} for transcript '{txt_audio}'")
    assert max_token > 0, f"Invalid value for max_token: {max_token}"
    if not txt_audio:
        gr.Warning(red("Empty transcription."))
        return "Empty transcription"
    if txt_audio.strip().startswith("Error"):
        raise Exception(red("Error when transcribing sound."))
    if not txt_chatgpt_context:
        raise Exception(red("No txt_chatgpt_context found."))
    assert isinstance(prompt_management, str), f"Invalid type of prompt_management: {prompt_management}"
    assert isinstance(cache_mode, bool), f"Invalid type of cache_mode: {cache_mode}"
    if txt_audio.startswith("Very short audio, so unreliable transcript: "):
        if cache_mode:
            return red(txt_audio)
        else:
            # gr.Warning(mess)
            return red(txt_audio)
    if (("fred" in txt_audio.lower() and "image" in txt_audio.lower()) or ("change d'image" in txt_audio.lower())) and len(txt_audio) < 40:
        mess = f"Image change detected: '{txt_audio}'"
        if cache_mode:
            return red(mess)
        else:
            # gr.Warning(mess)
            return red(mess)
    if "cour" in txt_audio.lower() and len(txt_audio) < 40 and re.search(r"d[eé]but", txt_audio.lower()):
        mess = f"Lesson switch detected: '{txt_audio}'"
        if cache_mode:
            return red(mess)
        else:
            # gr.Warning(mess)
            return red(mess)
    if txt_audio.count(" ") < 3:
        return red("Too few words in txt_audio to be plausible")

    if not (shared.pv["txt_openai_api_key"] or shared.pv["txt_mistral_api_key"] or shared.pv["txt_openrouter_api_key"]):
        raise Exception(red("No API key provided for any LLM. Do it in the settings."))

    # automatically split repeated newlines as several distinct cards
    txt_audio = txt_audio.strip()
    while "\n\n\n" in txt_audio:
        txt_audio = txt_audio.replace("\n\n\n", "\n\n")
    if "\n\n" in txt_audio:
        red(f"Detected txt_audio that needed to be split: '{txt_audio}'")
        txt_audio = txt_audio.replace("\n\n", "\n#####\n")

    # if contains #####, split into subclozes
    if "#####" in txt_audio:
        red(f"Splitting txt_audio for Alfred: '{txt_audio}'")

        splits = txt_audio.split("#####")
        splits = [sp.strip() for sp in splits if sp.strip()]
        if len(splits) == 1:
            gr.Warning(red(f"Found only 1 split in '{txt_audio}' which is '{splits[0]}'"))
            txt_audio = splits[0]
        else:
            answers = asyncio.run(async_parallel_alfred(splits, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, llm_choice, txt_keywords, prompt_management, cache_mode))
            assert len(answers) == len(splits), "Unexpected length"
            assert isinstance(answers, list), f"unexpected answers: {answers}"
            return "\n#####\n".join(answers).strip()

    formatted_messages = pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, txt_keywords, prompt_management, cache_mode)
    if not formatted_messages and max_token < 500:
        raise Exception(red(f"max_token is low at {max_token} which might be why pre_alfred output no memories."))
    for i, fm in enumerate(formatted_messages):
        if i == 0:
            assert fm["role"] == "system", f"expected system role but got {fm['role']}"
        elif i % 2 == 0:
            assert fm["role"] == "assistant", f"expected assistant role but got {fm['role']}"
        elif i % 2 == 1:
            assert fm["role"] == "user", f"expected user role but got {fm['role']}"

        formatted_messages[i]["content"] = fm["content"].replace("<br/>", "\n").replace("<br>", "\n")

    # check no duplicate in messages
    if len(set([fm["content"] for fm in formatted_messages])) != len(formatted_messages):
        contents = [pm["content"] for pm in formatted_messages]
        dupli = [dp for dp in contents if contents.count(dp) > 1 and "alfred:" not in dp.lower()]
        if dupli:
            raise Exception(f"{len(dupli)} duplicate prompts found: {dupli}:" + "\n* " + "\n* ".join(list(set(dupli))))

    model_price = shared.llm_info[llm_choice]
    whi(f"Will use model {llm_choice}")

    # pprint("Prompt for LLM:")
    # pprint(formatted_messages)

    try:
        response = litellm.completion(
                model=llm_choice,
                messages=formatted_messages,
                temperature=temperature,
                num_retries=2
                )
    except Exception as e:
        raise Exception(f"Error when calling {llm_choice} with temperature {temperature}: ") from e

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]
    tkn_cost = [input_tkn_cost, output_tkn_cost]

    tkn_cost_dol = input_tkn_cost * model_price["input_cost_per_token"] + output_tkn_cost * model_price["output_cost_per_token"]
    shared.pv["total_llm_cost"] += tkn_cost_dol

    cloz = response["choices"][0]["message"]["content"]
    # for cosmetic purposes in the textbox
    cloz = cloz.replace(" •", "     -")
    cloz = cloz.replace("<br>", "\n")
    cloz = cloz.replace("<br/>", "\n")
    cloz = cloz.replace("\n- {{c1::", "\n{{c1::- ")
    cloz = cloz.replace("&nbsp;", " ")
    cloz = cloz.replace("#####", "\n#####\n")  # make sure to separate cleanly the clozes

    # make sure the thinking are at the end
    cloz = "\n".join([c.strip() for c in cloz.splitlines() if c.strip()])
    cont, th = split_thinking(cloz)
    cloz = f"{cont}\n{th}"

    # # if contains cloze in multiple parts but in the same line, merge them
    # sl = cloz.splitlines()
    # if len(sl) == 2:
    #     if "{{c" not in sl[0] and "}}" not in sl[0]:
    #         if sl[1].count("{{c1::") > 1:
    #             sl[1] = re.sub("}}(.*?){{c1::", r"\1", sl[1])
    #         cloz = "\n".join(sl)

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        gr.Warning(red(f"LLM's reason to stop was not 'stop' but '{reason}'"))

    # add to the shared module the infonrmation of this card creation.
    # if a card is created then this will be added to the db to
    # create LORA fine tunes later on.
    with shared.db_lock:
        shared.llm_to_db_buffer[cloz] = json.dumps(
                {
                    "type": "anki_card",
                    "timestamp": time.time(),
                    "token_cost": tkn_cost,
                    "dollar_cost": tkn_cost_dol,
                    "temperature": temperature,
                    "LLM_context": txt_chatgpt_context,
                    "Voice2Anki_profile": shared.pv.profile_name,
                    "transcribed_input": txt_audio,
                    "llm_output": response.json(),
                    "model_name": llm_choice,
                    "last_message_from_conversation": formatted_messages[-1],
                    "nb_of_message_in_conversation": len(formatted_messages),
                    "system_prompt": default_system_prompt["content"],
                    "cloze": cloz,
                    "Voice2Anki_version": shared.VERSION,
                    "request_information": shared.request,
                    })

    yel(f"\n\nLLM answer: '{cloz}'")
    whi(f"LLM cost: {shared.pv['total_llm_cost']} (${tkn_cost_dol:.3f}, not counting whisper)")
    red(f"Total LLM cost so far: ${shared.pv['total_llm_cost']:.4f} (not counting whisper)")

    return cloz


@optional_typecheck
@trace
@Critical
def dirload_splitted(checkbox: bool, *audios: Optional[Union[List, bool]]) -> List[Union[dict, gr.Audio, str, bool, None]]:
    """
    load the audio file that were splitted previously one by one in the
    available audio slots
    """
    if not checkbox:
        whi("Not running Dirload because checkbox is unchecked")
        return audios
    else:
        assert shared.pv["enable_dirload"], "Incoherent UI"

    # warn the user if there is one None surrounded by non None
    if not all(a is None for a in audios):
        check = False
        for i, a in enumerate(audios):
            if a is None:
                if check:
                    gr.Warning(red("There seems to be a missing audios in the slots. This can indicate a bug so you are advised to restart the app;"))
                    break
                check = True

    # make sure to move any empty slot at the end
    audios = [a for a in audios if a is not None]
    empty_slots = shared.audio_slot_nb - len(audios)
    while len(audios) < shared.audio_slot_nb:
        audios += [None]

    if empty_slots == shared.audio_slot_nb:
        if not shared.dirload_queue.empty:
            assert not (shared.dirload_queue["moved"] == False).all(), "shared.dirload_queue contains moved files even though you're loading all slots. This is suspicious."

    # check how many audio are needed
    whi(f"Number of empty sound slots: {empty_slots}")
    if empty_slots < 0:
        gr.Warning(red("Invalid number of empty audio slots: {empty_slots}!"))
        return audios
    if len(audios) > shared.audio_slot_nb:
        gr.Warning(red("Invalid number of audio slots: {empty_slots}!"))
        return audios
    if not empty_slots:
        gr.Warning(red("No empty audio slots!"))
        return audios

    # sort by oldest
    # sort by name
    shared.dirload_queue = shared.dirload_queue.sort_index()
    if not shared.dirload_queue.empty:
        max_n = max(shared.dirload_queue["n"].values)
    else:
        max_n = 0
    for todo_file in sorted([p for p in shared.splitted_dir.rglob("*.mp3")], key=lambda x: str(x)):  # by oldest: key=lambda x: x.stat().st_ctime)
        todo_file = str(todo_file)
        if todo_file not in shared.dirload_queue.index:
            with shared.dirload_lock:
                shared.dirload_queue.loc[todo_file, :] = False
                shared.dirload_queue.loc[todo_file, "temp_path"] = None
                shared.dirload_queue.loc[todo_file, "n"] = max_n + 1
            max_n += 1
        else:
            assert shared.dirload_queue.loc[todo_file, "moved"] is False, f"File {todo_file} is already in shared.dirload_queue and moved is True"
    shared.dirload_queue = shared.dirload_queue.sort_index()

    if not (shared.dirload_queue["moved"] == False).any():
        gr.Warning(red("No mp3 files in shared.dirload_queue"))
        return audios

    # check that loaded audio are all next to each other
    loadeds = shared.dirload_queue[shared.dirload_queue["loaded"] == True].index.tolist()
    index = shared.dirload_queue.index.tolist()
    assert ",".join(loadeds) in ",".join(index), f"Loaded audio are not neighbors!"

    # iterate over each files from the dir. If images are found, load them
    # into gallery but if the images are found after sounds, stops iterating
    sounds_to_load = []
    new_threads = []
    todo_path = shared.dirload_queue[shared.dirload_queue["moved"] == False]
    todo_path = todo_path[todo_path["loaded"] == False]
    for path in todo_path.index.tolist()[:empty_slots]:
        path = Path(path)
        to_temp = shared.tmp_dir / path.name
        shutil.copy2(path, to_temp)
        assert (path.exists() and (to_temp).exists()), "unexpected sound location"

        to_temp = sound_preprocessing(to_temp)
        with shared.dirload_lock:
            shared.dirload_queue.loc[str(path), "temp_path"] = str(to_temp)
            shared.dirload_queue.loc[str(path), "sound_preprocessed"] = True

        whi(f"Will load sound {to_temp}")
        sounds_to_load.append(
                gr.update(
                    value=to_temp,
                    label=Path(to_temp).name,
                    )
                )
        thread = threading.Thread(
            target=thread_whisp_then_llm,
            args=(to_temp,),
            )
        with shared.dirload_lock:
            shared.dirload_queue.loc[str(path), "transcribed"] = "started"
        thread.start()
        new_threads.append(thread)
        with shared.dirload_lock:
            shared.dirload_queue.loc[str(path), "loaded"] = True

    whi(f"Loading {len(sounds_to_load)} sounds from splitted")
    output = audios[:-len(sounds_to_load)] + sounds_to_load

    if new_threads:
        if len(sounds_to_load) == len(output):
            # loaded all the slots, so waiting for the first transcription to
            # finish
            whi("Waiting for first transcription to finish")
            new_threads[0].join()
            whi("Finished first transcription.")
            with shared.thread_lock:
                shared.running_threads["transcribing_audio"].extend(new_threads[1:])
        else:
            # the sound in the first slot was not loaded by this function so
            # not waiting for the transcription
            with shared.thread_lock:
                shared.running_threads["transcribing_audio"].extend(new_threads)

    with shared.dirload_lock:
        try:
            while len(shared.dirload_queue[shared.dirload_queue["loaded"] == True]) > shared.audio_slot_nb:
                p = shared.dirload_queue[shared.dirload_queue["loaded"] == True].iloc[0].name
                assert shared.dirload_queue.loc[p, "moved"] is False, f"File {p} was already moved"
                assert not shared.dirload_queue.loc[p, "transcribed"] in [False, "started"], f"File {p} shouldn't have to be moved as it has not been transcribed"
                if shared.dirload_queue.loc[p, "alfreded"] in [False, "started"]:
                    gr.Warning(red(f"File {p} was moved but had not been sent to alfred"))
                red(f"Moving {p} to done_dir")
                shutil.copy2(p, shared.done_dir / Path(p).name)
                assert (shared.done_dir / (Path(p).name)).exists(), f"Error when moving {p}"
                Path(p).unlink(missing_ok=False)
                shared.dirload_queue.loc[p, "loaded"] = False
                shared.dirload_queue.loc[p, "moved"] = True
        except Exception as err:
            gr.Warning(red(f"Error when moving audio, restart app to avoid unexpected behaviors!\nError: {err}"))
            raise

    while len(output) < shared.audio_slot_nb:
        output.append(None)

    assert len(output) == shared.audio_slot_nb, "Invalid number of audio slots in output"

    return output

@optional_typecheck
@trace
@Critical
def dirload_splitted_last(checkbox: bool) -> Union[str, gr.Audio, dict, None]:
    """wrapper for dirload_splitted to only load the last slot. This is faster
    because gradio does not have to send all 5 sounds if I just rolled"""
    audios = [True] * (shared.audio_slot_nb - 1) + [None]
    new_audio = dirload_splitted(checkbox, *audios)[-1]
    return new_audio

@optional_typecheck
@trace
def audio_edit(
        audio: Union[str, dict],
        audio_txt: str,
        txt_audio: str,
        txt_whisp_prompt: str,
        txt_whisp_lang: str,
        txt_chatgpt_cloz: str,
        txt_chatgpt_context: str) -> str:
    """function called by a microphone. It will use whisper to transcribe
    your voice. Then use the instructions in your voice to modify the
    output from LLM."""

    os.environ["OPENAI_API_KEY"] = shared.pv["txt_openai_api_key"].strip()
    if not shared.pv["txt_openai_api_key"]:
        raise Exception(red("No API key provided for OpenAI in the settings."))


    assert (audio is None and audio_txt) or (audio is not None and audio_txt is None), f"Can't give both audio and text to AudioEdit"
    if not audio_txt:
        red("Transcribing audio for audio_edit.")
        with open(audio, "rb") as f:
            audio_hash = hashlib.sha256(f.read()).hexdigest()
        transcript = whisper_cached(
                audio_path=audio,
                audio_hash=audio_hash,
                txt_whisp_prompt=None,
                txt_whisp_lang=txt_whisp_lang,
                sld_whisp_temp=0,
                stt_model=shared.pv["stt_choice"],
                )
        instructions = transcript["text"]
    else:
        instructions = audio_txt

    sys_prompt = dedent("""
    You receive an anki flashcard created from an audio transcript. Your answer must be the same flashcard after applying modifications mentionned in the instructions.
    Don't answer anything else.
    Don't acknowledge those instructions.
    Don't use symbols to wrap your answer, just answer the modified flashcard.
    Always answer the full flashcard, never answer only the question or answer or something that is not a complete flashcard.

    Here are the instructions that were given to the model who created the questionnable flashcard:
    '''
    OTHER_SYSPROMPT
    '''

    I'm counting on you.
    """).replace("OTHER_SYSPROMPT", default_system_prompt["content"])
    prompt_example = dedent("""
    Context:
    '''
    Dictée vocale de cours sur la maladie de Parkinson
    '''
    Original audio transcript:
    '''
    Les traitements dans la maladie de Parkinson survenant après 65-70 ans est du L-Dopa avec inhibiteur de la dopa des carboxylases, éventuellement associé à un IMAOP, et prescription de Domperidone.
    '''
    Flashcard you have to modify:
    '''
    Quels sont les traitements recommandés dans la maladie de Parkinson survenant après 65-70 ans ?<br/>{{c1::L-Dopa avec inhibiteur de la dopa des carboxylases, éventuellement associé à un IMAO-B, et prescription de Domperidone}}.
    '''
    Instructions:
    '''
    Commence la réponse par la question.
    '''
    """)
    answer_example = dedent("""
    Quels sont les traitements recommandés dans la maladie de Parkinson survenant après 65-70 ans ?<br/>{{c1::Les traitements recommandés dans la maladie de Parkinson survenant après 65-70 and sont le L-Dopa avec inhibiteur de la dopa des carboxylases, éventuellement associé à un IMAO-B, et prescription de Domperidone}}.
    """)
    prompt_example2 = dedent("""
    Context:
    '''
    Dictée vocale de cours sur la sclérose en plaque
    '''
    Original audio transcript:
    '''
    Le nom des critères diagnostiques de la dissémination spatiale et temporelle dans la sclérose en plaque est les critères de McDonald 2017.
    '''
    Flashcard you have to modify:
    '''
    Quel est le nom des critères diagnostiques de la dissémination spatiale et temporelle dans la sclérose en plaque ?<br/>{{c1::Les critères de McDonald 2017.}}
    '''
    Instructions:
    '''
    Réformule la réponse pour qu'elle commence comme la question, c'est plus naturel.
    '''
    """)
    answer_example2 = dedent("""
    Quel est le nom des critères diagnostiques de la dissémination spatiale et temporelle dans la sclérose en plaque ?<br/>{{c1::Le nom des critères diagnostiques de la dissémination spatiale et temporelle dans la sclérose en plaque est les critères de McDonald 2017.}}
    """)
    cloze = txt_chatgpt_cloz.replace("\n", "<br/>")
    prompt = dedent(f"""
    Context:
    '''
    {txt_chatgpt_context}
    '''
    Original audio transcript:
    '''
    {txt_audio}
    '''
    Flashcard you have to modify:
    '''
    {cloze}
    '''
    Instructions:
    '''
    {instructions}
    '''
    """)
    messages = [
            {
                "role": "system",
                "content": sys_prompt,
                },
            {
                "role": "user",
                "content": prompt_example,
                },
            {
                "role": "assistant",
                "content": answer_example,
                },
            {
                "role": "user",
                "content": prompt_example2,
                },
            {
                "role": "assistant",
                "content": answer_example2,
                },
            {
                "role": "user",
                "content": prompt,
                }
            ]

    #model_to_use = "openai/gpt-4o"
    model_to_use = shared.pv["llm_choice"]
    model_price = shared.llm_info[model_to_use]

    whi(f"Editing via {model_to_use}:")
    whi(prompt)
    response = litellm.completion(
            model=model_to_use,
            messages=messages,
            temperature=0,
            )

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]

    tkn_cost_dol = input_tkn_cost * model_price["input_cost_per_token"] + output_tkn_cost * model_price["output_cost_per_token"]
    shared.pv["total_llm_cost"] += tkn_cost_dol
    cloz = response["choices"][0]["message"]["content"]
    cloz = cloz.replace("<br/>", "\n").strip()  # for cosmetic purposes in the textbox

    yel(f"\n\nLLM answer:\n{cloz}\n\n")
    red(f"Total LLM cost so far: ${shared.pv['total_llm_cost']:.4f} (not counting whisper)")

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        red(f"LLM's reason to stop was not 'stop' but '{reason}'")

    return cloz, None, None

@optional_typecheck
@trace
def gather_threads(thread_keys: List[str]) -> None:
    n_running = {k: sum([t.is_alive() for t in threads]) for k, threads in shared.running_threads.items() if k == thread_keys}
    i = 0
    while sum(n_running.values()):
        i += 1
        n_running = {k: sum([t.is_alive() for t in threads]) for k, threads in shared.running_threads.items() if k == thread_keys}
        if i % 10 == 0:
            for k, n in n_running.items():
                if n == 0:
                    thread_keys.remove(k)
            yel(f"Waiting for {sum(n_running.values())} threads to finish from {thread_keys}")
        time.sleep(0.1)

@optional_typecheck
@trace
def wait_for_queue(q: queue.Queue, source: str, t=1):
    "source : https://stackoverflow.com/questions/19206130/does-queue-get-block-main"
    start = time.time()
    while True:
        try:
            # Waits for X seconds, otherwise throws `Queue.Empty`
            data = q.get(True, t)
            break
        except queue.Empty:
            red(f"Waiting for {source} queue to output (for {time.time()-start:.1f}s)")
            data = None
    return data


@optional_typecheck
@trace
def kill_threads() -> None:
    """the threads in timeout are stored in the shared module, if they
    get replaced by None the threads will be ignored.
    Also resets the smartcache"""
    with shared.thread_lock:
        for k in shared.running_threads:
            n = sum([t.is_alive() for t in shared.running_threads[k]])
            if n >= 1:
                red(f"Killing the {n} alive threads of {k}")
            else:
                whi(f"No thread to kill of {k}")
            shared.running_threads[k] = []
        with shared.timeout_lock:
            shared.smartcache.clear()


@optional_typecheck
@trace
def Voice2Anki_db_save(
        txt_chatgpt_cloz: str,
        txt_chatgpt_context: str,
        txt_audio: str,
        note_ids: List[int],
        ) -> None:
    """when an anki card is created, find the information about its creation
    in the shared module then save it to the db. It can be missing from the db
    if the result from alfred was loaded from cache for example."""
    if shared.llm_to_db_buffer:
        buffer_keys = [k for k in shared.llm_to_db_buffer.keys()]
        ratio_buffer_keys = [levratio(txt_chatgpt_cloz, x) for x in buffer_keys]
        max_ratio = max(ratio_buffer_keys)
        closest_buffer_key = buffer_keys[ratio_buffer_keys.index(max_ratio)]
    else:
        max_ratio = 1
    orig_save_dict = {
        "type": "anki_card",
        "timestamp": time.time(),
        "token_cost": None,
        "temperature": None,
        "LLM_context": txt_chatgpt_context,
        "Voice2Anki_profile": shared.pv.profile_name,
        "transcribed_input": txt_audio,
        "model_name": f"Probably:{shared.pv['llm_choice']}",
        "last_message_from_conversation": None,
        "nb_of_message_in_conversation": None,
        "system_prompt": default_system_prompt["content"],
        "cloze": txt_chatgpt_cloz,
        "Voice2Anki_version": shared.VERSION,
        "request_information": shared.request,
        "anki_nids": note_ids,
        "save_dict_trust": "high",
    }
    if max_ratio >= 90:
        save_dict = json.loads(shared.llm_to_db_buffer[closest_buffer_key])
        save_dict["save_dict_trust"] = "less"
        save_dict["orig_save_dict_diff"] = {}
        for k, v in orig_save_dict.items():
            if k not in save_dict or save_dict[k] != v:
                save_dict["orig_save_dict_diff"][k] = v
        del shared.llm_to_db_buffer[closest_buffer_key]
    else:
        save_dict = orig_save_dict

    with shared.thread_lock:
        if shared.running_threads["saving_chatgpt"]:
            [t.join() for t in shared.running_threads["saving_chatgpt"]]
        while shared.running_threads["saving_chatgpt"]:
            shared.running_threads["saving_chatgpt"].pop()
    thread = threading.Thread(
            target=store_to_db,
            name="saving_chatgpt",
            kwargs={
                "dictionnary": save_dict,
                "db_name": "llm"})
    thread.start()
    with shared.thread_lock:
        shared.running_threads["saving_whisper"].append(thread)


@optional_typecheck
@trace
def to_anki(
        audio_mp3_1: Optional[Union[dict, str]],
        txt_audio: Optional[str],
        txt_chatgpt_cloz: str,
        txt_chatgpt_context: Optional[str],
        txt_deck: str,
        txt_tags: Optional[List[str]],
        gallery: Optional[List[Union[np.ndarray, gr.Gallery, dict]]],
        check_marked: bool,
        txt_extra_source: Optional[str],
        ) -> None:
    "function called to do wrap it up and send to anki"
    whi("Entering to_anki")
    if not txt_audio:
        raise Exception(red("missing txt_audio"))
    if not txt_chatgpt_cloz:
        raise Exception(red("missing txt_chatgpt_cloz"))
    if not txt_deck:
        raise Exception(red("missing txt_deck"))
    if not txt_tags:
        raise Exception(red("missing txt_tags"))

    if txt_chatgpt_cloz.startswith("Error with ChatGPT"):
        raise Exception(red(f"Error with chatgpt: '{txt_chatgpt_cloz}'"))

    clozetext, thinking = split_thinking(txt_chatgpt_cloz)

    # make sure the audio is a valid path
    if isinstance(audio_mp3_1, dict):
        audio_mp3_oname = audio_mp3_1["orig_name"]
        assert Path(audio_mp3_1["path"]).exists(), f"Not found: {audio_mp3_1['path']}"
    else:
        audio_mp3_oname = audio_mp3_1

    # checks clozes validity
    clozes = [c.strip() for c in clozetext.split("#####") if c.strip()]
    if not clozes or "{{c1::" not in clozetext:
        raise Exception(red(f"Invalid cloze: '{clozetext}'"))

    if "alfred" in clozetext.lower():
        raise Exception(red(f"COMMUNICATION REQUESTED:\n'{clozetext}'"))
    if re.findall("{{c\d:}}", clozetext.lower()):
        raise Exception(red(f"EMPTY CLOZE DETECTED:\n'{clozetext}'"))

    # load the source text of the image in the gallery
    txt_source_queue = queue.Queue()
    txt_source = None
    if gallery is None or len(gallery) == 0 or shared.pv["enable_gallery"] is False:
        red("you should probably specify an image in source")
        txt_source = ""
    else:
        assert shared.pv["enable_gallery"] is True, "Incoherent UI"
        thread = threading.Thread(
                target=get_img_source,
                args=(gallery, txt_source_queue)
                )
        thread.start()
        with shared.thread_lock:
            shared.running_threads["ocr"].append(thread)

    # send audio to anki
    if audio_mp3_1 is not None:
        add_audio_to_anki_queue = queue.Queue()
        thread = threading.Thread(
                target=add_audio_to_anki,
                args=(audio_mp3_1, add_audio_to_anki_queue),
                )
        thread.start()
        with shared.thread_lock:
            shared.running_threads["add_audio_to_anki"].append(thread)

    # send to anki
    metadata = {
        "author": "Voice2Anki",
        "transcripted_text": txt_audio,
        "chatgpt_context": txt_chatgpt_context,
        "llm_used": shared.pv["llm_choice"],
        "stt_used": shared.pv["stt_choice"],
        "version": shared.VERSION,
        "timestamp": time.time(),
        "user-agent": shared.request["user-agent"] if shared.request else None,
        "mp3": str(audio_mp3_1),
        "thinking": thinking,
        "client_type": "gui" if shared.client_type == "gui" else "cli",
    }
    results = []

    # mention in the metadata the original mp3 name
    if not shared.dirload_queue.empty:
        audio_row = shared.dirload_queue.loc[shared.dirload_queue["temp_path"].str.endswith(audio_mp3_oname)==True]
        if audio_row.empty:
            # the ' can be stripped from the original path
            audio_row = shared.dirload_queue.loc[shared.dirload_queue["temp_path"].str.replace("'", "").str.endswith(audio_mp3_oname)==True]
        if audio_row.empty:
            # the _proc can be present or not
            audio_row = shared.dirload_queue.loc[shared.dirload_queue["temp_path"].str.replace("'", "").str.replace("_proc", "").str.endswith(audio_mp3_oname)==True]
        assert not audio_row.empty, f"Empty for {audio_mp3_1}\nThe temp_path present were {shared.dirload_queue['temp_path'].tolist()}"
        assert len(audio_row) == 1, f"More than one audio found in dirload_queue for {audio_mp3_1}"
        metadata["original_mp3_path"] = audio_row.reset_index()["path"].tolist()[0]

    whi("Sending to anki:")

    # sending sound file to anki media
    if audio_mp3_1 is not None:
        audio_html = wait_for_queue(add_audio_to_anki_queue, "add_audio_to_anki")
    else:
        audio_html = "No audio"
    if "Error" in audio_html:  # then out is an error message and not the source
        gather_threads(["add_audio_to_anki", "ocr"])
        raise Exception(f"Error in audio_html: '{audio_html}'")

    # gather text from the source image(s)
    if txt_source is None:
        txt_source = wait_for_queue(txt_source_queue, "txt_source")
        if "Error" in txt_source:  # then out is an error message and not the source
            gr.Warning(red(f"Error in gallery source but continuing nonetheless: '{txt_source}'"))
            # gather_threads(["add_audio_to_anki", "ocr"])
            # raise Exception(f"Error in gallery source: '{txt_source}'")

    # anki tags
    new_tags = txt_tags + [f"Voice2Anki::{today}"]
    if check_marked:
        new_tags += ["marked"]
    if "<img" not in txt_source:
        # if no image in source: add a tag to find them easily later on
        new_tags += ["Voice2Anki::no_img_in_source"]

    if txt_extra_source is None:
        txt_extra_source = ""
    else:
        txt_extra_source = txt_extra_source.strip()

    if not shared.dirload_queue.empty:
        with shared.dirload_lock:
            audio_row.loc[:, "ankified"] = "started"

    # if the txt_audio contains multiple cloze, either the number of created
    # cloze correspond to the number of sections in txt_audio, then the
    # metadata should reflect that. Or the number differ and each cloze
    # will be treated as if it was created from the same input text
    metadatas = [metadata for i in clozes]
    audio_split = txt_audio.split("\n\n")
    audio_split = [a.strip() for a in audio_split if a.strip()]
    if len(audio_split) > 1 and len(audio_split) == len(clozes):
        red("Each cloze will be assigned to the corresponding txt_audio section.")
        for i in range(len(metadatas)):
            metadatas[i]["transcripted_text"] = audio_split[i]

    results = add_note_to_anki(
            bodies=clozes,
            source=txt_source,
            source_extra=txt_extra_source,
            source_audio=audio_html,
            notes_metadata=metadatas,
            tags=new_tags,
            deck_name=txt_deck)

    errors = [f"#{results.index(r)+1}/{len(results)}: {r}" for r in results if not str(r).isdigit()]
    results = [r for r in results if str(r).isdigit()]
    shared.added_note_ids.append(results)

    if not len(results) == len(clozes):
        gather_threads(["audio_to_anki", "ocr"])
        raise Exception(red(f"Some flashcards were not added:{','.join(errors)}"))
    if not shared.dirload_queue.empty:
        with shared.dirload_lock:
            audio_row.loc[:, "ankified"] = True

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    # add the latest generated cards to the message buffer
    if len(audio_split) > 1:
        if len(audio_split) != len(clozes):
            red("Not saving card to message buffer because the number of split in txt_audio is not the same as in clozes.")
        else:
            for aud, cl in zip(audio_split, clozes):
                if aud not in [mb["unformatted_txt_audio"] for mb in shared.message_buffer] and cl not in [mb["unformatted_txt_chatgpt_cloz"] for mb in shared.message_buffer]:
                    shared.message_buffer.append(
                            {
                                "unformatted_txt_audio": aud,
                                "unformatted_txt_chatgpt_cloz": cl,
                                "question": transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", aud),
                                "answer": cl.replace("\n", "<br/>"),
                                "was_split": True,
                                }
                            )
    else:
        if txt_audio not in [mb["unformatted_txt_audio"] for mb in shared.message_buffer] and clozetext not in [mb["unformatted_txt_chatgpt_cloz"] for mb in shared.message_buffer]:
            shared.message_buffer.append(
                    {
                        "unformatted_txt_audio": txt_audio,
                        "unformatted_txt_chatgpt_cloz": txt_chatgpt_cloz,
                        "question": transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", txt_audio),
                        "answer": clozetext.replace("\n", "<br/>"),
                        "was_split": False,
                        }
                    )

    # cap the number of messages
    shared.message_buffer = shared.message_buffer[-shared.max_message_buffer:]
    shared.pv["message_buffer"] = shared.message_buffer

    Voice2Anki_db_save(
        txt_chatgpt_cloz=txt_chatgpt_cloz,
        txt_chatgpt_context=txt_chatgpt_context,
        txt_audio=txt_audio,
        note_ids=results,
    )

    gather_threads(["audio_to_anki", "ocr", "saving_chatgpt"])
