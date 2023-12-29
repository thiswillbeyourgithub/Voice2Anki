import os
import asyncio
import pickle
from tqdm import tqdm
import re
import uuid
import Levenshtein as lev
import shutil
import queue
import threading
import hashlib
import base64
import json
from textwrap import dedent, indent
import rtoml
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import joblib

import litellm
import openai

from .anki_utils import add_to_anki, audio_to_anki, sync_anki
from .shared_module import shared
from .logger import red, whi, yel, store_to_db, trace, Timeout
from .memory import prompt_filter, load_prev_prompts, tokenize, transcript_template, default_system_prompt
from .media import sound_preprocessing, get_img_source, format_audio_component
from .profiles import ValueStorage

splitted_dir = Path("./user_directory/splitted")
done_dir = Path("./user_directory/done")
unsplitted_dir = Path("./user_directory/unsplitted")
tmp_dir = Path("/tmp/gradio")


Path("user_directory").mkdir(exist_ok=True)
splitted_dir.mkdir(exist_ok=True)
unsplitted_dir.mkdir(exist_ok=True)
done_dir.mkdir(exist_ok=True)

shared.pv = ValueStorage()
pv = shared.pv
shared.message_buffer = pv["message_buffer"]

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

stt_cache = joblib.Memory("cache/transcript_cache", verbose=0)
llm_cache = joblib.Memory("cache/llm_cache", verbose=0)

@trace
def clear_llm_cache():
    # reset the llm cache to make sure shared.llm_to_db_buffer is up to date
    llm_cache.clear()

# trigger a sync on startup to test if anki is running and with ankiconnect enabled
sync_anki()

@trace
def pop_buffer():
    "remove the latest card from message buffer"
    removed = shared.message_buffer.pop(-1)
    red(f"Message buffer size is now {len(shared.message_buffer)} after removing '{removed}'")

def floatizer(func):
    "used to cast the ints as float to make sure the cache is used"
    def wrapper(*args, **kwargs):
        args = [float(ar) if isinstance(ar, int) else ar for ar in args]
        kwargs = {k: float(v) if isinstance(v, int) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

@floatizer
@trace
@stt_cache.cache(ignore=["audio_path"])
def whisper_cached(
        audio_path,
        audio_hash,
        modelname,
        txt_whisp_prompt,
        txt_whisp_lang,
        sld_whisp_temp,
        ):
    """this is a call to OpenAI's whisper. It's called as soon as the
    recording is done to begin caching. The audio_path can change so a hash
    of the content is used instead."""
    red(f"Calling whisper because not in cache: {audio_path}")
    assert shared.pv["txt_openai_api_key"], f"Missing openai key, needed for whisper"
    client = openai.OpenAI(api_key=shared.pv["txt_openai_api_key"].strip())
    assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
    try:
        cnt = 0
        while True:
            try:
                cnt += 1
                with open(audio_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model=modelname,
                        file=audio_file,
                        prompt=txt_whisp_prompt,
                        language=txt_whisp_lang,
                        temperature=sld_whisp_temp,
                        response_format="verbose_json",
                        )
                    if sld_whisp_temp == 0:
                        temps = [seg["temperature"] for seg in transcript["segments"]]
                        if sum(temps) / len(temps) == 1:
                            raise Exception(red(f"Whisper increased temperature to maximum, probably because no words could be heard."))

                return transcript
            except openai.RateLimitError as err:
                if cnt >= 5:
                    raise Exception(red(f"Cached whisper: RateLimitError >5: '{err}'"))
                else:
                    gr.Error(red(f"Cached whisper: RateLimitError #{cnt}/5 from cached whisper: '{err}'"))
                    time.sleep(2 * cnt)
    except Exception as err:
        raise Exception(red(f"Error when cache transcribing audio: '{err}'"))

@trace
def thread_whisp_then_llm(
        audio_mp3,
        txt_whisp_prompt,
        txt_whisp_lang,
        sld_whisp_temp,

        txt_chatgpt_context,
        txt_profile,
        max_token,
        temperature,
        sld_buffer,
        llm_choice,
        txt_keywords,
        ):
    """run whisper on the audio and return nothing. This is used to cache in
    advance and in parallel the transcription."""
    if audio_mp3 is None:
        return

    if txt_whisp_prompt is None:
        return

    if txt_whisp_lang is None:
        return

    whi("Transcribing audio for the cache")
    modelname = "whisper-1"
    audio_mp3 = format_audio_component(audio_mp3)
    if shared.latest_stt_used != modelname:
        shared.latest_stt_used = modelname

    os.environ["OPENAI_API_KEY"] = shared.pv["txt_openai_api_key"].strip()
    os.environ["REPLICATE_API_KEY"] = shared.pv["txt_replicate_api_key"].strip()
    if not shared.pv["txt_openai_api_key"] and not shared.pv["txt_replicate_api_key"]:
        raise Exception(red("No API key provided for either OpenAI or replicate in the settings."))

    with open(audio_mp3, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    tmp_df = shared.dirload_queue.reset_index().set_index("temp_path")
    assert str(audio_mp3) in tmp_df.index, f"Missing {audio_mp3} from shared.dirload_queue"
    assert tmp_df.index.tolist().count(str(audio_mp3)) == 1, f"Duplicate temp_path in shared.dirload_queue: {audio_mp3}"
    orig_path = tmp_df.loc[str(audio_mp3), "path"]

    txt_audio = whisper_cached(
            audio_mp3,
            audio_hash,
            modelname,
            txt_whisp_prompt,
            txt_whisp_lang,
            sld_whisp_temp,
            ).text
    with shared.dirload_lock:
        shared.dirload_queue.loc[orig_path, "transcribed"] = txt_audio
        shared.dirload_queue.loc[orig_path, "alfreded"] = "started"

    cloze = alfred(txt_audio, txt_chatgpt_context, txt_profile, max_token, temperature, sld_buffer, llm_choice, txt_keywords, cache_mode=True)
    with shared.dirload_lock:
        shared.dirload_queue.loc[orig_path, "alfreded"] = cloze

@trace
def transcribe(audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp):
    "turn the 1st audio track into text"
    whi("Transcribing audio")

    if audio_mp3_1 is None:
        raise Exception(red("Error: None audio_mp3_1"))

    if txt_whisp_prompt is None:
        raise Exception(red("Error: None whisper prompt"))

    if txt_whisp_lang is None:
        raise Exception(red("Error: None whisper language"))

    modelname = "whisper-1"
    if shared.latest_stt_used != modelname:
        shared.latest_stt_used = modelname

    audio_mp3_1 = format_audio_component(audio_mp3_1)

    with open(audio_mp3_1, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        whi(f"Asking Cached Whisper for {audio_mp3_1}")
        transcript = whisper_cached(
                audio_mp3_1,
                audio_hash,
                modelname,
                txt_whisp_prompt,
                txt_whisp_lang,
                sld_whisp_temp
                )
        with open(audio_mp3_1, "rb") as audio_file:
            mp3_content = audio_file.read()
        txt_audio = transcript.text
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
                        "whisper_language": txt_whisp_lang,
                        "whisper_context": txt_whisp_prompt,
                        "whisper_temperature": sld_whisp_temp,
                        "Voice2Anki_profile": pv.profile_name,
                        "transcribed_input": txt_audio,
                        "full_whisper_output": transcript,
                        "model_name": modelname,
                        "audio_mp3": base64.b64encode(mp3_content).decode(),
                        "Voice2Anki_version": shared.VERSION,
                        },
                    "db_name": "whisper"
                    })
        thread.start()
        with shared.thread_lock:
            shared.running_threads["saving_whisper"].append(thread)

        txt_audio = txt_audio.replace(" Stop. ", "\n\n").strip()

        return txt_audio
    except Exception as err:
        raise Exception(red(f"Error when transcribing audio: '{err}'"))


@trace
def flag_audio(
        txt_profile,
        txt_audio,
        txt_whisp_lang,
        txt_whisp_prompt,
        txt_chatgpt_cloz,
        txt_chatgpt_context,
        gallery,
        ):
    """copy audio in slot #1 to the user_directory/flagged folder"""
    # move audio file
    if not (shared.dirload_queue["loaded"] == True).any():
        raise Exception(red("No loaded files in shared.dirload_queue"))
    aud = Path(shared.dirload_queue[shared.dirload_queue["loaded"] == True].iloc[0].name)
    assert aud.exists(), f"File not found: {aud}"
    new_filename = f"user_directory/flagged/{aud.name}"
    if Path(new_filename).exists():
        raise Exception(red(f"Audio you're trying to flag already exists: {new_filename}"))
    shutil.copy2(aud, new_filename)
    red(f"Flagged {aud} to {new_filename}")

    # make sure the gallery is saved as image and not as path
    if gallery is not None:
        if hasattr(gallery, "root"):
            gallery = gallery.root
        assert isinstance(gallery, (type(None), list)), "Gallery is not a list or None"
        new_gal = []
        for img in gallery:
            try:
                decoded = cv2.imread(img.image.path, flags=1)
            except:
                decoded = cv2.imread(img["image"]["path"], flags=1)
            new_gal.append(decoded)
        gallery = new_gal

    # move the other component's data
    to_save = {
            "txt_profile": txt_profile,
            "txt_audio": txt_audio,
            "txt_whisp_lang": txt_whisp_lang,
            "txt_whisp_prompt": txt_whisp_prompt,
            "txt_chatgpt_cloz": txt_chatgpt_cloz,
            "txt_chatgpt_context": txt_chatgpt_context,
            "gallery": gallery,
            }
    with open(f"user_directory/flagged/{aud.name}.pickle", "wb") as f:
        pickle.dump(to_save, f)


@trace
def pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, txt_keywords, cache_mode):
    """used to prepare the prompts for alfred call. This is a distinct
    function to make it callable by the cached function too."""
    # don't print when using cache
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
    prev_prompts = load_prev_prompts(profile)

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
            if lev.ratio(txt_audio, mb["unformatted_txt_audio"]) >= 0.95:
                # skip this one
                red(f"Skipped buffer: {mb}")
                continue
            if lev.ratio(txt_audio, mb["question"]) >= 0.95:
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
            temperature,
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
    # add the current prompt
    formatted_messages.append(new_prompt)

    # check the number of token is fine and format the previous
    # prompts in chatgpt format
    tkns = 0
    for mess in formatted_messages:
        tkns += len(tokenize(mess["content"]))

    yel(f"Number of messages that will be sent to ChatGPT: {len(formatted_messages)} (representing {tkns} tokens)")

    if tkns >= 15700:
        red("More than 15700 tokens before calling ChatGPT. Bypassing to ask "
            "with fewer tokens to make sure you have room for the answer")
        return pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature, sld_buffer, txt_keywords, cache_mode)

    assert tkns <= 15700, f"Too many tokens: {tkns}"

    # print prompts used for the call:
    n = len(formatted_messages)
    whi("ChatGPT prompt:")
    for i, p in enumerate(formatted_messages):
        whi(f"* {i+1}/{n}: {p['role']}")
        whi(indent(p['content'], " " * 5))

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

@floatizer
@trace
@Timeout(180)
@llm_cache.cache(ignore=["cache_mode", "sld_buffer"])
def alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, llm_choice, txt_keywords, cache_mode=False):
    "send the previous prompt and transcribed speech to the LLM"
    red(f"Calling Alfred in cache_mode={cache_mode} for transcript '{txt_audio}'")
    if not txt_audio:
        raise Exception(red("No transcribed audio found."))
    if txt_audio.strip().startswith("Error"):
        raise Exception(red("Error when transcribing sound."))
    if not txt_chatgpt_context:
        raise Exception(red("No txt_chatgpt_context found."))
    if (("fred" in txt_audio.lower() and "image" in txt_audio.lower()) or ("change d'image" in txt_audio.lower())) and len(txt_audio) < 40:
        mess = f"Image change detected: '{txt_audio}'"
        if cache_mode:
            return red(mess)
        else:
            gr.Error(mess)
            return red(mess)

    os.environ["OPENAI_API_KEY"] = shared.pv["txt_openai_api_key"].strip()
    os.environ["REPLICATE_API_KEY"] = shared.pv["txt_replicate_api_key"].strip()
    if not shared.pv["txt_openai_api_key"] and not shared.pv["txt_replicate_api_key"]:
        raise Exception(red("No API key provided for either OpenAI or replicate in the settings."))

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
            gr.Error(red(f"Found only 1 split in '{txt_audio}' which is '{splits[0]}'"))
            txt_audio = splits[0]
        else:
            answers = asyncio.run(async_parallel_alfred(splits, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, llm_choice, txt_keywords, cache_mode))
            assert len(answers) == len(splits), "Unexpected length"
            return "\n#####\n".join(answers).strip()

    formatted_messages = pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, txt_keywords, cache_mode)
    for i, fm in enumerate(formatted_messages):
        if i == 0:
            assert fm["role"] == "system"
        elif i % 2 == 0:
            assert fm["role"] == "assistant"
        elif i % 2 == 1:
            assert fm["role"] == "user"

    # check no duplicate in messages
    if len(set([fm["content"] for fm in formatted_messages])) != len(formatted_messages):
        contents = [pm["content"] for pm in formatted_messages]
        dupli = [dp for dp in contents if contents.count(dp) > 1]
        raise Exception(f"{len(dupli)} duplicate prompts found: {dupli}")

    model_price = shared.llm_price[llm_choice]
    whi(f"Will use model {llm_choice}")

    cnt = 0
    while True:
        try:
            whi("Asking LLM")
            cnt += 1
            response = litellm.completion(
                    model=llm_choice,
                    messages=formatted_messages,
                    temperature=temperature,
                    )
            break
        except (litellm.RateLimitError, openai.RateLimitError) as err:
            if cnt >= 2:
                raise Exception(red("LLM: too many retries."))
            red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
            time.sleep(2 * cnt)

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]
    tkn_cost = [input_tkn_cost, output_tkn_cost]

    # in case recur improv is called
    if llm_choice != shared.latest_llm_used:
        shared.latest_llm_used = llm_choice

    tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
    pv["total_llm_cost"] += tkn_cost_dol

    cloz = response["choices"][0]["message"]["content"]
    # for cosmetic purposes in the textbox
    cloz = cloz.replace("<br/>", "\n")
    cloz = cloz.replace("&nbsp;", " ")
    cloz = cloz.replace("#####", "\n#####\n")  # make sure to separate cleanly the clozes
    cloz = "\n".join([cl.strip() for cl in cloz.splitlines() if cl.strip()])

    # if contains cloze in multiple parts but in the same line, merge them
    sl = cloz.splitlines()
    if len(sl) == 2:
        if "{{c" not in sl[0] and "}}" not in sl[0]:
            if sl[1].count("{{c1::") > 1:
                sl[1] = re.sub("}}(.*?){{c1::", r"\1", sl[1])
            cloz = "\n".join(sl)

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        red(f"LLM's reason to stop was not 'stop' but '{reason}'")

    # add to the shared module the infonrmation of this card creation.
    # if a card is created then this will be added to the db to
    # create LORA fine tunes later on.
    with shared.db_lock:
        shared.llm_to_db_buffer[cloz] = json.dumps(
                {
                    "type": "anki_card",
                    "timestamp": time.time(),
                    "token_cost": tkn_cost,
                    "temperature": temperature,
                    "LLM_context": txt_chatgpt_context,
                    "Voice2Anki_profile": pv.profile_name,
                    "transcribed_input": txt_audio,
                    "model_name": llm_choice,
                    "last_message_from_conversation": formatted_messages[-1],
                    "nb_of_message_in_conversation": len(formatted_messages),
                    "system_prompt": default_system_prompt["content"],
                    "cloze": cloz,
                    "Voice2Anki_version": shared.VERSION,
                    })

    yel(f"\n\nLLM answer:\n{cloz}\n\n")
    whi(f"LLM cost: {pv['total_llm_cost']} (${tkn_cost_dol:.3f}, not counting whisper)")
    red(f"Total LLM cost so far: ${pv['total_llm_cost']:.4f} (not counting whisper)")

    return cloz


@trace
def dirload_splitted(
        checkbox,
        txt_whisp_prompt,
        txt_whisp_lang,
        sld_whisp_temp,

        txt_chatgpt_context,
        txt_profile,
        max_token,
        temperature,
        sld_buffer,
        llm_choice,
        txt_keywords,

        *audios,
        ):
    """
    load the audio file that were splitted previously one by one in the
    available audio slots
    """
    if not checkbox:
        whi("Not running Dirload because checkbox is unchecked")
        return audios

    # make sure to move any empty slot at the end
    audios = [a for a in audios if a is not None]
    empty_slots = shared.audio_slot_nb - len(audios)
    while len(audios) < shared.audio_slot_nb:
        audios += [None]

    if empty_slots == shared.audio_slot_nb:
        if not shared.dirload_queue.empty:
            assert not (shared.dirload_queue["moved"] == False).all(), "shared.dirload_queue contains moved files even though you're loading all slots. This is suspiciou."

    # check how many audio are needed
    whi(f"Number of empty sound slots: {empty_slots}")
    if empty_slots < 0:
        gr.Error(red("Invalid number of empty audio slots: {empty_slots}!"))
        return audios
    if len(audios) > shared.audio_slot_nb:
        gr.Error(red("Invalid number of audio slots: {empty_slots}!"))
        return audios
    if not empty_slots:
        gr.Error(red("No empty audio slots!"))
        return audios

    # sort by oldest
    # sort by name
    shared.dirload_queue = shared.dirload_queue.sort_index()
    if not shared.dirload_queue.empty:
        max_n = max(shared.dirload_queue["n"].values)
    else:
        max_n = 0
    for todo_file in sorted([p for p in splitted_dir.rglob("*.mp3")], key=lambda x: str(x)):  # by oldest: key=lambda x: x.stat().st_ctime)
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
        gr.Error(red("No mp3 files in shared.dirload_queue"))
        return audios

    # iterate over each files from the dir. If images are found, load them
    # into gallery but if the images are found after sounds, stops iterating
    sounds_to_load = []
    new_threads = []
    todo_path = shared.dirload_queue[shared.dirload_queue["moved"] == False]
    todo_path = todo_path[todo_path["loaded"] == False]
    for path in todo_path.index.tolist()[:empty_slots]:
        path = Path(path)
        to_temp = tmp_dir / path.name
        shutil.copy2(path, to_temp)
        assert (path.exists() and (to_temp).exists()), "unexpected sound location"

        to_temp = sound_preprocessing(to_temp)
        with shared.dirload_lock:
            shared.dirload_queue.loc[str(path), "temp_path"] = str(to_temp)
            shared.dirload_queue.loc[str(path), "sound_preprocessed"] = True

        whi(f"Will load sound {to_temp}")
        sounds_to_load.append(to_temp)
        if txt_whisp_prompt and txt_whisp_lang:
            thread = threading.Thread(
                target=thread_whisp_then_llm,
                args=(
                        to_temp,
                        txt_whisp_prompt,
                        txt_whisp_lang,
                        sld_whisp_temp,

                        txt_chatgpt_context,
                        txt_profile,
                        max_token,
                        temperature,
                        sld_buffer,
                        llm_choice,
                        txt_keywords,
                        ),
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
        while len(shared.dirload_queue[shared.dirload_queue["loaded"] == True]) > shared.audio_slot_nb:
            p = shared.dirload_queue[shared.dirload_queue["loaded"] == True].iloc[0].name
            assert shared.dirload_queue.loc[p, "moved"] is False, f"File {p} was already moved"
            assert not shared.dirload_queue.loc[p, "transcribed"] in [False, "started"], f"File {p} shouldn't have to be moved as it has not been transcribed"
            if shared.dirload_queue.loc[p, "alfreded"] in [False, "started"]:
                gr.Error(red(f"File {p} was moved but had not been sent to alfred"))
            red(f"Moving {p} to done_dir")
            shutil.move(p, done_dir / Path(p).name)
            shared.dirload_queue.loc[p, "loaded"] = False
            shared.dirload_queue.loc[p, "moved"] = True

    while len(output) < shared.audio_slot_nb:
        output.append(None)

    assert len(output) == shared.audio_slot_nb, "Invalid number of audio slots in output"

    return output

@trace
def dirload_splitted_last(
        checkbox,
        txt_whisp_prompt,
        txt_whisp_lang,
        sld_whisp_temp,

        txt_chatgpt_context,
        txt_profile,
        max_token,
        temperature,
        sld_buffer,
        llm_choice,
        txt_keywords,
        ):
    """wrapper for dirload_splitted to only load the last slot. This is faster
    because gradio does not have to send all 5 sounds if I just rolled"""
    audios = [True] * (shared.audio_slot_nb - 1) + [None]
    return dirload_splitted(
            checkbox,
            txt_whisp_prompt,
            txt_whisp_lang,
            sld_whisp_temp,

            txt_chatgpt_context,
            txt_profile,
            max_token,
            temperature,
            sld_buffer,
            llm_choice,
            txt_keywords,

            *audios,
            )[-1]

@trace
def audio_edit(audio, audio_txt, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context):
    """function called by a microphone. It will use whisper to transcribe
    your voice. Then use the instructions in your voice to modify the
    output from LLM."""

    os.environ["OPENAI_API_KEY"] = shared.pv["txt_openai_api_key"].strip()
    if not shared.pv["txt_openai_api_key"]:
        raise Exception(red("No API key provided for OpenAI in the settings."))


    assert (audio is None and audio_txt) or (audio is not None and audio_txt is None), f"Can't give both audio and text to AudioEdit"
    if not audio_txt:
        red("Transcribing audio for audio_edit.")
        instructions = transcribe(
                audio,
                txt_whisp_prompt="Instruction: ",
                txt_whisp_lang=txt_whisp_lang,
                sld_whisp_temp=0,
                )
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

    model_to_use = "openai/gpt-4-1106-preview"
    model_price = shared.llm_price[model_to_use]

    whi(f"Editing via {model_to_use}:")
    whi(prompt)
    response = litellm.completion(
            model=model_to_use,
            messages=messages,
            temperature=0,
            )

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]

    tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
    pv["total_llm_cost"] += tkn_cost_dol
    cloz = response["choices"][0]["message"]["content"]
    cloz = cloz.replace("<br/>", "\n").strip()  # for cosmetic purposes in the textbox

    yel(f"\n\nLLM answer:\n{cloz}\n\n")
    red(f"Total LLM cost so far: ${pv['total_llm_cost']:.4f} (not counting whisper)")

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        red(f"LLM's reason to stop was not 'stop' but '{reason}'")

    return cloz, None, None

@trace
def gather_threads(thread_keys):
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

@trace
def wait_for_queue(q, source, t=1):
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


@trace
def kill_threads():
    """the threads in timeout are stored in the shared module, if they
    get replaced by None the threads will be ignored."""
    with shared.thread_lock:
        for k in shared.running_threads:
            n = sum([t.is_alive() for t in shared.running_threads[k]])
            if n >= 1:
                red(f"Killing the {n} alive threads of {k}")
            else:
                whi(f"No thread to kill of {k}")
            shared.running_threads[k] = []


@trace
def Voice2Anki_db_save(txt_chatgpt_cloz, txt_chatgpt_context, txt_audio):
    """when an anki card is created, find the information about its creation
    in the shared module then save it to the db. It can be missing from the db
    if the result from alfred was loaded from cache for example."""
    if shared.llm_to_db_buffer:
        buffer_keys = [k for k in shared.llm_to_db_buffer.keys()]
        dist_buffer_keys = [lev.ratio(txt_chatgpt_cloz, x) for x in buffer_keys]
        min_dist = min(dist_buffer_keys)
        closest_buffer_key = buffer_keys[dist_buffer_keys.index(min_dist)]
    else:
        min_dist = 0
    if min_dist < 0.90:
        save_dict = {
                "type": "anki_card",
                "timestamp": time.time(),
                "token_cost": None,
                "temperature": None,
                "LLM_context": txt_chatgpt_context,
                "Voice2Anki_profile": shared.pv.profile_name,
                "transcribed_input": txt_audio,
                "model_name": f"Probably:{shared.latest_llm_used}",
                "last_message_from_conversation": None,
                "nb_of_message_in_conversation": None,
                "system_prompt": default_system_prompt["content"],
                "cloze": txt_chatgpt_cloz,
                "Voice2Anki_version": shared.VERSION,
                }
    else:
        save_dict = json.loads(shared.llm_to_db_buffer[closest_buffer_key])
        del shared.llm_to_db_buffer[closest_buffer_key]
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


@trace
def to_anki(
        audio_mp3_1,
        txt_audio,
        txt_chatgpt_cloz,
        txt_chatgpt_context,
        txt_deck,
        txt_tags,
        gallery,
        check_marked,
        txt_extra_source,
        ):
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

    # checks clozes validity
    clozes = [c.strip() for c in txt_chatgpt_cloz.split("#####") if c.strip()]
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        raise Exception(red(f"Invalid cloze: '{txt_chatgpt_cloz}'"))

    if "alfred" in txt_chatgpt_cloz.lower():
        raise Exception(red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_cloz}'"))

    # load the source text of the image in the gallery
    txt_source_queue = queue.Queue()
    txt_source = ""
    if gallery is None or len(gallery) == 0:
        red("you should probably specify an image in source")
        txt_source = "<br>"
    else:
        thread = threading.Thread(
                target=get_img_source,
                args=(gallery, txt_source_queue)
                )
        thread.start()
        with shared.thread_lock:
            shared.running_threads["ocr"].append(thread)

    # send audio to anki
    audio_to_anki_queue = queue.Queue()
    thread = threading.Thread(
            target=audio_to_anki,
            args=(audio_mp3_1, audio_to_anki_queue),
            )
    thread.start()
    with shared.thread_lock:
        shared.running_threads["audio_to_anki"].append(thread)

    # send to anki
    metadata = rtoml.dumps(
            {
                "author": "Voice2FormattedText",
                "transcripted_text": txt_audio,
                "chatgpt_context": txt_chatgpt_context,
                "llm_used": shared.latest_llm_used,
                "tts_used": shared.latest_stt_used,
                "version": shared.VERSION,
                "timestamp": time.time(),
                }, pretty=True)
    results = []

    whi("Sending to anki:")

    # sending sound file to anki media
    audio_html = wait_for_queue(audio_to_anki_queue, "audio_to_anki")
    if "Error" in audio_html:  # then out is an error message and not the source
        gather_threads(["audio_to_anki", "ocr"])
        raise Exception(f"Error in audio_html: '{audio_html}'")

    # gather text from the source image(s)
    if not txt_source:
        txt_source = wait_for_queue(txt_source_queue, "txt_source") + audio_html
    else:
        txt_source += audio_html

    # anki tags
    new_tags = txt_tags.split(" ") + [f"WhisperToAnki::{today}"]
    if check_marked:
        new_tags += ["marked"]
    if "<img" not in txt_source:
        # if no image in source: add a tag to find them easily later on
        new_tags += ["WhisperToAnki::no_img_in_source"]

    if txt_extra_source.strip():
        txt_source += f"<br>{txt_extra_source}"

    with shared.dirload_lock:
        shared.dirload_queue.loc[shared.dirload_queue["temp_path"] == str(audio_mp3_1), "ankified"] = "started"

    for cl in clozes:
        cl = cl.strip()
        if "\n" in cl:
            cl = cl.replace("\n", "<br/>")
        try:
            res = add_to_anki(
                    body=cl,
                    source=txt_source,
                    note_metadata=metadata,
                    tags=new_tags,
                    deck_name=txt_deck)
        except Exception as err:
            red(f"Error when adding to anki: '{cl}': '{err}'")
            res = err
        results.append(res)
        whi(f"* {cl}")

    errors = [f"#{results.index(r)+1}/{len(results)}: {r}" for r in results if not str(r).isdigit()]
    results = [str(r) for r in results if str(r).isdigit()]
    shared.added_note_ids.append([int(r) for r in results])

    if not len(results) == len(clozes):
        gather_threads(["audio_to_anki", "ocr"])
        raise Exception(red(f"Some flashcards were not added:{','.join(errors)}"))
    with shared.dirload_lock:
        shared.dirload_queue.loc[shared.dirload_queue["temp_path"] == str(audio_mp3_1), "ankified"] = True

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    # add the latest generated cards to the message bugger
    if "\n\n" in txt_audio:
        audio_split = txt_audio.split("\n\n")
        audio_split = [a.strip() for a in audio_split if a.strip()]
        if len(audio_split) != len(clozes):
            red("No saving card to message buffer because the number of split in txt_audio is not the same as in clozes.")
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
        if txt_audio not in [mb["unformatted_txt_audio"] for mb in shared.message_buffer] and txt_chatgpt_cloz not in [mb["unformatted_txt_chatgpt_cloz"] for mb in shared.message_buffer]:
            shared.message_buffer.append(
                    {
                        "unformatted_txt_audio": txt_audio,
                        "unformatted_txt_chatgpt_cloz": txt_chatgpt_cloz,
                        "question": transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", txt_audio),
                        "answer": txt_chatgpt_cloz.replace("\n", "<br/>"),
                        "was_split": False,
                        }
                    )

    # cap the number of messages
    shared.message_buffer = shared.message_buffer[-shared.max_message_buffer:]
    pv["message_buffer"] = shared.message_buffer

    Voice2Anki_db_save(txt_chatgpt_cloz, txt_chatgpt_context, txt_audio)

    gather_threads(["audio_to_anki", "ocr", "saving_chatgpt"])
    return
