import re
import uuid
import Levenshtein as lev
import shutil
import queue
import threading
import hashlib
import base64
import joblib
import json
import cv2
from textwrap import dedent
import rtoml
import time
from datetime import datetime
import openai
from openai.error import RateLimitError
from pathlib import Path

from .anki_utils import add_to_anki, audio_to_anki
from .shared_module import shared
from .logger import red, whi, yel, store_to_db, trace, Timeout
from .memory import prompt_filter, load_prev_prompts, embedder, tokenize, transcript_template
from .media import sound_preprocessing, get_img_source, format_audio_component
from .profiles import ValueStorage

user_identifier = str(uuid.uuid4())

splitted_dir = Path("./user_directory/splitted")
done_dir = Path("./user_directory/done")
doing_dir = Path("./user_directory/doing")
unsplitted_dir = Path("./user_directory/unsplitted")
tmp_dir = Path("/tmp/gradio")


Path("user_directory").mkdir(exist_ok=True)
splitted_dir.mkdir(exist_ok=True)
unsplitted_dir.mkdir(exist_ok=True)
done_dir.mkdir(exist_ok=True)
doing_dir.mkdir(exist_ok=True)

# move any file in doing to todos
doings = sorted([p for p in doing_dir.rglob("*.mp3")])
for p in doings:
    whi(f"Starting up so moved files from doing to splitted: {p}")
    shutil.move(p, splitted_dir / p.name)

assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()

shared.pv = ValueStorage()
pv = shared.pv

message_buffer = pv["message_buffer"]

running_tasks = {
        "saving_chatgpt": [],
        "saving_whisper": [],
        "transcribing_audio": [],
        }

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

stt_cache = joblib.Memory("cache/transcript_cache", verbose=0)

@stt_cache.cache(ignore=["audio_path"])
def whisper_cached(
        audio_path,
        audio_hash,
        modelname,
        txt_whisp_prompt,
        txt_whisp_lang,
        sld_whisp_temp,
        ):
    """this is a call to openai's whisper. It's called as soon as the
    recording is done to begin caching. The audio_path can change so a hash
    of the content is used instead."""
    red(f"Calling whisper because not in cache: {audio_path}")
    assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
    try:
        cnt = 0
        while True:
            try:
                cnt += 1
                with open(audio_path, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                        model=modelname,
                        file=audio_file,
                        prompt=txt_whisp_prompt,
                        language=txt_whisp_lang,
                        temperature=sld_whisp_temp,
                        response_format="verbose_json",
                        )

                return transcript
            except RateLimitError as err:
                if cnt >= 5:
                    return red(f"Cached whisper: RateLimitError >5: '{err}'")
                else:
                    red(f"Cached whisper: RateLimitError #{cnt}/5 from cached whisper: '{err}'")
                    time.sleep(2 * cnt)
    except Exception as err:
        return red(f"Error when cache transcribing audio: '{err}'")

@trace
def transcribe_cache(audio_mp3, txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp):
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
    shared.latest_stt_used = modelname

    with open(audio_mp3, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    _ = whisper_cached(
            audio_mp3,
            audio_hash,
            modelname,
            txt_whisp_prompt,
            txt_whisp_lang,
            sld_whisp_temp,
            )
    return None


@trace
def transcribe_cache_async(audio_mp3, txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp):
    thread = threading.Thread(
            target=transcribe_cache,
            args=(audio_mp3, txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp)
            )
    thread.start()
    return thread


@trace
def transcribe(audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp):
    "turn the 1st audio track into text"
    whi("Transcribing audio")

    if audio_mp3_1 is None:
        return red("Error: None audio_mp3_1")

    if txt_whisp_prompt is None:
        return red("Error: None whisper prompt")

    if txt_whisp_lang is None:
        return red("Error: None whisper language")

    modelname = "whisper-1"

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
        txt_audio = transcript["text"]
        yel(f"\nWhisper transcript: {txt_audio}")

        if running_tasks["saving_whisper"]:
            running_tasks["saving_whisper"][-1].join()
        while running_tasks["saving_whisper"]:
            running_tasks["saving_whisper"].pop()
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
                        "V2FT_profile": pv.profile_name,
                        "transcribed_input": txt_audio,
                        "full_whisper_output": transcript,
                        "model_name": f"OpenAI {modelname}",
                        "audio_mp3": base64.b64encode(mp3_content).decode(),
                        "V2FT_version": shared.VERSION,
                        },
                    "db_name": "anki_whisper"
                    })
        thread.start()
        running_tasks["saving_whisper"].append(thread)

        return txt_audio
    except Exception as err:
        return red(f"Error when transcribing audio: '{err}'")


@trace
@Timeout(30)
def alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, check_gpt4, txt_keywords):
    "send the previous prompt and transcribed speech to the LLM"
    if not txt_audio:
        shared.latest_llm_cost = [0, 0]
        return "No transcribed audio found."
    if txt_audio.strip().startswith("Error "):
        shared.latest_llm_cost = [0, 0]
        return "Error when transcribing sound."
    if not txt_chatgpt_context:
        shared.latest_llm_cost = [0, 0]
        return "No txt_chatgpt_context found."

    if (("fred" in txt_audio.lower() and "image" in txt_audio.lower()) or ("change d'image" in txt_audio.lower())) and len(txt_audio) < 40:
        shared.latest_llm_cost = [0, 0]
        return f"Image change detected: '{txt_audio}'"

    if "," in txt_keywords:
        keywords = [re.compile(kw.strip(), flags=re.DOTALL|re.MULTILINE|re.IGNORECASE) for kw in txt_keywords.split(",")]
        keywords = [kw for kw in keywords if re.search(kw, txt_audio)]
    else:
        keywords = []

    prev_prompts = load_prev_prompts(profile)
    new_prompt = {
            "role": "user",
            "content": dedent(
                transcript_template.replace("CONTEXT", txt_chatgpt_context
                    ).replace("TRANSCRIPT", txt_audio))
                }

    # the last few transcript/answer pair is always saved in message_buffer
    # even if it will not be saved to memory.
    buffer_to_add = []
    if sld_buffer and message_buffer["question"]:
        whi(f"Length of message_buffer: {len(message_buffer['answer'])}")
        for i in range(min(len(message_buffer["question"]), sld_buffer), 0):
            if lev.ratio(
                    txt_audio.lower(),
                    message_buffer["question"][-i].lower(),
                    ) < 0.8 :
                buffer_to_add.extend(
                        [
                            {
                                "role": "user",
                                "content": message_buffer["question"][-i]
                                },
                            {
                                "role": "assistant",
                                "content": message_buffer["answer"][-i]
                                }
                            ]
                        )
                whi("Added message_buffer to the prompt.")
    else:
        whi("Ignored message buffer")

    prev_prompts = prompt_filter(
            prev_prompts,
            max_token,
            temperature,
            prompt_messages=buffer_to_add + [new_prompt],
            keywords=keywords,
            )

    # check the number of token is fine and format the previous
    # prompts in chatgpt format
    formatted_messages = []

    tkns = 0
    for m in prev_prompts:
        formatted_messages.append(
                {
                    "role": m["role"],
                    "content": m["content"],
                    }
                )
        tkns += m["tkn_len_in"]
        tkns += m["tkn_len_out"]
        if "answer" in m:
            assert m["role"] == "user", "expected user"
            formatted_messages.append({
                "role": "assistant",
                "content": m["answer"]})
    formatted_messages.extend(buffer_to_add)
    formatted_messages.append(new_prompt)

    tkns += len(tokenize(formatted_messages[-1]["content"]))
    yel(f"Number of messages that will be sent to ChatGPT: {len(formatted_messages)} (representing {tkns} tokens)")

    if tkns >= 15700:
        red("More than 15700 tokens before calling ChatGPT. Bypassing to ask "
            "with fewer tokens to make sure you have room for the answer")
        return alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature, sld_buffer, txt_keywords)

    if not check_gpt4:
        model_to_use = "gpt-3.5-turbo-1106"
        model_price = (0.001, 0.002)
    else:
        model_to_use = "gpt-4-1106-preview"
        model_price = (0.01, 0.03)
    whi(f"Will use model {model_to_use}")

    # in case recur improv is called
    shared.latest_llm_used = model_to_use

    # print prompts used for the call:
    n = len(formatted_messages)
    whi("ChatGPT prompt :")
    for i, p in enumerate(formatted_messages):
        whi(f"    * {i+1}/{n}: {p['role']:>10}: {str(p['content'])[:200]}")

    assert tkns <= 16000, f"Too many tokens! ({tkns})"
    try:
        cnt = 0
        while True:
            try:
                whi("Asking ChatGPT")
                cnt += 1
                response = openai.ChatCompletion.create(
                        model=model_to_use,
                        messages=formatted_messages,
                        stop="END",
                        temperature=temperature,
                        user=user_identifier,
                        )
                break
            except RateLimitError as err:
                if cnt >= 5:
                    shared.latest_llm_cost = [0, 0]
                    return red("ChatGPT: too many retries.")
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)


        input_tkn_cost = response["usage"]["prompt_tokens"]
        output_tkn_cost = response["usage"]["completion_tokens"]
        tkn_cost = [input_tkn_cost, output_tkn_cost]

        tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
        pv["total_llm_cost"] += tkn_cost_dol
        cloz = response["choices"][0]["message"]["content"]
        cloz = cloz.replace("<br/>", "\n")  # for cosmetic purposes in the textbox

        yel(f"\n###\nChatGPT answer:\n{cloz}\n###\n")
        red(f"Total ChatGPT cost so far: ${pv['total_llm_cost']:.4f} (not counting whisper)")

        reason = response["choices"][0]["finish_reason"]
        if reason.lower() != "stop":
            red(f"ChatGPT's reason to stop was not 'stop' but '{reason}'")

        # add to db to create LORA fine tunes later
        if running_tasks["saving_chatgpt"]:
            running_tasks["saving_chatgpt"][-1].join()
        while running_tasks["saving_chatgpt"]:
            running_tasks["saving_chatgpt"].pop()
        thread = threading.Thread(
                target=store_to_db,
                name="saving_chatgpt",
                kwargs={
                    "dictionnary": {
                        "type": "anki_card",
                        "timestamp": time.time(),
                        "token_cost": tkn_cost,
                        "temperature": temperature,
                        "LLM_context": txt_chatgpt_context,
                        "V2FT_profile": pv.profile_name,
                        "transcribed_input": txt_audio,
                        "model_name": model_to_use,
                        "last_message_from_conversation": formatted_messages[-1],
                        "nb_of_message_in_conversation": len(formatted_messages),
                        "system_prompt": formatted_messages[0],
                        "cloze": cloz,
                        "V2FT_version": shared.VERSION,
                        },
                    "db_name": "anki_llm"})
        thread.start()
        running_tasks["saving_whisper"].append(thread)

        shared.latest_llm_cost = tkn_cost
        return cloz
    except Exception as err:
        shared.latest_llm_cost = [0, 0]
        return red(f"Error with ChatGPT: '{err}'")


@trace
def dirload_splitted(checkbox, a1, a2, a3, a4, a5):
    """
    load the audio file that were splitted previously one by one in the
    available audio slots
    """
    if not checkbox:
        whi("Not running Dirload because checkbox is unchecked")
        return a1, a2, a3, a4, a5
    # check how many audio are needed
    sound_slots = 0
    for sound in [a5, a4, a3, a2, a1]:
        if sound is None:
            sound_slots += 1
        else:
            break
    whi(f"Number of empty sound slots: {sound_slots}")

    # move any file in doing to done
    doings = sorted([p for p in doing_dir.rglob("*.mp3")])
    if not doings:
        whi("No mp3 files in doing")
    else:
        for p in doings[:sound_slots]:
            yel(f"Refilling so moving files from doing to done: {p}")
            shutil.move(p, done_dir / p.name)

    # count the number of mp3 files in the splitted dir
    splitteds = [p for p in splitted_dir.rglob("*.mp3")]
    if not splitteds:
        red("Splitted subdir contains no mp3")
        return a1, a2, a3, a4, a5

    # sort by oldest
    #splitteds = sorted(splitteds, key=lambda x: x.stat().st_ctime)
    # sort by name
    splitteds = sorted(splitteds, key=lambda x: str(x))

    # iterate over each files from the dir. If images are found, load them
    # into gallery but if the images are found after sounds, stops iterating
    sounds_to_load = []
    new_threads = []
    for path in splitteds[:sound_slots]:
        moved = doing_dir / path.name
        shutil.move(path, moved)
        to_temp = tmp_dir / moved.name
        shutil.copy2(moved, to_temp)
        assert (moved.exists() and (to_temp).exists()) and (
                not path.exists()), "unexpected sound location"

        to_temp = sound_preprocessing(to_temp)

        whi(f"Will load sound {to_temp}")
        sounds_to_load.append(str(to_temp))

    whi(f"Loading {len(sounds_to_load)} sounds from splitted")
    filled_slots = [a1, a2, a3, a4, a5]
    output = filled_slots[:-len(sounds_to_load)] + sounds_to_load
    assert len(filled_slots) == len(output), f"invalid output length: {len(filled_slots)} vs {len(output)}"

    # auto roll over if leading None present
    while output[0] is None:
        output.append(output.pop(0))

    if new_threads:
        if len(sounds_to_load) == len(output):
            # loaded all the slots, so waiting for the first transcription to
            # finish
            gather_threads([new_threads[0]], "Transcribing first file")
            running_tasks["transcribing_audio"].extend(new_threads[1:])
        else:
            # the sound in the first slot was not loaded by this function so
            # not waiting for the transcription
            running_tasks["transcribing_audio"].extend(new_threads)

    return output

@trace
def dirload_splitted_last(checkbox):
    """wrapper for dirload_splitted to only load the last slot. This is faster
    because gradio does not have to send all 5 sounds if I just rolled"""
    if not checkbox:
        return None
    out = dirload_splitted(
            checkbox,
            a1=True,
            a2=True,
            a3=True,
            a4=True,
            a5=None,
            )
    return out[-1]

@trace
def gather_threads(threads, source="to_anki"):
    n = len([t for t in threads if t.is_alive()])
    i = 0
    while n:
        i += 1
        n = len([t for t in threads if t.is_alive()])
        if i % 10 == 0:
            yel(f"Waiting for {n} threads to finish from {source}")
        time.sleep(0.1)

@trace
def wait_for_queue(q, source, t=1):
    "source : https://stackoverflow.com/questions/19206130/does-queue-get-block-main"
    start = time.time()
    while True:
        # try:
        #     data = q.get(False)
        #     # If `False`, the program is not blocked. `Queue.Empty` is thrown if
        #     # the queue is empty
        # except queue.Queue.Empty:
        #     data = None

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
    with threading.Lock():
        n = sum([t.is_alive() for t in shared.threads])
        red(f"Killing {n} running threads")
        shared.threads = []

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
        ):
    "function called to do wrap it up and send to anki"
    whi("Entering to_anki")
    if not txt_audio:
        red("missing txt_audio")
        return
    if not txt_chatgpt_cloz:
        red("missing txt_chatgpt_cloz")
        return
    if not txt_deck:
        red("missing txt_deck")
        return
    if not txt_tags:
        red("missing txt_tags")
        return

    # check that the tkn_cost is sound
    if isinstance(shared.latest_llm_cost, str):
        shared.latest_llm_cost = [int(x) for x in json.loads(shared.latest_llm_cost)]
    if not shared.latest_llm_cost:
        red("No token cost found, setting to 0")
        shared.latest_llm_cost = [0, 0]
    if txt_chatgpt_cloz.startswith("Error with ChatGPT") or 0 in shared.latest_llm_cost:
        red(f"Error with chatgpt: '{txt_chatgpt_cloz}'")
        return

    # checks clozes validity
    clozes = txt_chatgpt_cloz.split("#####")
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        red(f"Invalid cloze: '{txt_chatgpt_cloz}'")
        return

    if "alfred" in txt_chatgpt_cloz.lower():
        red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_cloz}'"),
        return

    threads = []

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
        threads.append(thread)

    # send audio to anki
    audio_to_anki_queue = queue.Queue()
    thread = threading.Thread(
            target=audio_to_anki,
            args=(audio_mp3_1, audio_to_anki_queue),
            )
    thread.start()
    threads.append(thread)

    tkn_cost_dol = int(shared.latest_llm_cost[0]) / 1000 * 0.003 + int(shared.latest_llm_cost[1]) / 1000 * 0.004

    # add cloze to output
    whi(f"ChatGPT cost: {shared.latest_llm_cost} (${tkn_cost_dol:.3f}, not counting whisper)")
    whi(f"ChatGPT answer:\n{txt_chatgpt_cloz}")

    # send to anki
    metadata = rtoml.dumps(
            {
                "author": "Voice2FormattedText",
                "transcripted_text": txt_audio,
                "chatgpt_context": txt_chatgpt_context,
                "llm_used": shared.latest_llm_used,
                "tts_used": shared.latest_stt_used,
                "version": shared.VERSION,
                "chatgpt_tkn_cost": shared.latest_llm_cost,
                "chatgpt_dollars_cost": tkn_cost_dol,
                "timestamp": time.time(),
                }, pretty=True)
    results = []

    whi("Sending to anki:")

    # sending sound file to anki media
    audio_html = wait_for_queue(audio_to_anki_queue, "audio_to_anki")
    if "Error" in audio_html:  # then out is an error message and not the source
        gather_threads(threads)
        return

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

    for cl in clozes:
        cl = cl.strip()
        if "\n" in cl:
            whi("Replaced newlines in clozes")
            cl = cl.replace("\n", "<br/>")
        results.append(
                add_to_anki(
                    body=cl,
                    source=txt_source,
                    note_metadata=metadata,
                    tags=new_tags,
                    deck_name=txt_deck,
                )
                )
        whi(f"* {cl}")

    results = [str(r) for r in results if str(r).isdigit()]

    if not len(results) == len(clozes):
        red("Some flashcards were not added!"),
        gather_threads(threads)
        return

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    # add the latest generated cards to the message bugger
    message_buffer["question"].append(txt_audio)
    message_buffer["answer"].append(txt_chatgpt_cloz)

    # cap the number of messages
    message_buffer["question"] = message_buffer["question"][-shared.max_message_buffer:]
    message_buffer["answer"] = message_buffer["answer"][-shared.max_message_buffer:]

    gather_threads(threads)
    return
