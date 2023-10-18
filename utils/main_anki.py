import uuid
import shutil
import queue
import threading
import pickle
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

from .anki_utils import add_to_anki, audio_to_anki, sync_anki
from .misc import tokenize, transcript_template, backend_config, format_audio_component
from .logger import red, whi, yel, store_to_db
from .memory import prompt_filter, load_prev_prompts
from .media import sound_preprocessing, get_img_source
from .profiles import ValueStorage

user_identifier = str(uuid.uuid4())

splitted_dir = Path("./user_directory/splitted")
done_dir = Path("./user_directory/done")
doing_dir = Path("./user_directory/doing")
unsplitted_dir = Path("./user_directory/unsplitted")
tmp_dir = Path("/tmp/gradio")

assert Path("user_directory").exists(), "No 'user_directory' found"
assert splitted_dir.exists(), "No 'splitted' subdir found"
assert unsplitted_dir.exists(), "No 'unsplitted' subdir found"
assert done_dir.exists(), "No 'done' subdir found"
assert doing_dir.exists(), "No 'doing' subdir found"

# move any file in doing to todos
doings = [p for p in doing_dir.rglob("*.mp3")]
for p in doings:
    whi(f"Starting up so moved files from doing to splitted: {p}")
    shutil.move(p, splitted_dir / p.name)

assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()

global pv
assert Path("profiles/anki/latest_profile.pickle").exists(), "latest_profile not found, it should be created by gui_anki before!"
whi("Reloading previous profile.")
with open("profiles/anki/latest_profile.pickle", "rb") as f:
    pv = ValueStorage(pickle.load(f))

message_buffer = {"question": "", "answer": ""}

running_tasks = {
        "saving_chatgpt": [],
        "saving_whisper": [],
        "transcribing_audio": [],
        }

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

stt_cache = joblib.Memory("transcript_cache", verbose=0)
soundpreprocess_cache = joblib.Memory("sound_preprocessing_cache", verbose=0)

def _whisper_cached(audio_path, audio_hash, modelname, txt_whisp_prompt, txt_whisp_lang):
    """this is a call to openai's whisper. It's called as soon as the
    recording is done to begin caching. The audio_path can change so a hash
    of the content is used instead."""
    red(f"Calling whisper instead of using cache for {audio_path}")
    with open(audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            model=modelname,
            file=audio_file,
            prompt=txt_whisp_prompt,
            language=txt_whisp_lang)
    return transcript
whisper_cached = stt_cache.cache(
        func=_whisper_cached,
        ignore=["audio_path"],
        )

sound_preprocessing_cached = soundpreprocess_cache.cache(sound_preprocessing)


def transcribe_cache(audio_mp3, txt_whisp_prompt, txt_whisp_lang):
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

    # try to remove silences
    # try:
    #     audio_mp3 = sound_preprocessing_cached(audio_mp3)
    # except Exception as err:
    #     red(f"Error when preprocessing sound: '{err}'")

    with open(audio_mp3, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
        cnt = 0
        while True:
            try:
                cnt += 1
                transcript = whisper_cached(
                        audio_mp3,
                        audio_hash,
                        modelname,
                        txt_whisp_prompt,
                        txt_whisp_lang)
                return None
            except RateLimitError as err:
                if cnt >= 5:
                    Path(audio_mp3).unlink(missing_ok=False)
                    red("Cached whisper: too many retries.")
                    return
                red(f"Error from cached whisper: '{err}'")
                time.sleep(2 * cnt)
    except Exception as err:
        return red(f"Error when cache transcribing audio: '{err}'")

def transcribe_cache_async(audio_mp3, txt_whisp_prompt, txt_whisp_lang):
    thread = threading.Thread(
            target=transcribe_cache,
            args=(audio_mp3, txt_whisp_prompt, txt_whisp_lang)
            )
    thread.start()
    return thread


def transcribe(audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile):
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

    # try:  # preprocess sound, cached to make sure it only run once
    #     audio_mp3_1 = sound_preprocessing_cached(audio_mp3_1)
    # except Exception as err:
    #     red(f"Error when preprocessing sound: '{err}'")

    with open(audio_mp3_1, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    try:
        assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
        cnt = 0
        while True:
            try:
                whi(f"Asking Whisper for {audio_mp3_1} using cache")
                cnt += 1
                transcript = whisper_cached(
                        audio_mp3_1,
                        audio_hash,
                        modelname,
                        txt_whisp_prompt,
                        txt_whisp_lang)
                with open(audio_mp3_1, "rb") as audio_file:
                    mp3_content = audio_file.read()
                txt_audio = transcript["text"]
                yel(f"\nWhisper transcript: {txt_audio}")
                Path(audio_mp3_1).unlink(missing_ok=False)

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
                                "V2FT_profile": txt_profile,
                                "transcribed_input": txt_audio,
                                "model_name": f"OpenAI {modelname}",
                                "audio_mp3": base64.b64encode(mp3_content).decode(),
                                "V2FT_version": backend_config.VERSION,
                                },
                            "db_name": "anki_whisper"
                            })
                thread.start()
                running_tasks["saving_whisper"].append(thread)

                return txt_audio
            except RateLimitError as err:
                if cnt >= 5:
                    Path(audio_mp3_1).unlink(missing_ok=False)
                    return red("Whisper: too many retries.")
                red(f"Error from whisper: '{err}'")
                time.sleep(2 * cnt)
    except Exception as err:
        Path(audio_mp3_1).unlink(missing_ok=False)
        return red(f"Error when transcribing audio: '{err}'")


def alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, mode="one"):
    "send the previous prompt and transcribed speech to the LLM"
    if not txt_audio:
        return "No transcribed audio found.", [0, 0]
    if not txt_chatgpt_context:
        return "No txt_chatgpt_context found.", [0, 0]

    prev_prompts = load_prev_prompts(profile)
    new_prompt = {
            "role": "user",
            "content": dedent(
                transcript_template.replace("CONTEXT", txt_chatgpt_context
                    ).replace("TRANSCRIPT", txt_audio))
                }

    # the last transcript/answer pair is always saved in message_buffer
    # even if it will not be saved to memory.
    buffer_to_add = []
    if message_buffer["question"] and message_buffer["answer"]:
        if txt_audio.lower() not in message_buffer["question"].lower() and message_buffer["question"].lower() not in txt_audio:
            buffer_to_add = [
                    {
                        "role": "user",
                        "content": message_buffer["question"]
                        },
                    {
                        "role": "assistant",
                        "content": message_buffer["answer"]
                        }
                    ]
            whi("Added message_buffer to the prompt.")
        else:
            message_buffer["question"] = ""
            message_buffer["answer"] = ""

    prompt_len_already = len(tokenize(new_prompt["content"]))
    for p in buffer_to_add:
        prompt_len_already += len(tokenize(p["content"]))
    prev_prompts = prompt_filter(
            prev_prompts,
            max_token,
            temperature,
            new_prompt_len=prompt_len_already,
            favor_list=True if " list" in txt_audio.lower() else False
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
        return alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature, mode)

    if tkns >= 3700:
        red(f"More than 3700 token in question, using ChatGPT 16k")
        model_to_use = "gpt-3.5-turbo-16k"
        model_price = (0.003, 0.004)
    else:
        model_to_use = "gpt-3.5-turbo"
        model_price = (0.0015, 0.002)

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
                    return red("ChatGPT: too many retries."), [0, 0]
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)


        input_tkn_cost = response["usage"]["prompt_tokens"]
        output_tkn_cost = response["usage"]["completion_tokens"]
        tkn_cost = [input_tkn_cost, output_tkn_cost]

        tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
        global pv
        if pv.profile_name != profile:
            pv = ValueStorage(profile)
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
                        "V2FT_profile": profile,
                        "transcribed_input": txt_audio,
                        "model_name": model_to_use,
                        "last_message_from_conversation": formatted_messages[-1],
                        "nb_of_message_in_conversation": len(formatted_messages),
                        "system_prompt": formatted_messages[0],
                        "cloze": cloz,
                        "V2FT_mode": mode,
                        "V2FT_version": backend_config.VERSION,
                        },
                    "db_name": "anki_llm"})
        thread.start()
        running_tasks["saving_whisper"].append(thread)

        return cloz, tkn_cost
    except Exception as err:
        message_buffer["question"] = ""
        message_buffer["answer"] = ""
        return red(f"Error with ChatGPT: '{err}'"), [0, 0]


def load_splitted_audio(a1, a2, a3, a4, a5, txt_whisp_prompt, txt_whisp_lang):
    """
    load the audio file that were splitted previously one by one in the
    available audio slots
    """
    # move any file in doing to done
    doings = [p for p in doing_dir.rglob("*.mp3")]
    if not doings:
        whi("No mp3 files in doing")
    for p in doings:
        whi(f"Refilling so moving files from doing to done: {p}")
        shutil.move(p, done_dir / p.name)

    # check how many audio are needed
    sound_slots = 0
    for sound in [a5, a4, a3, a2, a1]:
        if sound is None:
            sound_slots += 1
        else:
            break
    whi(f"Number of empty sound slots: {sound_slots}")

    # count the number of mp3 files in the splitted dir
    splitteds = [p for p in splitted_dir.rglob("*.mp3")]
    if not splitteds:
        red("splitted subdir contains no mp3")
        return a1, a2, a3, a4, a5

    # sort by oldest
    #splitteds = sorted(splitteds, key=lambda x: x.stat().st_ctime)
    # sort by name
    splitteds = sorted(splitteds, key=lambda x: str(x))

    # iterate over each files from the dir. If images are found, load them
    # into gallery but if the images are found after sounds, stops iterating
    sounds_to_load = []
    for path in splitteds[:sound_slots]:
        moved = doing_dir / path.name
        shutil.move(path, moved)
        to_temp = tmp_dir / moved.name
        shutil.copy2(moved, to_temp)
        assert (moved.exists() and (to_temp).exists()) and (
                not path.exists()), "unexpected sound location"
        sounds_to_load.append(to_temp)
        if txt_whisp_prompt and txt_whisp_lang:
            running_tasks["transcribing_audio"].append(transcribe_cache_async(to_temp, txt_whisp_prompt, txt_whisp_lang))

    whi(f"Loading {len(sounds_to_load)} sounds from splitted")
    filled_slots = [a1, a2, a3, a4, a5]
    output = filled_slots[:-len(sounds_to_load)] + sounds_to_load
    assert len(filled_slots) == len(output), f"invalid output length: {len(filled_slots)} vs {len(output)}"

    # auto roll over if leading None present
    while output[0] is None:
        output.append(output.pop(0))

    gather_threads(running_tasks["transcribing_audio"], source="transcribing audio")
    return output

def gather_threads(threads, source="main"):
    n = len([t for t in threads if t.is_alive()])
    while n:
        n = len([t for t in threads if t.is_alive()])
        time.sleep(0.5)
        yel(f"Waiting for {n} threads to finish in {source}")


def semiauto_mode(*args, **kwargs):
    "triggering function 'main' with mode 'semiauto'"
    whi("Triggering semiauto mode: doing everything but stop just before uploading to anki")
    return main(*args, **kwargs, mode="semiauto")


def auto_mode(*args, **kwargs):
    "triggering function 'main' with mode 'auto'"
    whi("Triggering auto mode")
    return main(*args, **kwargs, mode="auto")


def main(
        audio_mp3_1,
        txt_audio,
        txt_whisp_prompt,
        txt_whisp_lang,

        txt_chatgpt_tkncost,
        txt_chatgpt_cloz,

        txt_chatgpt_context,
        txt_deck,
        txt_tags,

        gallery,
        profile,
        sld_max_tkn,
        sld_temp,
        mode="one",
        ):
    "function called to do sequential actions: from audio to anki flashcard"
    whi("Entering main")
    if not (audio_mp3_1):
        return [
                red("None audio in microphone #1"),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]
    if not txt_whisp_prompt:
        return [
                red("No whisper prompt found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]
    if not txt_whisp_lang:
        return [
                red("No whisper language found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]
    if not txt_chatgpt_context:
        return [
                red("No txt_chatgpt_context found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]
    if not txt_deck:
        return [
                red("you should specify a deck"),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]
    if not txt_tags:
        return [
                red("you should specify tags"),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ]

    threads = []

    # to_return allows to keep track of what to output to which widget
    to_return = {}
    to_return["txt_audio"] = txt_audio
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz

    # store the default profile
    global pv
    if pv.profile_name != profile:
        pv = ValueStorage(profile)

    if gallery is None or len(gallery) == 0:
        red("you should probably specify an image in source")
        txt_source = "<br>"
    else:
        txt_source = get_img_source(gallery)

    # save state for next start
    pv["txt_deck"] = txt_deck
    pv["txt_tags"] = txt_tags
    pv["sld_max_tkn"] = sld_max_tkn
    pv["temperature"] = sld_temp
    pv["txt_chatgpt_context"] = txt_chatgpt_context
    pv["txt_whisp_prompt"] = txt_whisp_prompt
    pv["txt_whisp_lang"] = txt_whisp_lang

    # save image and audio for next startup
    if gallery is not None:
        saved_gallery = [
                    cv2.imread(
                        i["name"]
                        ) for i in gallery
                    ]
        pv["gallery"] = saved_gallery

    # get text from audio if not already parsed
    if (not txt_audio) or mode in ["auto", "semiauto"]:
        txt_audio = transcribe(audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, profile)
        to_return["txt_audio"] = txt_audio

    if mode != "semiauto":  # start copying the sound file right away
        audio_to_anki_queue = queue.Queue()
        thread = threading.Thread(
                target=audio_to_anki,
                args=(audio_mp3_1, audio_to_anki_queue),
                )
        thread.start()
        threads.append(thread)

    # ask chatgpt
    if (not txt_chatgpt_cloz) or mode in ["auto", "semiauto"]:
        txt_chatgpt_cloz, txt_chatgpt_tkncost = alfred(
                txt_audio,
                txt_chatgpt_context,
                profile,
                sld_max_tkn,
                sld_temp,
                mode)
    if isinstance(txt_chatgpt_tkncost, str):
        txt_chatgpt_tkncost = [int(x) for x in json.loads(txt_chatgpt_tkncost)]
    if not txt_chatgpt_tkncost:
        red("No token cost found, setting to 0")
        txt_chatgpt_tkncost = [0, 0]
    if txt_chatgpt_cloz.startswith("Error with ChatGPT") or 0 in txt_chatgpt_tkncost:
        return [
                to_return["txt_audio"],
                txt_chatgpt_tkncost,
                to_return["txt_chatgpt_cloz"],
                ]
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost

    tkn_cost_dol = int(txt_chatgpt_tkncost[0]) / 1000 * 0.003 + int(txt_chatgpt_tkncost[1]) / 1000 * 0.004

    # checks clozes validity
    clozes = txt_chatgpt_cloz.split("#####")
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        red(f"Invalid cloze: '{txt_chatgpt_cloz}'")
        gather_threads(threads)
        return [
                to_return["txt_audio"],
                txt_chatgpt_tkncost,
                to_return["txt_chatgpt_cloz"],
                ]

    if "alfred" in txt_chatgpt_cloz.lower():
        red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_cloz}'"),
        gather_threads(threads)
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    # add cloze to output
    whi(f"ChatGPT cost: {txt_chatgpt_tkncost} (${tkn_cost_dol:.3f}, not counting whisper)")
    whi(f"ChatGPT answer:\n{txt_chatgpt_cloz}")

    # send to anki
    metadata = rtoml.dumps(
            {
                "author": "Voice2FormattedText",
                "transcripted_text": txt_audio,
                "chatgpt_context": txt_chatgpt_context,
                "model": "chatgpt",
                "version": backend_config.VERSION,
                "chatgpt_tkn_cost": txt_chatgpt_tkncost,
                "chatgpt_dollars_cost": tkn_cost_dol,
                "timestamp": time.time(),
                }, pretty=True)
    results = []

    if mode == "semiauto":
        yel("Semiauto mode: stopping just before uploading to anki")
        gather_threads(threads)
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    whi("Sending to anki:")

    # sending sound file to anki media
    audio_html = audio_to_anki_queue.get()
    if "Error" in audio_html:  # then out is an error message and not the source
        gather_threads(threads)
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]
    txt_source += audio_html

    # anki tags
    new_tags = txt_tags.split(" ") + [f"WhisperToAnki::{today}"]
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

    # trigger anki sync
    thread = threading.Thread(target=sync_anki)
    thread.start()
    threads.append(thread)
    whi("Synchronized anki\n")


    if not len(results) == len(clozes):
        red(f"Some flashcards were not added!"),
        gather_threads(threads)
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    message_buffer["question"] = txt_audio
    message_buffer["answer"] = txt_chatgpt_cloz

    gather_threads(threads)
    return [
            to_return["txt_audio"],
            to_return["txt_chatgpt_tkncost"],
            to_return["txt_chatgpt_cloz"],
            ]
