from tqdm import tqdm
import gradio as gr
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
from textwrap import dedent, indent
import rtoml
import time
from datetime import datetime
import openai
from openai.error import RateLimitError
from pathlib import Path

from .anki_utils import add_to_anki, audio_to_anki
from .shared_module import shared
from .logger import red, whi, yel, store_to_db, trace, Timeout
from .memory import prompt_filter, load_prev_prompts, tokenize, transcript_template, default_system_prompt_anki
from .media import sound_preprocessing, get_img_source, format_audio_component
from .profiles import ValueStorage

user_identifier = str(uuid.uuid4())

splitted_dir = Path("./user_directory/splitted")
done_dir = Path("./user_directory/done")
unsplitted_dir = Path("./user_directory/unsplitted")
tmp_dir = Path("/tmp/gradio")


Path("user_directory").mkdir(exist_ok=True)
splitted_dir.mkdir(exist_ok=True)
unsplitted_dir.mkdir(exist_ok=True)
done_dir.mkdir(exist_ok=True)

assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()

shared.pv = ValueStorage()
pv = shared.pv
shared.message_buffer = pv["message_buffer"]

d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

stt_cache = joblib.Memory("cache/transcript_cache", verbose=0)
llm_cache = joblib.Memory("cache/llm_cache", verbose=0)
# llm_cache.clear()  # reset the llm cache to make sure shared.llm_to_db_buffer is up to date

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
        check_gpt4,
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

    with open(audio_mp3, "rb") as f:
        audio_hash = hashlib.sha256(f.read()).hexdigest()

    transcript = whisper_cached(
            audio_mp3,
            audio_hash,
            modelname,
            txt_whisp_prompt,
            txt_whisp_lang,
            sld_whisp_temp,
            )
    txt_audio = transcript["text"]
    _ = alfred(txt_audio, txt_chatgpt_context, txt_profile, max_token, temperature, sld_buffer, check_gpt4, txt_keywords, cache_mode=True)

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
        txt_audio = transcript["text"]
        yel(f"\nWhisper transcript: {txt_audio}")

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
        shared.running_threads["saving_whisper"].append(thread)

        return txt_audio
    except Exception as err:
        raise Exception(red(f"Error when transcribing audio: '{err}'"))

@trace
def pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, check_gpt4, txt_keywords, cache_mode):
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
            if len(buffer_to_add) > sld_buffer:
                break
            if txt_audio in [mb["unformatted_txt_audio"] for mb in shared.message_buffer]:
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

    # add all prompts together to format the messages list
    formatted_messages = []
    # add system prompt
    formatted_messages.insert(
            0,
            {"role": "system", "content": default_system_prompt_anki["content"]}
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
        return pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature, sld_buffer, check_gpt4, txt_keywords, cache_mode)

    assert tkns <= 15700, f"Too many tokens: {tkns}"

    # print prompts used for the call:
    n = len(formatted_messages)
    whi("ChatGPT prompt:")
    for i, p in enumerate(formatted_messages):
        whi(f"* {i+1}/{n}: {p['role']}")
        whi(indent(p['content'], " " * 5))

    return formatted_messages

@trace
@Timeout(30)
@llm_cache.cache(ignore=["cache_mode"])
def alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, check_gpt4, txt_keywords, cache_mode=False):
    "send the previous prompt and transcribed speech to the LLM"
    red(f"Calling Alfred in cache_mode={cache_mode} for transcript '{txt_audio}'")
    if not txt_audio:
        raise Exception(red("No transcribed audio found."))
    if txt_audio.strip().startswith("Error"):
        raise Exception(red("Error when transcribing sound."))
    if not txt_chatgpt_context:
        raise Exception(red("No txt_chatgpt_context found."))
    if (("fred" in txt_audio.lower() and "image" in txt_audio.lower()) or ("change d'image" in txt_audio.lower())) and len(txt_audio) < 40:
        if cache_mode:
            red(f"Image change detected: '{txt_audio}'")
            return
        else:
            gr.Error(red(f"Image change detected: '{txt_audio}'"))
            return

    formatted_messages = pre_alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature, sld_buffer, check_gpt4, txt_keywords, cache_mode)
    for i, fm in enumerate(formatted_messages):
        if i == 0:
            assert fm["role"] == "system"
        elif i % 2 == 0:
            assert fm["role"] == "assistant"
        elif i % 2 == 1:
            assert fm["role"] == "user"

    if not check_gpt4:
        model_to_use = "gpt-3.5-turbo-1106"
        model_price = (0.001, 0.002)
    else:
        model_to_use = "gpt-4-1106-preview"
        model_price = (0.01, 0.03)
    whi(f"Will use model {model_to_use}")

    # in case recur improv is called
    if model_to_use != shared.latest_llm_used:
        with threading.Lock():
            shared.latest_llm_used = model_to_use

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
                raise Exception(red("ChatGPT: too many retries."))
            red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
            time.sleep(2 * cnt)

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]
    tkn_cost = [input_tkn_cost, output_tkn_cost]

    tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
    pv["total_llm_cost"] += tkn_cost_dol

    cloz = response["choices"][0]["message"]["content"]
    # for cosmetic purposes in the textbox
    cloz = cloz.replace("<br/>", "\n")
    cloz = cloz.replace("&nbsp;", " ")
    cloz = cloz.replace("#####", "\n#####\n")  # make sure to separate cleanly the clozes
    cloz = "\n".join([cl.strip() for cl in cloz.splitlines() if cl.strip()])

    yel(f"\n###\nChatGPT answer:\n{cloz}\n###\n")
    red(f"Total ChatGPT cost so far: ${pv['total_llm_cost']:.4f} (not counting whisper)")

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        red(f"ChatGPT's reason to stop was not 'stop' but '{reason}'")

    # add to the shared module the infonrmation of this card creation.
    # if a card is created then this will be added to the db to
    # create LORA fine tunes later on.
    with threading.Lock():
        shared.llm_to_db_buffer[cloz] = json.dumps(
                {
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
                    "system_prompt": default_system_prompt_anki["content"],
                    "cloze": cloz,
                    "V2FT_version": shared.VERSION,
                    })

    whi(f"ChatGPT cost: {pv['total_llm_cost']} (${tkn_cost_dol:.3f}, not counting whisper)")
    whi(f"ChatGPT answer:\n{cloz}")

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
        check_gpt4,
        txt_keywords,

        *audios,
        prog=gr.Progress()):
    """
    load the audio file that were splitted previously one by one in the
    available audio slots
    """
    if not hasattr(shared, "prog_total"):
        shared.prog_total = len(list(Path("user_directory/splitted").rglob("*mp3")))
    pbar = prog.tqdm([True] * shared.prog_total, desc="MP3s")

    if not checkbox:
        whi("Not running Dirload because checkbox is unchecked")
        return audios

    # make sure to move any empty slot at the end
    audios = [a for a in audios if a is not None]
    empty_slots = shared.audio_slot_nb - len(audios)
    while len(audios) < shared.audio_slot_nb:
        audios += [None]

    if empty_slots == shared.audio_slot_nb:
        assert not shared.dirload_queue, "Non empty queue of shared!"
        assert not shared.dirload_doing, "Non empty doing of shared!"

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
    # shared.dirload_queue = sorted([p for p in splitted_dir.rglob("*.mp3")], key=lambda x: x.stat().st_ctime)
    # sort by name
    shared.dirload_queue = sorted([p for p in splitted_dir.rglob("*.mp3")], key=lambda x: str(x))

    # remove the doing from the queue
    for doing in shared.dirload_doing:
        assert doing == shared.dirload_queue[0]
        shared.dirload_queue.pop(0)
        assert doing not in shared.dirload_queue

    if not shared.dirload_queue:
        gr.Error(red("No mp3 files in shared.dirload_queue"))
        return audios

    # iterate over each files from the dir. If images are found, load them
    # into gallery but if the images are found after sounds, stops iterating
    sounds_to_load = []
    new_threads = []
    for path in shared.dirload_queue[:empty_slots]:
        pbar.update(1)
        to_temp = tmp_dir / path.name
        shutil.copy2(path, to_temp)
        assert (path.exists() and (to_temp).exists()), "unexpected sound location"

        to_temp = sound_preprocessing(to_temp)

        whi(f"Will load sound {to_temp}")
        sounds_to_load.append(to_temp)
        shared.dirload_doing.append(path)
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
                        check_gpt4,
                        txt_keywords,
                        ),
                )
            thread.start()
            new_threads.append(thread)

    whi(f"Loading {len(sounds_to_load)} sounds from splitted")
    output = audios[:-len(sounds_to_load)] + sounds_to_load

    if new_threads:
        if len(sounds_to_load) == len(output):
            # loaded all the slots, so waiting for the first transcription to
            # finish
            whi("Waiting for first transcription to finish")
            new_threads[0].join()
            whi("Finished first transcription.")
            shared.running_threads["transcribing_audio"].extend(new_threads[1:])
        else:
            # the sound in the first slot was not loaded by this function so
            # not waiting for the transcription
            shared.running_threads["transcribing_audio"].extend(new_threads)

    while len(shared.dirload_doing) > shared.audio_slot_nb:
        p = shared.dirload_doing.pop(0)
        red(f"Moving {p} to done_dir")
        shutil.move(p, done_dir / p.name)

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
        check_gpt4,
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
            check_gpt4,
            txt_keywords,

            *audios,
            )[-1]

@trace
def audio_edit(audio, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context):
    """function called by a microphone. It will use whisper to transcribe
    your voice. Then use the instructions in your voice to modify the
    output from chatgpt."""

    instructions = transcribe(
            audio,
            txt_whisp_prompt="Instruction: ",
            txt_whisp_lang=txt_whisp_lang,
            sld_whisp_temp=0,
            )

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
    """).replace("OTHER_SYSPROMPT", default_system_prompt_anki["content"])
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

    # model_to_use = "gpt-3.5-turbo-1106"
    # model_price = (0.001, 0.002)
    model_to_use = "gpt-4-1106-preview"
    model_price = (0.01, 0.03)
    # whi(f"Will use model {model_to_use}")

    whi("Editing via ChatGPT:")
    whi(prompt)
    response = openai.ChatCompletion.create(
            model=model_to_use,
            messages=messages,
            stop="END",
            temperature=0,
            user=user_identifier,
            )

    input_tkn_cost = response["usage"]["prompt_tokens"]
    output_tkn_cost = response["usage"]["completion_tokens"]

    tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
    pv["total_llm_cost"] += tkn_cost_dol
    cloz = response["choices"][0]["message"]["content"]
    cloz = cloz.replace("<br/>", "\n").strip()  # for cosmetic purposes in the textbox

    yel(f"\n###\nChatGPT answer:\n{cloz}\n###\n")
    red(f"Total ChatGPT cost so far: ${pv['total_llm_cost']:.4f} (not counting whisper)")

    reason = response["choices"][0]["finish_reason"]
    if reason.lower() != "stop":
        red(f"ChatGPT's reason to stop was not 'stop' but '{reason}'")

    return cloz, None

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
        for k in shared.running_threads:
            n = sum([t.is_alive() for t in shared.running_threads[k]])
            red(f"Killing the {n} alive threads of {k}")
            shared.running_threads[k] = []

@trace
def v2ft_db_save(txt_chatgpt_cloz, txt_chatgpt_context, txt_audio):
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
                "V2FT_profile": shared.pv.profile_name,
                "transcribed_input": txt_audio,
                "model_name": f"Probably:{shared.latest_llm_used}",
                "last_message_from_conversation": None,
                "nb_of_message_in_conversation": None,
                "system_prompt": default_system_prompt_anki["content"],
                "cloze": txt_chatgpt_cloz,
                "V2FT_version": shared.VERSION,
                }
    else:
        save_dict = json.loads(shared.llm_to_db_buffer[closest_buffer_key])
        del shared.llm_to_db_buffer[closest_buffer_key]
    if shared.running_threads["saving_chatgpt"]:
        [t.join() for t in shared.running_threads["saving_chatgpt"]]
    while shared.running_threads["saving_chatgpt"]:
        shared.running_threads["saving_chatgpt"].pop()
    thread = threading.Thread(
            target=store_to_db,
            name="saving_chatgpt",
            kwargs={
                "dictionnary": save_dict,
                "db_name": "anki_llm"})
    thread.start()
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
    clozes = txt_chatgpt_cloz.split("#####")
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        red(f"Invalid cloze: '{txt_chatgpt_cloz}'")
        return

    if "alfred" in txt_chatgpt_cloz.lower():
        red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_cloz}'"),
        return

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
        shared.running_threads["ocr"].append(thread)

    # send audio to anki
    audio_to_anki_queue = queue.Queue()
    thread = threading.Thread(
            target=audio_to_anki,
            args=(audio_mp3_1, audio_to_anki_queue),
            )
    thread.start()
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
        gr.Error(f"Error in audio_html: '{audio_html}'")
        gather_threads(["audio_to_anki", "ocr"])
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

    if txt_extra_source.strip():
        txt_source += f"<br>{txt_extra_source}"

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
    shared.added_note_ids.append([int(r) for r in results])

    if not len(results) == len(clozes):
        red("Some flashcards were not added!"),
        gather_threads(["audio_to_anki", "ocr"])
        return

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    # add the latest generated cards to the message bugger
    if txt_audio not in [mb["unformatted_txt_audio"] for mb in shared.message_buffer] and txt_chatgpt_cloz not in [mb["unformatted_txt_chatgpt_cloz"] for mb in shared.message_buffer]:
        shared.message_buffer.append(
                {
                    "unformatted_txt_audio": txt_audio,
                    "unformatted_txt_chatgpt_cloz": txt_chatgpt_cloz,
                    "question": transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", txt_audio),
                    "answer": txt_chatgpt_cloz.replace("\n", "<br/>"),
                    }
                )

    # cap the number of messages
    shared.message_buffer = shared.message_buffer[-shared.max_message_buffer:]
    pv["message_buffer"] = shared.message_buffer

    v2ft_db_save(txt_chatgpt_cloz, txt_chatgpt_context, txt_audio)

    gather_threads(["audio_to_anki", "ocr", "saving_chatgpt"])
    return
