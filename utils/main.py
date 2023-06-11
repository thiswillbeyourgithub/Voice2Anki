import csv
import cv2
import tempfile
from scipy.io.wavfile import write
from textwrap import dedent
import rtoml
import time
from datetime import datetime
import openai
from openai.error import RateLimitError
from pathlib import Path

from .anki import add_to_anki, audio_to_anki, sync_anki
from .misc import tokenize, transcript_template
from .logger import red, whi, yel
from .memory import prompt_filter, load_prev_prompts
from .media import sound_preprocessing, get_img_source
from .profiles import previous_values

assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()


def transcribe(audio_numpy_1, txt_whisp_prompt, txt_whisp_lang, txt_profile):
    "turn the 1st audio track into text"
    whi("Transcribing audio")

    if audio_numpy_1 is None:
        return red("Error: None audio_numpy_1")

    if txt_whisp_prompt is None:
        return red("Error: None whisper prompt")

    if txt_whisp_lang is None:
        return red("Error: None whisper language")

    # try to remove silences
    try:
        audio_numpy_1 = sound_preprocessing(audio_numpy_1)
    except Exception as err:
        red(f"Error when preprocessing sound: '{err}'")

    # save audio for next startup
    pv = previous_values(txt_profile)
    pv["audio_numpy_1"] = audio_numpy_1

    # save audio to temp file
    whi("Saving audio as wav file")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="transcribe")
    write(tmp.name, audio_numpy_1[0], audio_numpy_1[1])

    try:
        assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
        cnt = 0
        while True:
            try:
                whi("Asking Whisper")
                cnt += 1
                with open(tmp.name, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        prompt=txt_whisp_prompt,
                        language=txt_whisp_lang)
                    txt_audio = transcript["text"]
                    yel(f"\nWhisper transcript: {txt_audio}")
                    Path(tmp.name).unlink(missing_ok=False)
                    return txt_audio
            except RateLimitError as err:
                if cnt >= 5:
                    Path(tmp.name).unlink(missing_ok=False)
                    return red("Whisper: too many retries.")
                red(f"Error from whisper: '{err}'")
                time.sleep(2 * cnt)
    except Exception as err:
        Path(tmp.name).unlink(missing_ok=False)
        return red(f"Error when transcribing audio: '{err}'")


def alfred(txt_audio, txt_chatgpt_context, profile, max_token, temperature):
    "send the previous prompt and transcribed speech to the LLM"
    if not txt_audio:
        return "No transcribed audio found.", None
    if not txt_chatgpt_context:
        return "No txt_chatgpt_context found.", None

    prev_prompts = load_prev_prompts(profile)
    prev_prompts = prompt_filter(prev_prompts, max_token, temperature)

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
        tkns += m["tkn_len"]
        if "answer" in m:
            assert m["role"] == "user", "expected user"
            formatted_messages.append({
                "role": "assistant",
                "content": m["answer"]})

    formatted_messages.append(
            {
                "role": "user",
                "content": dedent(
                    transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", txt_audio))})
    tkns += len(tokenize(formatted_messages[-1]["content"]))
    yel(f"Number of messages that will be sent to ChatGPT: {len(formatted_messages)} (representing {tkns} tokens)")

    if tkns >= 3700:
        red("More than 3750 tokens before calling ChatGPT. Bypassing to ask "
            "with fewer tokens to make sure you have room for the answer")
        return alfred(txt_audio, txt_chatgpt_context, profile, max_token-500, temperature)

    assert tkns <= 4000, f"Too many tokens! ({tkns})"
    try:
        cnt = 0
        while True:
            try:
                red("Asking ChatGPT")
                cnt += 1
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=formatted_messages,
                        stop="END",
                        temperature=temperature,
                        )
                break
            except RateLimitError as err:
                if cnt >= 5:
                    return red("ChatGPT: too many retries."), None
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)

        cloz = response["choices"][0]["message"]["content"]
        cloz = cloz.replace("<br/>", "\n")  # for cosmetic purposes in the textbox
        yel(f"\n###\nChatGPT answer:\n{cloz}\n###\n")

        tkn_cost = response["usage"]["total_tokens"]

        reason = response["choices"][0]["finish_reason"]
        if reason.lower() != "stop":
            red(f"ChatGPT's reason to strop was not 'stop' but '{reason}'")

        return cloz, tkn_cost
    except Exception as err:
        return red(f"Error with ChatGPT: '{err}'"), 0


def semiauto_mode(*args, **kwargs):
    "triggering function 'main' with mode 'semiauto'"
    whi("Triggering semiauto mode: doing everything but stop just before uploading to anki")
    return main(*args, **kwargs, mode="semiauto")


def auto_mode(*args, **kwargs):
    "triggering function 'main' with mode 'auto'"
    whi("Triggering auto mode")
    return main(*args, **kwargs, mode="auto")


def main(
        audio_numpy_1,
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
    if not (audio_numpy_1):
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

    # to_return allows to keep track of what to output to which widget
    to_return = {}
    to_return["txt_audio"] = txt_audio
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz

    # store the default profile
    pv = previous_values(profile)
    pv["latest_profile"] = profile

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
        txt_audio = transcribe(audio_numpy_1, txt_whisp_prompt, txt_whisp_lang, profile)
        to_return["txt_audio"] = txt_audio

    # ask chatgpt
    if (not txt_chatgpt_cloz) or mode in ["auto", "semiauto"]:
        txt_chatgpt_cloz, txt_chatgpt_tkncost = alfred(txt_audio, txt_chatgpt_context, profile, sld_max_tkn, sld_temp)
    if not txt_chatgpt_tkncost:
        red("No token cost found, setting to 0")
        txt_chatgpt_tkncost = 0
    if txt_chatgpt_cloz.startswith("Error with ChatGPT") or txt_chatgpt_tkncost == 0:
        return [
                to_return["txt_audio"],
                txt_chatgpt_tkncost,
                to_return["txt_chatgpt_cloz"],
                ]
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost

    tkn_cost_dol = int(txt_chatgpt_tkncost) / 1000 * 0.002

    # checks clozes validity
    clozes = txt_chatgpt_cloz.split("#####")
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        red(f"Invalid cloze: '{txt_chatgpt_cloz}'")
        return [
                to_return["txt_audio"],
                txt_chatgpt_tkncost,
                to_return["txt_chatgpt_cloz"],
                ]

    if "alfred" in txt_chatgpt_cloz.lower():
        red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_cloz}'"),
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    # add cloze to output
    red(f"ChatGPT cost: {txt_chatgpt_tkncost} (${tkn_cost_dol:.3f})")
    red(f"ChatGPT answer:\n{txt_chatgpt_cloz}")

    # send to anki
    metadata = rtoml.dumps(
            {
            "author": "WhisperToAnki",
            "transcripted_text": txt_audio,
            "chatgpt_tkn_cost": txt_chatgpt_tkncost,
            "chatgpt_dollars_cost": tkn_cost_dol,
            }, pretty=True)
    results = []

    if mode == "semiauto":
        yel("Semiauto mode: stopping just before uploading to anki")
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    red("Sending to anki:")

    # sending sound file to anki media
    audio_html = audio_to_anki(audio_numpy_1)
    if "Error" in audio_html:  # then out is an error message and not the source
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]
    txt_source += audio_html

    # if not found or empty: create csv file in profile containing the header
    try:
        if (not Path(f'./profiles/{profile}/sent_to_anki.csv').exists()
                ) or not (
                        Path(f'./profiles/{profile}/sent_to_anki.csv').read_text().strip()
                        ):
            yel(f"Creating sent_to_anki.csv file in profiles/{profile}")
            with open(f'./profiles/{profile}/sent_to_anki.csv', 'w', newline='') as csvfile:
                file = csv.writer(csvfile, delimiter=",")
                file.writerow("body,source,GPToAnkiMetadata,tags,deck".split(","))
    except Exception as err:
        red(f"Error when creating csv file: '{err}'")

    # anki tags
    d = datetime.today()
    today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"
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

        # add to csv file
        try:
            with open(f'./profiles/{profile}/sent_to_anki.csv', 'a', newline='') as csvfile:
                file = csv.writer(csvfile, delimiter=",")
                to_add = []
                # remove newlines
                for new in [cl, txt_source, metadata, new_tags, txt_deck]:
                    if isinstance(new, (list, set)):
                        new = ";".join(new)
                    new = str(new)
                    new = new.replace(",", r"\,").replace("\n", "\\n")
                    to_add.append(new)
                file.writerow(to_add)
        except Exception as err:
            red(f"Error when appending cloze csv file: '{err}'")

    results = [str(r) for r in results if str(r).isdigit()]

    # trigger anki sync
    sync_anki()
    whi("Synchronized anki\n")

    if not len(results) == len(clozes):
        red(f"Some flashcards were not added!"),
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    whi("Finished adding card.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    return [
            to_return["txt_audio"],
            to_return["txt_chatgpt_tkncost"],
            to_return["txt_chatgpt_cloz"],
            ]
