import json
import tempfile
from scipy.io.wavfile import write
from textwrap import dedent
import time
import openai
from openai.error import RateLimitError
from pathlib import Path

from .misc import tokenize, transcript_template
from .logger import red, whi, yel
from .memory import prompt_filter, load_prev_prompts
from .media import sound_preprocessing
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

    # save audio for next startup
    pv = previous_values(txt_profile)
    pv["audio_numpy_1"] = audio_numpy_1

    # try to remove silences
    try:
        audio_numpy_1 = sound_preprocessing(audio_numpy_1)
    except Exception as err:
        red(f"Error when preprocessing sound: '{err}'")

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


def alfred(txt_audio, txt_chatgpt_context, txt_profile, max_token, temperature):
    "send the previous prompt and transcribed speech to the LLM"
    if not txt_audio:
        return "No transcribed audio found.", [0, 0]
    if not txt_chatgpt_context:
        return "No txt_chatgpt_context found.", [0, 0]

    new_prompt = {
            "role": "user",
            "content": dedent(
                transcript_template.replace("CONTEXT", txt_chatgpt_context
                    ).replace("TRANSCRIPT", txt_audio))
                }

    prev_prompts = load_prev_prompts(txt_profile)
    prev_prompts = prompt_filter(prev_prompts, max_token, temperature, new_prompt_len=len(tokenize(new_prompt["content"])))

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

    formatted_messages.append(new_prompt)

    tkns += len(tokenize(formatted_messages[-1]["content"]))
    yel(f"Number of messages that will be sent to ChatGPT: {len(formatted_messages)} (representing {tkns} tokens)")

    if tkns >= 15700:
        red("More than 15700 tokens before calling ChatGPT. Bypassing to ask "
            "with fewer tokens to make sure you have room for the answer")
        return alfred(txt_audio, txt_chatgpt_context, txt_profile, max_token-500, temperature)

    if tkns >= 3700:
        red(f"More than 3700 token in question, using ChatGPT 16k")
        model_to_use = "gpt-3.5-turbo-16k"
        model_price = (0.003, 0.004)
    else:
        red(f"Using ChatGPT")
        model_to_use = "gpt-3.5-turbo"
        model_price = (0.0015, 0.002)

    assert tkns <= 16000, f"Too many tokens! ({tkns})"
    try:
        cnt = 0
        while True:
            try:
                red("Asking ChatGPT")
                cnt += 1
                response = openai.ChatCompletion.create(
                        model=model_to_use,
                        messages=formatted_messages,
                        stop="END",
                        temperature=temperature,
                        )
                break
            except RateLimitError as err:
                if cnt >= 5:
                    return red("ChatGPT: too many retries."), [0, 0]
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)

        outstr = response["choices"][0]["message"]["content"]
        outstr = outstr.replace("<br/>", "\n")  # for cosmetic purposes in the textbox

        yel(f"\n###\nChatGPT answer:\n{outstr}\n###\n")

        input_tkn_cost = response["usage"]["prompt_tokens"]
        output_tkn_cost = response["usage"]["completion_tokens"]
        tkn_cost = [input_tkn_cost, output_tkn_cost]

        tkn_cost_dol = input_tkn_cost / 1000 * model_price[0] + output_tkn_cost / 1000 * model_price[1]
        pv = previous_values(txt_profile)
        pv["total_llm_cost"] += tkn_cost_dol
        red(f"Total ChatGPT cost so far: ${pv['total_llm_cost']:.2f} (not counting whisper)")

        reason = response["choices"][0]["finish_reason"]
        if reason.lower() != "stop":
            red(f"ChatGPT's reason to strop was not 'stop' but '{reason}'")

        return outstr, tkn_cost
    except Exception as err:
        return red(f"Error with ChatGPT: '{err}'"), [0, 0]


def semiauto_mode(*args, **kwargs):
    "triggering function 'main' with mode 'semiauto'"
    whi("Triggering semiauto mode: doing everything but stop just before uploading to file")
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
        txt_chatgpt_outputstr,

        txt_chatgpt_context,
        txt_mdpath,

        txt_profile,
        sld_max_tkn,
        sld_temp,
        mode="one",
        ):
    "function called to do sequential actions: from audio to md file"
    whi("Entering main")
    if not (audio_numpy_1):
        return [
                red("None audio in microphone #1"),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                ]
    if not txt_whisp_prompt:
        return [
                red("No whisper prompt found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                ]
    if not txt_whisp_lang:
        return [
                red("No whisper language found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                ]
    if not txt_chatgpt_context:
        return [
                red("No txt_chatgpt_context found."),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                ]
    if not txt_mdpath:
        return [
                red("you should specify a markdown file path"),
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                ]

    # to_return allows to keep track of what to output to which widget
    to_return = {}
    to_return["txt_audio"] = txt_audio
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost
    to_return["txt_chatgpt_outputstr"] = txt_chatgpt_outputstr
    if Path(txt_mdpath).exists():
        to_return["mdoutput_elem"] = "\n".join(Path(txt_mdpath).read_text().strip().split("\n")[-100:])
    else:
        to_return["mdoutput_elem"] = f"File '{txt_mdpath}' not found"

    # store the default profile
    pv = previous_values(txt_profile)
    pv["latest_profile"] = txt_profile

    # save state for next start
    pv["txt_mdpath"] = txt_mdpath
    pv["sld_max_tkn"] = sld_max_tkn
    pv["temperature"] = sld_temp
    pv["txt_chatgpt_context"] = txt_chatgpt_context
    pv["txt_whisp_prompt"] = txt_whisp_prompt
    pv["txt_whisp_lang"] = txt_whisp_lang

    # get text from audio if not already parsed
    if (not txt_audio) or mode in ["auto", "semiauto"]:
        txt_audio = transcribe(audio_numpy_1, txt_whisp_prompt, txt_whisp_lang, txt_profile)
        to_return["txt_audio"] = txt_audio

    # ask chatgpt
    if (not txt_chatgpt_outputstr) or mode in ["auto", "semiauto"]:
        txt_chatgpt_outputstr, txt_chatgpt_tkncost = alfred(txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp)
    if isinstance(txt_chatgpt_tkncost, str):
        txt_chatgpt_tkncost = [int(x) for x in json.loads(txt_chatgpt_tkncost)]
    if not txt_chatgpt_tkncost:
        red("No token cost found, setting to 0")
        txt_chatgpt_tkncost = [0, 0]
    if txt_chatgpt_outputstr.startswith("Error with ChatGPT") or 0 in txt_chatgpt_tkncost:
        return [
                to_return["txt_audio"],
                txt_chatgpt_tkncost,
                to_return["txt_chatgpt_outputstr"],
                to_return["mdoutput_elem"],
                ]
    to_return["txt_chatgpt_outputstr"] = txt_chatgpt_outputstr
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost

    tkn_cost_dol = int(txt_chatgpt_tkncost[0]) / 1000 * 0.003 + int(txt_chatgpt_tkncost[1]) / 1000 * 0.004

    if "alfred" in txt_chatgpt_outputstr.lower():
        red(f"COMMUNICATION REQUESTED:\n'{txt_chatgpt_outputstr}'"),
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_outputstr"],
                to_return["mdoutput_elem"],
                ]

    # write to log output
    red(f"ChatGPT cost: {txt_chatgpt_tkncost} (${tkn_cost_dol:.3f}, not counting whisper)")
    red(f"ChatGPT answer:\n{txt_chatgpt_outputstr}")

    if mode == "semiauto":
        yel("Semiauto mode: stopping just before uploading to file")
        return [
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_outputstr"],
                to_return["mdoutput_elem"],
                ]

    red("Sending to file:")
    with open(txt_mdpath, "a") as f:
        f.write("\n")
        f.write(txt_chatgpt_outputstr.strip())
    to_return["mdoutput_elem"] = Path(txt_mdpath).read_text()

    whi("Finished adding text.\n\n")

    whi("\n\n ------------------------------------- \n\n")

    return [
            to_return["txt_audio"],
            to_return["txt_chatgpt_tkncost"],
            to_return["txt_chatgpt_outputstr"],
            to_return["mdoutput_elem"],
            ]
