import cv2
import tempfile
from scipy.io.wavfile import write
from textwrap import dedent
import rtoml
import time
from datetime import datetime
import gradio as gr
import openai
from openai.error import RateLimitError
from pathlib import Path

from utils.anki import add_to_anki, audio_to_anki, sync_anki
from utils.misc import tokenize, transcript_template
from utils.logger import red, whi, yel
from utils.memory import prompt_filter, recur_improv, load_prev_prompts
from utils.media import remove_silences, get_image, get_img_source, reset_audio, reset_image
from utils.profiles import get_profiles, switch_profile, previous_values

# misc init values
Path("./cache").mkdir(exist_ok=True)
assert Path("API_KEY.txt").exists(), "No api key found. Create a file API_KEY.txt and paste your openai API key inside"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()
d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

pv = previous_values()

def save_audio(profile, audio_numpy):
    whi("Saving audio to profile")
    pv = previous_values(profile)
    pv["audio_numpy"] = audio_numpy

def transcribe(audio_numpy, txt_whisp_prompt, output):
    whi("Transcribing audio")

    if audio_numpy is None:
        return red("Error: None audio_numpy"), f"Error: None audio_numpy\n\n{output}"

    if txt_whisp_prompt is None:
        return red("Error: None whisper prompt"), f"Error: None whisper prompt\n\n{output}"

    # try to remove silences
    audio_numpy = remove_silences(audio_numpy)

    # DISABLED: it seems to completely screw up the audio :(
    # try to enhance quality
    # audio_numpy = enhance_audio(audio_numpy)

    # save audio for next startup
    pv["audio_numpy"] = audio_numpy

    # save audio to temp file
    whi("Saving audio as wav file")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav")
    write(tmp.name, audio_numpy[0], audio_numpy[1])
    tempwavpath = tmp.name

    try:
        assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
        cnt = 0
        while True:
            try:
                whi("Asking Whisper")
                cnt += 1
                with open(tempwavpath, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio_file,
                        prompt=txt_whisp_prompt,
                        language="fr")
                    txt_audio = transcript["text"]
                    yel(f"\nWhisper transcript: {txt_audio}")
                    return txt_audio, f"Whisper transcription: {txt_audio}\n\n{output}"
            except RateLimitError as err:
                if cnt >= 5:
                    return red("Whisper: too many retries."), f"Whisper: too many retries.\n\n{output}"
                red(f"Error from whisper: '{err}'")
                time.sleep(2 * cnt)
    except Exception as err:
        return red(f"Error when transcribing audio: '{err}'"), f"Error when transcribing audio: '{err}'\n\n{output}"


def alfred(txt_audio, txt_chatgpt_context, profile, max_token, output):
    if not txt_audio:
        return "No transcribed audio found.", None, f"No transcribed audio found\n\n{output}"
    if not txt_chatgpt_context:
        return "No txt_chatgpt_context found.", None, f"No txt_chatgpt_context found\n\n{output}"

    prev_prompts = load_prev_prompts(profile)
    prev_prompts = prompt_filter(prev_prompts, max_token)

    # check the number of token is fine and format the previous
    # prompts in chatgpt format
    formatted_messages = []
    tkns = 0
    for m in prev_prompts:
        formatted_messages.append(m.copy())
        tkns += m["tkn_len"]
        if "answer" in m:
            assert m["role"] == "user", "expected user"
            formatted_messages.append({
                "role": "assistant",
                "content": m["answer"]})

    for i, fm in enumerate(formatted_messages):
        for col in ["timestamp", "priority", "tkn_len", "answer", "disabled"]:
            if col in fm:
                del formatted_messages[i][col]
    formatted_messages.append(
            {
                "role": "user",
                "content": dedent(
                    transcript_template.replace("CONTEXT", txt_chatgpt_context).replace("TRANSCRIPT", txt_audio))})
    tkns += len(tokenize(formatted_messages[-1]["content"]))
    yel(f"Number of messages in ChatGPT prompt: {len(formatted_messages)} (tokens: {tkns})")

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
                        temperature=0.5,
                        )
                break
            except RateLimitError as err:
                if cnt >= 5:
                    return red("ChatGPT: too many retries."), None, f"ChatGPT: too many retries.\n\n{output}"
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)

        cloz = response["choices"][0]["message"]["content"]
        cloz = cloz.replace("<br/>", "\n")  # for cosmetic purposes in the textbox
        yel(f"\n###\nChatGPT answer:\n{cloz}\n###\n")

        reason = response["choices"][0]["finish_reason"]
        assert reason == "stop", "ChatGPT's reason to strop was not stop"

        tkn_cost = response["usage"]["total_tokens"]

        return cloz, tkn_cost, f"New cloze created: {cloz}\n\n{output}"
    except Exception as err:
        return None, 0, f"Error with ChatGPT: '{err}'\n\n{output}"


def semiauto_mode(*args, **kwargs):
    whi("Triggering semiauto mode: doing everything but stop just before uploading to anki")
    return main(*args, **kwargs, mode="semiauto")

def auto_mode(*args, **kwargs):
    whi("Triggering auto mode")
    return main(*args, **kwargs, mode="auto")


def main(
        audio_numpy,
        txt_audio,
        txt_whisp_prompt,

        txt_chatgpt_tkncost,
        txt_chatgpt_cloz,

        txt_chatgpt_context,
        txt_deck,
        txt_tags,

        gallery,
        profile,
        sld_max_tkn,
        old_output,
        mode="one",
        ):
    whi("Entering main")
    if not (audio_numpy or txt_audio):
        return [
                red("No audio in either microphone data or audio file"),
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
    to_return["output"] = old_output
    to_return["txt_audio"] = txt_audio
    to_return["txt_chatgpt_tkncost"] = txt_chatgpt_tkncost
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz

    global pv
    pv["profile"] = profile
    pv = previous_values(profile)

    if gallery is None or len(gallery) == 0:
        to_return["output"] = red("you should probably specify an image in source\n") + to_return["output"]
        txt_source = "<br>"
    else:
        txt_source = get_img_source(gallery)

    # save state for next start
    pv["txt_deck"] = txt_deck
    pv["txt_tags"] = txt_tags
    pv["sld_max_tkn"] = sld_max_tkn
    pv["profile"] = profile
    pv["txt_chatgpt_context"] = txt_chatgpt_context
    pv["txt_whisp_prompt"] = txt_whisp_prompt

    # manage sound path
    audio_html = audio_to_anki(audio_numpy)
    if "Error" in audio_html:  # then out is an error message and not the source
        to_return["output"] = audio_html + to_return["output"]
        return [
                to_return["output"],
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]
    txt_source += audio_html

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
        txt_audio, to_return["output"] = transcribe(audio_numpy, txt_whisp_prompt, to_return["output"])
        to_return["txt_audio"] = txt_audio

    # ask chatgpt
    if (not txt_chatgpt_cloz) or mode in ["auto", "semiauto"]:
        txt_chatgpt_cloz, txt_chatgpt_tkncost, to_return["output"] = alfred(txt_audio, txt_chatgpt_context, profile, sld_max_tkn, to_return["output"])
    if not txt_chatgpt_tkncost:
        red("No token cost found, setting to 0")
        txt_chatgpt_tkncost = 0
    if to_return["output"].startswith("Error with ChatGPT"):
        return [
                to_return["output"],
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
        return red(f"{to_return['output']}\n\nInvalid cloze: '{txt_chatgpt_cloz}'")

    if "alfred:" in txt_chatgpt_cloz.lower():
        return [
                red(f"{to_return['output']}\nCOMMUNICATION REQUESTED:\n{txt_chatgpt_cloz}"),
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    # show output
    to_return["output"] += f"\n>>>> ChatGPT {txt_chatgpt_tkncost} (${tkn_cost_dol:.3f}):\n{txt_chatgpt_cloz}"

    # send to anki
    metadata = {
            "author": "WhisperToAnki",
            "transcripted_text": txt_audio,
            "chatgpt_tkn_cost": txt_chatgpt_tkncost,
            "chatgpt_dollars_cost": tkn_cost_dol,
            }
    results = []
    to_return["output"] += "\n>>>> To anki:\n"

    if mode == "semiauto":
        yel("Semiauto mode: stopping just before uploading to anki")
        return [
                to_return["output"],
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

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
                    note_metadata=rtoml.dumps(metadata, pretty=True),
                    tags=new_tags,
                    deck_name=txt_deck,
                )
                )
        to_return["output"] += f"* {cl}\n"
    results = [str(r) for r in results if str(r).isdigit()]

    # trigger anki sync
    sync_anki()
    to_return["output"] += "Synchronized anki\n"

    if not len(results) == len(clozes):
        return [
                red(f"{to_return['output']}\n\n-> Some cards were not added!"),
                to_return["txt_audio"],
                to_return["txt_chatgpt_tkncost"],
                to_return["txt_chatgpt_cloz"],
                ]

    whi("Finished loop.\n\n")

    to_return["output"] += "\n\nDONE"

    to_return["output"] += "\n\n ------------------------------------- \n\n" + old_output.strip()

    # the order matters because it's used to reset some fields
    return [
            to_return["output"],
            to_return["txt_audio"],
            to_return["txt_chatgpt_tkncost"],
            to_return["txt_chatgpt_cloz"],
            ]


with gr.Blocks(analytics_enabled=False, title="WhisperToAnki") as demo:
    gr.Markdown("WhisperToAnki")

    # hidden, to store the request answer from chatgpt
    txt_chatgpt_tkncost = gr.Textbox(value=None, visible=False)

    with gr.Row():
        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(value=pv["gallery"], label="Source images").style(columns=3, rows=1, object_fit="scale-down", height="auto", container=True)
                with gr.Row():
                    rst_img_btn = gr.Button(value="Clear", variant="primary").style(full_width=False, size="sm")
                    img_btn = gr.Button(value="Add image from clipboard", variant="secondary").style(full_width=False, size="sm")
            with gr.Column(scale=2):
                txt_profile = gr.Textbox(placeholder=",".join(get_profiles()), label="Profile")
                txt_deck = gr.Textbox(value=pv["txt_deck"], label="Deck name", max_lines=1)
                txt_tags = gr.Textbox(value=pv["txt_tags"], label="Tags", lines=1)
                txt_chatgpt_context = gr.Textbox(value=pv["txt_chatgpt_context"], label="Context for Chat model")
                txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], label="Context for SpeechToText")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                rst_audio_btn = gr.Button(value="Reset audio", variant="primary").style(full_width=False, size="sm")
                audio_numpy = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=None)
        with gr.Column(scale=3):
            txt_audio = gr.Textbox(value=pv["txt_audio"], label="Audio transcript", lines=5, max_lines=10)
            txt_chatgpt_cloz = gr.Textbox(value=pv["txt_chatgpt_cloz"], label="ChatGPT output", lines=5, max_lines=10)

    with gr.Row():
        with gr.Column(scale=1):
            auto_btn = gr.Button(value="Autopilot", variant="secondary")  # .style(full_width=False)  #, size="sm")

        with gr.Column(scale=9):
            with gr.Row():
                semiauto_btn = gr.Button(value="Speech to Cloze", variant="stop")  # .style(full_width=False)  #, size="sm")
                transcript_btn = gr.Button(value="Speech to text", variant="stop")
                chatgpt_btn = gr.Button(value="Text to cloze(s)", variant="stop")
                anki_btn = gr.Button(value="Cloze to Anki", variant="stop")

    with gr.Row():
        with gr.Column(scale=9):
            with gr.Row():
                improve_btn = gr.Button(value="Improve", variant="secondary")
                sld_improve = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="New example priority")
        with gr.Column(scale=1):
            sld_max_tkn = gr.Slider(minimum=500, maximum=3500, value=pv["max_tkn"], step=500, label="ChatGPT history token size")

    # output
    output_elem = gr.Textbox(value="Welcome.", label="Logging", lines=20, max_lines=100)

    # events
    audio_numpy.change(fn=save_audio, inputs=[txt_profile, audio_numpy])
    txt_profile.submit(fn=switch_profile, inputs=[txt_profile, output_elem], outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, gallery, audio_numpy, txt_audio, txt_chatgpt_cloz, txt_profile, output_elem])
    txt_profile.blur(fn=switch_profile, inputs=[txt_profile, output_elem], outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, gallery, audio_numpy, txt_audio, txt_chatgpt_cloz, txt_profile, output_elem])
    chatgpt_btn.click(fn=alfred, inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, output_elem], outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost, output_elem])
    transcript_btn.click(fn=transcribe, inputs=[audio_numpy, txt_whisp_prompt, output_elem], outputs=[txt_audio, output_elem])
    img_btn.click(fn=get_image, inputs=[gallery, output_elem], outputs=[gallery, output_elem])
    rst_audio_btn.click(fn=reset_audio, inputs=[output_elem], outputs=[audio_numpy, output_elem])
    rst_img_btn.click(fn=reset_image, inputs=[output_elem], outputs=[gallery, output_elem])

    anki_btn.click(
            fn=main,
            inputs=[
                audio_numpy,
                txt_audio,
                txt_whisp_prompt,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                output_elem,
                ],
            outputs=[
                output_elem,
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ],
            )
    auto_btn.click(
            fn=auto_mode,
            inputs=[
                audio_numpy,
                txt_audio,
                txt_whisp_prompt,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                output_elem,
                ],
            outputs=[
                output_elem,
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ],
            # batch=True,  # TODO: enable batching when you figure out how to convert all to iterables
            # max_batch_size=10,
            )
    semiauto_btn.click(
            fn=semiauto_mode,
            inputs=[
                audio_numpy,
                txt_audio,
                txt_whisp_prompt,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                output_elem,
                ],
            outputs=[
                output_elem,
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ],
            )
    improve_btn.click(
            fn=recur_improv,
            inputs=[
                txt_profile,
                txt_audio,
                txt_whisp_prompt,
                txt_chatgpt_cloz,
                txt_chatgpt_context,
                sld_improve,
                output_elem,
                ],
            outputs=[output_elem],
            )

demo.queue()
