import json
import pickle
from textwrap import dedent
import rtoml
import time
from datetime import datetime
import gradio as gr
import openai
from openai.error import RateLimitError
from pathlib import Path

from utils.to_anki import add_to_anki, audio_to_anki
from utils.misc import tokenize, transcript_template, check_source
from utils.logger import red, whi, yel
from utils.memory import curate_previous_prompts, recur_improv, previous_values, memorized_prompts
from utils.media import remove_silences, enhance_audio, get_image, get_img_source, reset_audio, reset_image

# misc init values
Path("./cache").mkdir(exist_ok=True)
assert Path("API_KEY.txt").exists(), "No api key found"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()
d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

pv = previous_values()

def transcribe(audio_path, txt_whisp_prompt):
    whi("Transcribing audio")

    # try to remove silences
    audio_path = remove_silences(audio_path)

    # try to enhance quality
    audio_path = enhance_audio(audio_path)

    # save audio for next startup
    pv["audio_path"] = audio_path
    try:
        assert "TRANSCRIPT" not in txt_whisp_prompt, "found TRANSCRIPT in txt_whisp_prompt"
        cnt = 0
        while True:
            try:
                whi("Asking Whisper")
                cnt += 1
                with open(audio_path, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                            model="whisper-1",
                            file=audio_file,
                            prompt=txt_whisp_prompt,
                            language="fr",
                            )
                    txt_audio = transcript["text"]
                    return txt_audio
            except Exception as err:
                if cnt >= 5:
                    return red("Whisper: too many retries.")
                red(f"Error from whisper: '{err}'")
                time.sleep(2 * cnt)
    except Exception as err:
        return red(f"Error when transcribing audio: '{err}'")


def alfred(txt_audio, context):
    if not txt_audio:
        return "No transcribed audio found.", None
    if not context:
        return "No context found.", None

    global memorized_prompts
    memorized_prompts = curate_previous_prompts(memorized_prompts)

    formatted_messages = []
    tkns = 0
    for m in memorized_prompts:
        if m["disabled"] is True:
            continue
        formatted_messages.append(m.copy())
        tkns += len(tokenize(m["content"]))
        del formatted_messages[-1]["disabled"]
    formatted_messages.append(
            {"role": "user", "content": dedent(
                transcript_template.replace("CONTEXT", context).replace("TRANSCRIPT", txt_audio)),
                })
    tkns += len(tokenize(formatted_messages[-1]["content"]))
    yel(f"Number of messages in ChatGPT prompt: {len(formatted_messages)} (tokens: {tkns})")

    assert tkns <= 4000, f"Too many tokens! ({tkns})"
    try:
        cnt = 0
        while True:
            try:
                red("Asking chatgpt")
                cnt += 1
                response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=formatted_messages,
                        stop="END",
                        temperature=0.1,
                        )
                break
            except RateLimitError as err:
                if cnt >= 5:
                    return red("ChatGPT: too many retries.")
                red(f"Server overloaded #{cnt}, retrying in {2 * cnt}s : '{err}'")
                time.sleep(2 * cnt)
        cloz = response["choices"][0]["message"]["content"]
        cloz = cloz.replace("<br/>", "\n")  # for cosmetic purposes in the textbox
        return cloz, response
    except Exception as err:
        return None, red(f"Error with ChatGPT: '{err}'")



def auto_mode(*args, **kwargs):
    whi("Triggering auto mode")
    return main(*args, **kwargs, auto_mode=True)

def main(
        audio_path,
        txt_audio,
        txt_whisp_prompt,

        txt_chatgpt_resp,
        txt_chatgpt_cloz,

        txt_context,
        txt_deck,
        txt_tags,

        gallery,
        txt_source,
        old_output,
        auto_mode=False,
        ):
    to_return = {}
    whi("Entering main")
    to_return["output"] = ""

    if not (audio_path or txt_audio):
        return red("No audio in either microphone data or audio file")
    if not txt_whisp_prompt:
        return red("No whisper prompt found.")
    if not txt_context:
        return red("No txt_context found.")
    if not txt_deck:
        return red("you should specify a deck")
    if not txt_tags:
        return red("you should specify tags")

    if not ((gallery is not None) or txt_source):
        to_return["output"] += red("you should probably specify either image+source or source")
    if (gallery is not None) and not txt_source:
        return red("Image but no source: please load the source")

    txt_source = check_source(txt_source)

    # manage sound path
    audio_html = audio_to_anki(audio_path)
    if "Error" in audio_html:  # then out is an error message and not the source
        to_return["output"] = audio_html + to_return["audio_html"]
        return [to_return["output"],
                to_return["txt_audio"],
                to_return["txt_chatgpt_resp"],
                to_return["txt_chatgpt_cloz"],
                ]
    txt_source += audio_html

    # save image and audio for next startup
    if gallery is not None:
        pv["gallery"] = gallery

    # get text from audio if not already parsed
    if (not txt_audio) or auto_mode:
        txt_audio = transcribe(audio_path, txt_whisp_prompt)
    to_return["txt_audio"] = txt_audio

    to_return["output"] += f"\n>>>> Whisper:\n{txt_audio}"
    yel(f"\n###\nTranscript:\n{txt_audio}\n###\n")

    # ask chatgpt
    if (not txt_chatgpt_cloz) or auto_mode:
        txt_chatgpt_cloz, txt_chatgpt_resp = alfred(txt_audio, txt_context)
    if isinstance(txt_chatgpt_resp, str):
        txt_chatgpt_resp = json.loads(txt_chatgpt_resp)
    to_return["txt_chatgpt_cloz"] = txt_chatgpt_cloz
    to_return["txt_chatgpt_resp"] = txt_chatgpt_resp

    tkn_cost = txt_chatgpt_resp["usage"]["total_tokens"]
    tkn_cost_dol = tkn_cost / 1000 * 0.002
    yel(f"\n###\nChatGPT answer:\n{txt_chatgpt_cloz}\n###\n")

    # checks clozes validity
    clozes = txt_chatgpt_cloz.split("#####")
    if not clozes or "{{c1::" not in txt_chatgpt_cloz:
        return red(f"{to_return['output']}\n\nInvalid cloze: '{txt_chatgpt_cloz}'")

    if "alfred:" in txt_chatgpt_cloz.lower():
        return [
                red(f"{to_return['output']}\nCOMMUNICATION REQUESTED:\n{txt_chatgpt_cloz}"),
                to_return["txt_audio"],
                to_return["txt_chatgpt_resp"],
                to_return["txt_chatgpt_cloz"],
                ]

    # show output
    to_return["output"] += f"\n>>>> ChatGpt {tkn_cost} (${tkn_cost_dol:.3f}):\n{txt_chatgpt_cloz}"

    # send to anki
    metadata = {
            "author": "WhisperToAnki",
            "transcripted_text": txt_audio,
            "chatgpt_tkn_cost": tkn_cost,
            "chatgpt_dollars_cost": tkn_cost_dol,
            }
    results = []
    to_return["output"] += "\n>>>> To anki:\n"

    # anki tags
    new_tags = txt_tags.split(" ") + [f"WhisperToAnki::{today}"],
    if "img" not in txt_source:
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
    if not len(results) == len(clozes):
        return [
                red(f"{to_return['output']}\n\n-> Some cards were not added!"),
                to_return["txt_audio"],
                to_return["txt_chatgpt_resp"],
                to_return["txt_chatgpt_cloz"],
                ]


    whi("Finished loop.\n\n")

    # save state for next start
    pv["txt_deck"] = txt_deck
    pv["txt_tags"] = txt_tags
    pv["txt_content"] = txt_context
    pv["txt_whisp_prompt"] = txt_whisp_prompt

    to_return["output"] += "\n\nDONE"

    to_return["output"] += "\n\n ------------------------------------- \n\n" + old_output.strip()

    # the order matters because it's used to reset some fields
    return [
            to_return["output"],
            to_return["txt_audio"],
            to_return["txt_chatgpt_resp"],
            to_return["txt_chatgpt_cloz"],
            ]


#with gr.Blocks(analytics_enabled=False, title="WhisperToAnki", theme=gr.themes.Soft()) as demo:
with gr.Blocks(analytics_enabled=False, title="WhisperToAnki") as demo:
        gr.Markdown("Valide tes partiels")

        # hidden, to store the request answer from chatgpt
        txt_chatgpt_resp = gr.Textbox(value=None, visible=False)

        with gr.Row():
            with gr.Row():
                with gr.Column():
                    gallery = gr.Gallery(value=pv["gallery"], label="Source images").style(columns=[1], rows=[1], object_fit="none", height="auto", container=True)
                    with gr.Row():
                        rst_img_btn = gr.Button(value="Clear", variant="primary")
                        img_btn = gr.Button(value="Add image from clipboard", variant="secondary")
                        source_btn = gr.Button(value="Load source from gallery", variant="secondary")
                    txt_source = gr.Textbox(value=pv["txt_source"], label="Source field", lines=1)
            with gr.Column():
                txt_deck = gr.Textbox(value=pv["txt_deck"], label="Deck name", max_lines=1)
                txt_tag = gr.Textbox(value=pv["txt_tags"], label="Tags", lines=1)
                txt_context = gr.Textbox(value=pv["txt_context"], label="Contexte pour ChatGPT")
                txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], label="Contexte pour Whisper")

        with gr.Row():
            with gr.Column():
                audio_path = gr.Audio(source="microphone", type="filepath", label="Audio", format="wav", value=pv["audio_path"])
                rst_audio_btn = gr.Button(value="Reset audio", variant="secondary")
            with gr.Column():
                txt_audio = gr.Textbox(value=pv["txt_audio"], label="Audio transcript", lines=10, max_lines=10)
                txt_chatgpt_cloz = gr.Textbox(value=pv["txt_chatgpt_cloz"], label="ChatGPT output", lines=10, max_lines=10)

        with gr.Row():
            with gr.Column(scale=1):
                auto_btn = gr.Button(value="Auto", variant="secondary")

            with gr.Column(scale=9):
                with gr.Row():
                    transcript_btn = gr.Button(value="To Whisper", variant="stop")
                    chatgpt_btn = gr.Button(value="To ChatGPT", variant="stop")
                    anki_btn = gr.Button(value="To Anki", variant="stop")

                    improve_btn = gr.Button(value="Improve", variant="secondary")


        # output
        output_elem = gr.Textbox(value="Welcome.", label="Logging", lines=20, max_lines=100)

        # events
        source_btn.click(fn=get_img_source, inputs=[gallery], outputs=[txt_source])
        chatgpt_btn.click(fn=alfred, inputs=[txt_audio, txt_context], outputs=[txt_chatgpt_cloz, txt_chatgpt_resp])
        transcript_btn.click(fn=transcribe, inputs=[audio_path, txt_whisp_prompt], outputs=[txt_audio])
        img_btn.click(fn=get_image, inputs=[txt_source, gallery, output_elem], outputs=[gallery, output_elem])
        rst_audio_btn.click(fn=reset_audio, outputs=[audio_path])
        rst_img_btn.click(fn=reset_image, outputs=[gallery, txt_source])
        anki_btn.click(
                fn=main,
                inputs=[
                    audio_path,
                    txt_audio,
                    txt_whisp_prompt,

                    txt_chatgpt_resp,
                    txt_chatgpt_cloz,

                    txt_context,
                    txt_deck,
                    txt_tag,

                    gallery,
                    txt_source,
                    output_elem,
                    ],
                outputs=[
                    output_elem,
                    txt_audio,
                    txt_chatgpt_resp,
                    txt_chatgpt_cloz,
                    ],
                )
        auto_btn.click(
                fn=auto_mode,
                inputs=[
                    audio_path,
                    txt_audio,
                    txt_whisp_prompt,

                    txt_chatgpt_resp,
                    txt_chatgpt_cloz,

                    txt_context,
                    txt_deck,
                    txt_tag,

                    gallery,
                    txt_source,
                    output_elem,
                    ],
                outputs=[
                    output_elem,
                    txt_audio,
                    txt_chatgpt_resp,
                    txt_chatgpt_cloz,
                    ],
                )
        improve_btn.click(
                fn=recur_improv,
                inputs=[
                    txt_audio,
                    txt_whisp_prompt,
                    txt_chatgpt_cloz,
                    txt_context,
                    output_elem,
                    ],
                outputs=[output_elem],
                )

demo.queue()
