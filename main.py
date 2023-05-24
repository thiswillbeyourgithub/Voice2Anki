from unsilence import Unsilence
import torchaudio
from speechbrain.pretrained import WaveformEnhancement
import tiktoken
import json
from joblib import Memory
from bs4 import BeautifulSoup
import pickle
from textwrap import dedent
import rtoml
import hashlib
import time
import ankipandas as akp
import cv2
import numpy as np
import pyperclip3
from datetime import datetime
import gradio as gr
import openai
from openai.error import RateLimitError
from pathlib import Path
import shutil

from utils.to_anki import add_to_anki
from utils.misc import convert_paste, tokenize
from utils.logger import red, whi, yel, log
from utils.memory import curate_previous_prompts, memorized_prompts

# misc init values
Path("./cache").mkdir(exist_ok=True)
assert Path("API_KEY.txt").exists(), "No api key found"
openai.api_key = str(Path("API_KEY.txt").read_text()).strip()
d = datetime.today()
today = f"{d.day:02d}/{d.month:02d}/{d.year:04d}"

transcript_template = """
Contexte: "CONTEXT"

Transcript:
'''
TRANSCRIPT
'''
"""


# load anki profile using ankipandas just to get the media folder
db_path = akp.find_db(user="Main")
red(f"WhisperToAnki will use anki collection found at {db_path}")

# check that akp will not go in trash
if "trash" in str(db_path).lower():
    red("Ankipandas seems to have "
        "found a collection that might be located in "
        "the trash folder. If that is not your intention "
        "cancel now. Waiting 10s for you to see this "
        "message before proceeding.")
    time.sleep(1)
anki_media = Path(db_path).parent / "collection.media"
assert anki_media.exists(), "Media folder not found!"

# voice cleaning model
voice_cleaner = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir="cache/pretrained_models/mtl-mimic-voicebank",
)

# remember previous states
def default_state():
    return {
            "txt_deck": "Default",
            "txt_tags": "WhisperToAnki::untagged",
            "txt_context": "CONTEXT",
            "txt_whisp_prompt": "TRANSCRIPT",
            }
if Path("./inferences/whisperToAnki_state.toml").exists():
    try:
        with open("./inferences/whisperToAnki_state.toml", "r") as f:
            last_state = rtoml.load(f)
    except Exception as err:
        red(f"Exception: '{err}'")
        last_state = default_state()
else:
    last_state = default_state()

red(f"Last state: {last_state}")

# reset the last state if missing some values
if sorted([k for k in last_state.keys()]) != sorted([k for k in default_state().keys()]):
    red(f"Malformed last state, resetting.")
    last_state = default_state()

# rtoml saves None as null, replacing by None
for k, v in last_state.items():
    if v == "null":
        last_state[k] = None

# reload previous image and microphone
if Path("./cache/voice_cards_last_image.pickle").exists():
    with open("./cache/voice_cards_last_image.pickle", "rb") as f:
        prev_image = pickle.load(f)
else:
    prev_image = None
if Path("./cache/voice_cards_last_source.pickle").exists():
    with open("./cache/voice_cards_last_source.pickle", "rb") as f:
        prev_source = pickle.load(f)
else:
    prev_source = None
if Path("./cache/voice_cards_last_audio.pickle").exists():
    with open("./cache/voice_cards_last_audio.pickle", "rb") as f:
        prev_audio = pickle.load(f)
else:
    prev_audio = None


def get_image(source):
    whi("Getting image")
    try:
        if source:
            soup = BeautifulSoup(source, 'html.parser')
            path = soup.find_all('img')[0]['src']
            decoded = cv2.imread(str(anki_media / path))
            return decoded

        yel("Getting image.")
        pasted = pyperclip3.paste()
        decoded = cv2.imdecode(np.frombuffer(pasted, np.uint8), flags=1)
        return decoded
    except Exception as err:
        return red(f"Error: {err}")


def get_img_source(decoded):
    whi("Getting source from image")
    try:
        img_hash = hashlib.md5(decoded).hexdigest()
        new = anki_media / f"{img_hash}.png"
        if not new.exists():
            cv2.imwrite(str(new), decoded)
        source = f'<img src="{new.name}" '
        source += 'type="made_by_WhisperToAnki">'
        return source
    except Exception as err:
        return red(f"Error getting source: '{err}'")


def reset_audio():
    whi("Reset audio.")
    return None


def enhance_audio(audio_path):
    whi("Cleaning voice")
    try:
        cleaned_sound = voice_cleaner.enhance_file(audio_path)

        # overwrites previous sound
        torchaudio.save(audio_path, cleaned_sound.unsqueeze(0).cpu(), 16000)

        whi("Done cleaning audio")
        return audio_path

    except Exception as err:
        red(f"Error when cleaning voice: '{err}'")
        return audio_path


def remove_silences(audio_path):
    whi("Removing silences")
    try:
        u = Unsilence(audio_path)
        u.detect_silence()

        # do it only if its worth it as it might degrade audio quality?
        estimated_time = u.estimate_time(audible_speed=1, silent_speed=2)  # Estimate time savings
        before = estimated_time["before"]["all"][0]
        after = estimated_time["after"]["all"][0]
        if after / before  > 0.9 and before - after < 5:
            whi(f"Not removing silence (orig: {before:.1f}s vs unsilenced: {after:.1f}s)")
            return audio_path

        yel(f"Removing silence: {before:.1f}s -> {after:.1f}s")
        u.render_media(audio_path, audible_speed=1, silent_speed=2, audio_only=True)
        whi("Done removing silences")
        return audio_path
    except Exception as err:
        red(f"Error when removing silences: '{err}'")
        return audio_path

def transcribe(audio_path, txt_whisp_prompt):
    whi("Transcribing audio")

    # try to remove silences
    audio_path = remove_silences(audio_path)

    # try to enhance quality
    audio_path = enhance_audio(audio_path)

    # save audio for next startup
    with open("./cache/voice_cards_last_audio.pickle", "wb") as f:
        pickle.dump(audio_path, f)
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


def check_source(source):
    "makes sure the source is only an img"
    whi("Checking source")
    if source:
        soup = BeautifulSoup(source, 'html.parser')
        imgs = soup.find_all("img")
        source = "</br>".join([str(x) for x in imgs])
        assert source, f"invalid source: {source}"
        # save for next startup
        with open("./cache/voice_cards_last_source.pickle", "wb") as f:
            pickle.dump(source, f)
    else:
        source = ""
    return source


def audio_to_anki(audio_path):
    whi("Sending audio to anki")
    try:
        with open(audio_path, "rb") as audio_file:
            audio_hash = hashlib.md5(audio_file.read()).hexdigest()
        shutil.copy(audio_path, anki_media / f"{audio_hash}.wav")
        html = f"</br>[sound:{audio_hash}.wav]"
        return html
    except Exception as err:
        return red(f"\n\nError when copying audio to anki media: '{err}'")


def auto_mode(*args, **kwargs):
    whi("Triggering auto mode")
    return main(*args, **kwargs, auto_mode=True)


def recur_improv(txt_audio, txt_whisp_prompt, txt_chatgpt_cloz, txt_context, output):
    whi("Recursively improving")
    global memorized_prompts
    if not txt_audio:
        return "No audio transcripts found.\n\n" + output
    if not txt_chatgpt_cloz:
        return "No chatgpt cloze found.\n\n" + output
    if "\n" in txt_chatgpt_cloz:
        whi("Replaced newlines in txt_chatgpt_cloz")
        txt_chatgpt_cloz = txt_chatgpt_cloz.replace("\n", "<br/>")

    try:
        assert len(memorized_prompts) % 2 == 1, "invalid length of new prompts before even updating it"
        to_add = [
                {
                    "role": "user",
                    "content": transcript_template.replace("CONTEXT", txt_context).replace("TRANSCRIPT", txt_audio),
                    "disabled": False,
                    },
                {
                    "role": "assistant",
                    "content": txt_chatgpt_cloz.replace("\n", "<br/>"),
                    "disabled": False,
                    }
                ]
        if to_add[0] in memorized_prompts:
            return f"Already present in previous outputs!\n\n{output}"
        if to_add[1] in memorized_prompts:
            return f"Already present in previous outputs!\n\n{output}"
        memorized_prompts.extend(to_add)

        memorized_prompts = curate_previous_prompts(memorized_prompts)

        assert len(memorized_prompts) % 2 == 1, "invalid length of new prompts"

        with open("audio_prompts.json", "w") as f:
            json.dump(memorized_prompts, f, indent=4)
    except Exception as err:
        return f"Error during recursive improvement: '{err}'\n\n{output}"
    return f"Recursively improved: {len(memorized_prompts)} total examples" + "\n\n" + output


def main(
        audio_path,
        txt_audio,
        txt_whisp_prompt,

        txt_chatgpt_resp,
        txt_chatgpt_cloz,

        txt_context,
        txt_deck,
        txt_tags,

        img_elem,
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

    if not ((img_elem is not None) or txt_source):
        to_return["output"] += red("you should probably specify either image+source or source")
    if (img_elem is not None) and not txt_source:
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
    if img_elem is not None:
        with open("./cache/voice_cards_last_image.pickle", "wb") as f:
            pickle.dump(img_elem, f)

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
    with open("./inferences/whisperToAnki_state.toml", "w") as f:
        rtoml.dump(
                {
                    "txt_deck": txt_deck,
                    "txt_tags": txt_tags,
                    "txt_context": txt_context,
                    "txt_whisp_prompt": txt_whisp_prompt,
                    }, f)


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
                img_btn = gr.Button(value="Load image from source or clipboard", variant="secondary")
                img_elem = gr.Image(value=prev_image, label="Source image")
                source_btn = gr.Button(value="Load source from image", variant="secondary")
                txt_source = gr.Textbox(value=get_img_source(prev_image), label="Source field", lines=1)
            with gr.Column():
                txt_deck = gr.Textbox(value=last_state["txt_deck"], label="Deck name", max_lines=1)
                txt_tag = gr.Textbox(value=last_state["txt_tags"], label="Tags", lines=1)
                txt_context = gr.Textbox(value=last_state["txt_context"], label="Contexte pour ChatGPT")
                txt_whisp_prompt = gr.Textbox(value=last_state["txt_whisp_prompt"], label="Contexte pour Whisper")

        with gr.Row():
            with gr.Column():
                audio_path = gr.Audio(source="microphone", type="filepath", label="Audio", format="wav", value=prev_audio)
                rst_btn = gr.Button(value="Reset audio", variant="secondary")
            with gr.Column():
                txt_audio = gr.Textbox(value=None, label="Audio transcript", lines=10, max_lines=10)
                txt_chatgpt_cloz = gr.Textbox(value=None, label="ChatGPT output", lines=10, max_lines=10)

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
        source_btn.click(fn=get_img_source, inputs=[img_elem], outputs=[txt_source])
        chatgpt_btn.click(fn=alfred, inputs=[txt_audio, txt_context], outputs=[txt_chatgpt_cloz, txt_chatgpt_resp])
        transcript_btn.click(fn=transcribe, inputs=[audio_path, txt_whisp_prompt], outputs=[txt_audio])
        img_btn.click(fn=get_image, inputs=[txt_source], outputs=[img_elem])
        source_btn.click(fn=get_img_source, inputs=[img_elem], outputs=[txt_source])
        rst_btn.click(fn=reset_audio, outputs=[audio_path])
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

                    img_elem,
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

                    img_elem,
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
