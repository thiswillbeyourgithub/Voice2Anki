import gradio as gr
from pathlib import Path
import pickle

from .profiles import get_profiles, switch_profile, ValueStorage, save_tags, save_deck
from .main_anki import transcribe, alfred, to_anki, transcribe_cache_async, load_splitted_audio, get_card_status
from .anki_utils import threaded_sync_anki

from .logger import get_log, whi, red
from .memory import recur_improv
from .media import get_image, reset_audio, reset_image, audio_saver, sound_preprocessing

theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="gray",
        text_size="sm",
        spacing_size="sm",
        radius_size="sm",
        font="ui-sans-serif",
        font_mono="ui-monospace",
        )
darkmode_js = """() => {
if (document.querySelectorAll('.dark').length) {
document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
} else {
document.querySelector('body').classList.add('dark');
}
}"""

# load default profile
Path("profiles").mkdir(exist_ok=True)
Path("profiles/anki").mkdir(exist_ok=True)
if Path("profiles/anki/latest_profile.pickle").exists():
    whi("Reloading previous profile.")
    with open("profiles/anki/latest_profile.pickle", "rb") as f:
        pv = ValueStorage(pickle.load(f))
else:
    red("Loading default profile")
    pv = ValueStorage("default")

with gr.Blocks(
        analytics_enabled=False,
        title="VoiceToFormattedText - Anki",
        theme=theme,
        ) as demo_anki:

    gr.Markdown("VoiceToFormattedText - Anki")

    # hidden, to store the request answer from chatgpt
    txt_chatgpt_tkncost = gr.Textbox(value=None, visible=False, placeholder="this string should never appear")

    with gr.Row():
        with gr.Row():
            with gr.Column(scale=1, min_width=100):
                gallery = gr.Gallery(value=pv["gallery"], label="Source images", columns=[2], rows=[1], object_fit="scale-down", height="auto", container=False)
                rst_img_btn = gr.Button(value="Clear then add", variant="secondary", size="sm")
                img_btn = gr.Button(value="Add image from clipboard", variant="primary", size="sm")
            with gr.Column(scale=3):
                with gr.Row():
                    with gr.Column(scale=10):
                        txt_profile = gr.Textbox(value=pv.profile_name, placeholder=",".join(get_profiles()), label="Profile", container=False)
                    with gr.Column(scale=1):
                         dark_mode_btn = gr.Button("Dark Mode", variant="secondary")
                txt_deck = gr.Textbox(value=pv["txt_deck"], label="Deck name", max_lines=1, placeholder="anki deck, e.g. Perso::Lessons", container=False)
                txt_tags = gr.Textbox(value=pv["txt_tags"], label="Tags", lines=1, placeholder="anki tags, e.g. science::math::geometry university_lectures::01", container=False)
                with gr.Row():
                    with gr.Column(scale=1, min_width=50):
                        txt_whisp_lang = gr.Textbox(value=pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr", container=False)
                    with gr.Column(scale=5):
                        with gr.Row():
                            txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], label="SpeechToText context", placeholder="context for whisper", container=False)
                            txt_chatgpt_context = gr.Textbox(value=pv["txt_chatgpt_context"], label="LLM context", placeholder="context for ChatGPT", container=False)
                with gr.Row():
                    rollaudio_btn = gr.Button(value="Roll + 1+2", variant="secondary")
                    rollaudio2_btn = gr.Button(value="Roll + 1+2+3", variant="secondary")
                    dir_load_btn = gr.Button(value="Dirload 1+2", variant="secondary")
                txt_audio = gr.Textbox(label="Transcript", lines=5, max_lines=10, placeholder="The transcript of the audio recording will appear here", container=False)
                txt_chatgpt_cloz = gr.Textbox(label="LLM cloze(s)", lines=5, max_lines=10, placeholder="The anki flashcard will appear here", container=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            rst_audio_btn = gr.Button(value="Clear audio", variant="secondary")
            audio_mp3_1 = gr.Audio(source="microphone", type="filepath", label="Audio1", format="mp3", value=None, container=False)
            audio_mp3_2 = gr.Audio(source="microphone", type="filepath", label="Audio2", format="mp3", value=None, container=False)
            audio_mp3_3 = gr.Audio(source="microphone", type="filepath", label="Audio3", format="mp3", value=None, container=False)
            audio_mp3_4 = gr.Audio(source="microphone", type="filepath", label="Audio4", format="mp3", value=None, container=False)
            audio_mp3_5 = gr.Audio(source="microphone", type="filepath", label="Audio5", format="mp3", value=None, container=False)

        with gr.Column(scale=1, min_width=250):
            transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary")
            chatgpt_btn = gr.Button(value="2. Transcript to cloze", variant="secondary")
            anki_btn = gr.Button(value="3. Cloze to Anki", variant="secondary")
            semiauto_btn = gr.Button(value="1+2. Speech to Cloze", variant="primary")
            auto_btn = gr.Button(value="1+2+3. Autopilot", variant="primary")

        with gr.Column(scale=1):
            improve_btn = gr.Button(value="Feed prompt back to LLM", variant="primary")
            sld_improve = gr.Slider(minimum=0, maximum=10, value=None, step=1, label="Feedback priority")
            sld_max_tkn = gr.Slider(minimum=500, maximum=15000, value=pv["sld_max_tkn"], step=500, label="LLM maxhistory token size.")
            sld_temp = gr.Slider(minimum=0, maximum=2, value=pv["temperature"], step=0.1, label="LLM temperature")
            with gr.Row():
                sync_btn = gr.Button(value="Sync anki", variant="primary")
                txt_card_done = gr.Textbox(value="", label="Card status", placeholder="Wether the card was already created", interactive=True, container=False)
                sld_buffer = gr.Slider(minimum=0, maximum=10, label="Buffer size", show_label=True, container=True, min_width=50)

    # output
    output_elem = gr.Textbox(value=get_log, label="Logging", lines=10, max_lines=100, every=1, interactive=False, placeholder="this string should never appear")
    txt_card_done.blur(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[txt_card_done],
            )


    # events
    # darkmode
    dark_mode_btn.click(fn=None, _js=darkmode_js)
    sync_btn.click(fn=threaded_sync_anki, queue=True)

    # change profile and load previous data
    txt_profile.submit(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_mp3_1, txt_audio, txt_chatgpt_cloz, txt_profile])
    # txt_profile.blur(
    #         fn=switch_profile,
    #         inputs=[txt_profile],
    #         outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_mp3_1, txt_audio, txt_chatgpt_cloz, txt_profile])
    txt_tags.submit(fn=save_tags, inputs=[txt_profile, txt_tags])
    txt_deck.submit(fn=save_deck, inputs=[txt_profile, txt_deck])

    # image
    img_btn.click(
            fn=get_image,
            inputs=[gallery],
            outputs=[gallery],
            queue=True)
    rst_img_btn.click(
            fn=reset_image,
            outputs=[gallery],
            queue=True,
            ).then(
                    fn=get_image,
                    inputs=[gallery],
                    outputs=[gallery],
                    queue=True)

    # audio

    # auto save audio
    asv = audio_saver(pv.profile_name)

    # trigger whisper in advance, this way the output will be cached
    aud_cache_event = []
    # the first slot will directly trigger 1+2 while the other slots will
    # just trigger caching

    # semi auto mode
    aud_cache_event.append(
        audio_mp3_1.stop_recording(
            fn=transcribe,
            inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_mp3_1,
                        txt_audio,
                        txt_chatgpt_cloz,
                        txt_chatgpt_context,
                        txt_chatgpt_tkncost,
                        txt_deck,
                        txt_tags,
                        txt_profile,
                        gallery,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    )
                    )
    aud_cache_event.append(
        audio_mp3_2.stop_recording(
            fn=transcribe_cache_async,
            inputs=[audio_mp3_2, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            preprocess=False,
            postprocess=False,
            queue=True).then(
                fn=asv.n2,
                inputs=[txt_profile, audio_mp3_2],
                preprocess=False,
                postprocess=False,
                queue=True))
    aud_cache_event.append(
        audio_mp3_3.stop_recording(
            fn=transcribe_cache_async,
            inputs=[audio_mp3_3, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            preprocess=False,
            postprocess=False,
            queue=True).then(
                fn=asv.n3,
                inputs=[txt_profile, audio_mp3_3],
                preprocess=False,
                postprocess=False,
                queue=True))
    aud_cache_event.append(
        audio_mp3_4.stop_recording(
            fn=transcribe_cache_async,
            inputs=[audio_mp3_4, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            preprocess=False,
            postprocess=False,
            queue=True).then(
                fn=asv.n4,
                inputs=[txt_profile, audio_mp3_4],
                preprocess=False,
                postprocess=False,
                queue=True))
    aud_cache_event.append(
        audio_mp3_5.stop_recording(
            fn=transcribe_cache_async,
            inputs=[audio_mp3_5, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            preprocess=False,
            postprocess=False,
            queue=True).then(
                fn=asv.n5,
                inputs=[txt_profile, audio_mp3_5],
                preprocess=False,
                postprocess=False,
                queue=True))

    audio_mp3_1.clear(cancels=[aud_cache_event[0]])
    audio_mp3_2.clear(cancels=[aud_cache_event[1]])
    audio_mp3_3.clear(cancels=[aud_cache_event[2]])
    audio_mp3_4.clear(cancels=[aud_cache_event[3]])
    audio_mp3_5.clear(cancels=[aud_cache_event[4]])

    rst_audio_btn.click(
            fn=reset_audio,
            outputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    rollaudio_btn.click(
            fn=asv.roll_audio,
            inputs=[txt_profile, audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            outputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        )
    rollaudio2_btn.click(
            fn=asv.roll_audio,
            inputs=[txt_profile, audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            outputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        ).then(
                            fn=to_anki,
                            inputs=[
                                audio_mp3_1,
                                txt_audio,
                                txt_chatgpt_cloz,
                                txt_chatgpt_context,
                                txt_chatgpt_tkncost,
                                txt_deck,
                                txt_tags,
                                txt_profile,
                                gallery,
                                ],
                            preprocess=False,
                            postprocess=False,
                            queue=True,
                            )

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use V2FT on the computer but record the audio
    # on another distance device
    dir_load_btn.click(
            fn=load_splitted_audio,
            inputs=[
                audio_mp3_1,
                audio_mp3_2,
                audio_mp3_3,
                audio_mp3_4,
                audio_mp3_5,
                txt_whisp_prompt,
                txt_whisp_lang,
                txt_profile,
                ],
            outputs=[
                audio_mp3_1,
                audio_mp3_2,
                audio_mp3_3,
                audio_mp3_4,
                audio_mp3_5,
                ],
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        )# .then(
                         #    fn=to_anki,
                         #    inputs=[
                         #        audio_mp3_1,
                         #        txt_audio,
                         #        txt_chatgpt_cloz,
                         #        txt_chatgpt_context,
                         #        txt_chatgpt_tkncost,
                         #        txt_deck,
                         #        txt_tags,
                         #        txt_profile,
                         #        gallery,
                         #        ],
                         #    preprocess=False,
                         #    postprocess=False,
                         #    queue=True,
                         #    )

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
            outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
            queue=True,
            )

    # send to anki
    anki_btn.click(
            fn=to_anki,
            inputs=[
                audio_mp3_1,
                txt_audio,
                txt_chatgpt_cloz,
                txt_chatgpt_context,
                txt_chatgpt_tkncost,
                txt_deck,
                txt_tags,
                txt_profile,
                gallery,
                ],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    # 1+2
    semiauto_btn.click(
            fn=transcribe,
            inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_mp3_1,
                        txt_audio,
                        txt_chatgpt_cloz,
                        txt_chatgpt_context,
                        txt_chatgpt_tkncost,
                        txt_deck,
                        txt_tags,
                        txt_profile,
                        gallery,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    )

    # 1+2+3
    auto_btn.click(
            fn=transcribe,
            inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_mp3_1,
                        txt_audio,
                        txt_chatgpt_cloz,
                        txt_chatgpt_context,
                        txt_chatgpt_tkncost,
                        txt_deck,
                        txt_tags,
                        txt_profile,
                        gallery,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
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
                ],
            preprocess=False,
            postprocess=False,
            queue=True,
            )
