import gradio as gr

from .profiles import get_profiles, switch_profile, save_tags, save_deck, save_buffer
from .main_anki import transcribe, alfred, to_anki, transcribe_cache_async, dirload_splitted, dirload_splitted_last, pv
from .anki_utils import threaded_sync_anki, get_card_status

from .logger import get_log
from .memory import recur_improv
from .media import get_image, reset_audio, reset_image, get_img_source

theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="gray",
        text_size="lg",
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

def roll_audio(*slots):
    assert len(slots) > 1, f"invalid number of audio slots: {len(slots)}"
    slots = list(slots)
    if all((slot is None for slot in slots)):
           return slots
    if all((slot is None for slot in slots[1:])):
           return slots
    slots[0] = None
    while slots[0] is None:
        slots.pop(0)
        audio_mp3 = gr.Audio(source="microphone", type="filepath", label=f"Audio{i}", format="mp3", value=None, container=False)
        slots.append(audio_mp3)
    return slots


with gr.Blocks(
        analytics_enabled=False,
        title="VoiceToFormattedText - Anki",
        theme=theme,
        ) as demo_anki:

    with gr.Row():
        gr.HTML(value="<h1 style=\"text-align: center; color: purple;\">VoiceToFormattedText - Anki</h1>", container=False, scale=5)
        dark_mode_btn = gr.Button("Dark Mode", variant="secondary", scale=0)
        sync_btn = gr.Button(value="Sync anki", variant="secondary", scale=0)

    # hidden, to store the request answer from chatgpt
    txt_chatgpt_tkncost = gr.Textbox(value=None, visible=False, placeholder="this string should never appear")

    with gr.Tab(label="Main"):

        with gr.Row(label="Main"):

            with gr.Column(scale=1, min_width=50):

                # audio
                audio_number = 5
                audio_slots = []
                for i in range(audio_number):
                    audio_mp3 = gr.Audio(source="microphone", type="filepath", label=f"Audio{i}", format="mp3", value=None, container=False)
                    audio_slots.append(audio_mp3)
                with gr.Row():
                    rst_audio_btn = gr.Button(value="Clear audio", variant="primary", min_width=50)
                    dir_load_btn = gr.Button(value="Dirload 1+2", variant="secondary", min_width=50)

                # image
                with gr.Accordion(label="Images", open=True if pv["gallery"] else False):
                    gallery = gr.Gallery(value=pv["gallery"], label="Source images", columns=[1], rows=[2], object_fit="scale-down", height="auto", container=False, min_width=50)
                    with gr.Row():
                        rst_img_btn = gr.Button(value="Clear image then add", variant="secondary", min_width=50)
                        img_btn = gr.Button(value="Add image from clipboard", variant="secondary", min_width=50)

            with gr.Column(scale=3):

                # whisper and chatgpt text output
                txt_audio = gr.Textbox(label="Transcript", lines=10, max_lines=100, placeholder="The transcript of the audio recording will appear here", container=False)
                txt_chatgpt_cloz = gr.Textbox(label="LLM cloze(s)", lines=10, max_lines=100, placeholder="The anki flashcard will appear here", container=False)

                # rolls
                with gr.Row():
                    rollaudio_btn = gr.Button(value="Roll + 1+2", variant="secondary", scale=4)
                    rollaudio2_btn = gr.Button(value="Roll + 1+2+3", variant="secondary", scale=4)

                # 1/2/3
                with gr.Row():
                    transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary")
                    chatgpt_btn = gr.Button(value="2. Transcript to cloze", variant="secondary")
                    anki_btn = gr.Button(value="3. Cloze to Anki", variant="secondary")
                    txt_card_done = gr.HTML(value="", label="Card status", placeholder="Wether the card was already created", container=False)

                # 1+2 / 1+2+3
                with gr.Row():
                    semiauto_btn = gr.Button(value="1+2. Speech to Cloze", variant="primary")
                    auto_btn = gr.Button(value="1+2+3. Autopilot", variant="primary")

                # quick settings
                with gr.Row():
                    sld_max_tkn = gr.Slider(minimum=500, maximum=15000, value=pv["sld_max_tkn"], step=500, label="LLM avail. tkn.")
                    sld_temp = gr.Slider(minimum=0, maximum=2, value=pv["temperature"], step=0.1, label="LLM temperature")
                    sld_buffer = gr.Slider(minimum=0, maximum=10, step=1, value=pv["sld_buffer"], label="Buffer size")
                    check_gpt4 = gr.Checkbox(value=pv["gpt4_checkbox"], interactive=True, label="Use GPT4?", show_label=True, scale=0)

    with gr.Tab(label="Settings"):
        roll_dirload_check = gr.Checkbox(value=pv["dirload_check"], interactive=True, label="'Roll' from dirload", show_label=True, scale=0)
        with gr.Row():
            sld_improve = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="Feedback priority", scale=5)
            improve_btn = gr.Button(value="LLM Feedback", variant="secondary", scale=0)
        with gr.Row():
            txt_profile = gr.Textbox(value=pv.profile_name, placeholder=",".join(get_profiles()), label="Profile")
        with gr.Row():
            txt_deck = gr.Textbox(value=pv["txt_deck"], label="Deck name", max_lines=1, placeholder="anki deck, e.g. Perso::Lessons")
            txt_whisp_lang = gr.Textbox(value=pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr")
        txt_tags = gr.Textbox(value=pv["txt_tags"], label="Tags", lines=2, placeholder="anki tags, e.g. science::math::geometry university_lectures::01")
        with gr.Row():
            txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], lines=2, label="SpeechToText context", placeholder="context for whisper")
            txt_chatgpt_context = gr.Textbox(value=pv["txt_chatgpt_context"], lines=2, label="LLM context", placeholder="context for ChatGPT")

        # output
        output_elem = gr.Textbox(value=get_log, label="Logging", lines=10, max_lines=1000, every=1, interactive=False, placeholder="this string should never appear")

    # events

    # darkmode
    dark_mode_btn.click(fn=None, _js=darkmode_js)

    # sync anki
    sync_btn.click(fn=threaded_sync_anki, queue=True)

    # display card status
    txt_chatgpt_cloz.change(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[txt_card_done],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    # change message buffer size
    sld_buffer.change(fn=save_buffer, inputs=[txt_profile, sld_buffer])

    # change profile and load previous data
    txt_profile.submit(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_slots[0], txt_audio, txt_chatgpt_cloz, txt_profile])
    # txt_profile.blur(
    #         fn=switch_profile,
    #         inputs=[txt_profile],
    #         outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_slots[0], txt_audio, txt_chatgpt_cloz, txt_profile])
    txt_tags.submit(fn=save_tags, inputs=[txt_profile, txt_tags])
    txt_deck.submit(fn=save_deck, inputs=[txt_profile, txt_deck])

    # image
    paste_image_event = img_btn.click(
            fn=get_image,
            inputs=[gallery],
            outputs=[gallery],
            queue=True).then(
                    fn=get_img_source,
                    inputs=[gallery],
                    queue=True,
                    )
    rst_img_btn.click(
            fn=reset_image,
            outputs=[gallery],
            queue=True,
            cancels=[paste_image_event],
            ).then(
                    fn=get_image,
                    inputs=[gallery],
                    outputs=[gallery],
                    queue=True).then(
                            fn=get_img_source,
                            inputs=[gallery],
                            queue=True,
                            )

    # audio

    # trigger whisper in advance, this way the output will be cached
    aud_cache_event = []
    # the first slot will directly trigger 1+2 while the other slots will
    # just trigger caching

    # semi auto mode
    aud_cache_event.append(
        audio_slots[0].change(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_slots[0],
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
    for audio_slot in audio_slots[1:]:
        aud_cache_event.append(
            audio_slot.change(
                fn=transcribe_cache_async,
                inputs=[audio_slot, txt_whisp_prompt, txt_whisp_lang],
                preprocess=False,
                postprocess=False,
                queue=True))

    rst_audio_btn.click(
            fn=reset_audio,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    rollaudio_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        ).then(
                                fn=dirload_splitted_last,
                                inputs=[
                                    roll_dirload_check,
                                    ],
                                outputs=[audio_slots[-1]],
                                preprocess=False,
                                # postprocess=False,
                                queue=True,
                                ).then(
                                        fn=get_card_status,
                                        inputs=[txt_chatgpt_cloz],
                                        outputs=[txt_card_done],
                                        preprocess=False,
                                        postprocess=False,
                                        queue=True,
                                        )
    rollaudio2_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        ).then(
                            fn=to_anki,
                            inputs=[
                                audio_slots[0],
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
                            ).then(
                                fn=dirload_splitted_last,
                                inputs=[
                                    roll_dirload_check,
                                    ],
                                outputs=[audio_slots[-1]],
                                preprocess=False,
                                # postprocess=False,
                                queue=True,
                                ).then(
                                        fn=get_card_status,
                                        inputs=[txt_chatgpt_cloz],
                                        outputs=[txt_card_done],
                                        preprocess=False,
                                        postprocess=False,
                                        queue=True,
                                        )

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use V2FT on the computer but record the audio
    # on another distance device
    dir_load_btn.click(
            fn=dirload_splitted,
            inputs=[roll_dirload_check] + audio_slots,
            outputs=audio_slots,
            queue=True,
            ).then(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                        outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        ).then(
                                fn=get_card_status,
                                inputs=[txt_chatgpt_cloz],
                                outputs=[txt_card_done],
                                preprocess=False,
                                postprocess=False,
                                queue=True,
                                )
                                # ).then(
                                #    fn=to_anki,
                                #    inputs=[
                                #        audio_slots[0],
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
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
            outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
            queue=True,
            )

    # send to anki
    anki_btn.click(
            fn=to_anki,
            inputs=[
                audio_slots[0],
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
            ).then(
                    fn=get_card_status,
                    inputs=[txt_chatgpt_cloz],
                    outputs=[txt_card_done],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    )

    # 1+2
    semiauto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_slots[0],
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
                    ).then(
                            fn=get_card_status,
                            inputs=[txt_chatgpt_cloz],
                            outputs=[txt_card_done],
                            preprocess=False,
                            postprocess=False,
                            queue=True,
                            )

    # 1+2+3
    auto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).then(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4],
                outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).then(
                    fn=to_anki,
                    inputs=[
                        audio_slots[0],
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
                    ).then(
                            fn=get_card_status,
                            inputs=[txt_chatgpt_cloz],
                            outputs=[txt_card_done],
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

    if pv.profile_name == "default":
        gr.Warning("Enter a profile then press enter.")
