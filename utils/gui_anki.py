import gradio as gr

from .profiles import get_profiles, switch_profile, ValueStorage
from .main_anki import transcribe, alfred, to_anki, transcribe_cache_async, dirload_splitted, dirload_splitted_last, kill_threads, audio_edit
from .anki_utils import threaded_sync_anki, get_card_status

from .logger import get_log
from .memory import recur_improv, display_price, show_memories
from .media import get_image, reset_audio, reset_image, get_img_source
from .shared_module import shared

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
        audio_mp3 = gr.Microphone(type="filepath", label=f"Audio{i}", format="mp3", value=None, container=False)
        slots.append(audio_mp3)

    return slots

def reset_marked():
    return False

with gr.Blocks(
        analytics_enabled=False,
        title="VoiceToFormattedText - Anki",
        theme=theme,
        ) as demo_anki:

    with gr.Row():
        gr.HTML(value="<h1 style=\"text-align: center; color: purple;\">VoiceToFormattedText - Anki</h1>")
        dark_mode_btn = gr.Button("Dark Mode", variant="secondary", scale=0)
        sync_btn = gr.Button(value="Sync anki", variant="secondary", scale=0)
        kill_threads_btn = gr.Button(value="Kill threads", variant="secondary", scale=0)

    with gr.Tab(label="Main"):

        with gr.Row():

            with gr.Column(scale=1, min_width=50):

                # audio
                audio_number = shared.audio_slot_nb
                audio_slots = []
                for i in range(audio_number):
                    audio_mp3 = gr.Microphone(type="filepath", label=f"Audio{i}", format="mp3", value=None, container=False, show_download_button=False)
                    audio_slots.append(audio_mp3)
                with gr.Row():
                    rst_audio_btn = gr.Button(value="Clear audio", variant="primary", min_width=50)
                    dir_load_btn = gr.Button(value="Dirload 1+2", variant="secondary", min_width=50)

                # image
                with gr.Accordion(label="Images", open=True if shared.pv["gallery"] else False):
                    gallery = gr.Gallery(value=shared.pv["gallery"], label="Source images", columns=[1], rows=[2], object_fit="scale-down", height="auto", container=False, min_width=50)
                    with gr.Row():
                        rst_img_btn = gr.Button(value="Clear image", variant="secondary", min_width=50)
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
                    txt_card_done = gr.HTML(value="", label="Card status")

                # 1+2 / 1+2+3
                with gr.Row():
                    semiauto_btn = gr.Button(value="1+2. Speech to Cloze", variant="primary")
                    auto_btn = gr.Button(value="1+2+3. Autopilot", variant="primary")
                    audio_corrector = gr.Microphone(format="mp3", value=None, label="AudioEdit", show_share_button=False, type="filepath", show_download_button=False, min_length=1)

                # quick settings
                with gr.Row():
                    sld_max_tkn = gr.Slider(minimum=500, maximum=15000, value=shared.pv["sld_max_tkn"], step=500, label="LLM avail. tkn.", scale=1)
                    sld_whisp_temp = gr.Slider(minimum=0, maximum=1, value=shared.pv["sld_whisp_temp"], step=0.1, label="Whisper temp", scale=1)
                    sld_temp = gr.Slider(minimum=0, maximum=2, value=shared.pv["sld_temp"], step=0.1, label="LLM temp", scale=1)
                    sld_buffer = gr.Slider(minimum=0, maximum=shared.max_message_buffer, step=1, value=shared.pv["sld_buffer"], label="Buffer size", scale=1)
                    check_gpt4 = gr.Checkbox(value=shared.pv["check_gpt4"], interactive=True, label="Use GPT4?", show_label=True, scale=0)
                    check_marked = gr.Checkbox(value=False, interactive=True, label="Mark", show_label=True, scale=0)
                    txt_keywords = gr.Textbox(value=shared.pv["txt_keywords"], lines=3, max_lines=2, label="Keywords", placeholder="Comma separated regex that, if present in the transcript, increase chances of matching memories to be selected. Each regex is stripped, case insensitive and can be used multiple times to increase the effect.")
                txt_price = gr.Textbox(value=lambda: display_price(shared.pv["sld_max_tkn"], shared.pv["check_gpt4"]), show_label=False, interactive=False, max_lines=2, lines=2)

                with gr.Row():
                    sld_improve = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="Feedback priority", scale=5)
                    improve_btn = gr.Button(value="LLM Feedback", variant="secondary", scale=0)

    with gr.Tab(label="Settings"):
        roll_dirload_check = gr.Checkbox(value=shared.pv["dirload_check"], interactive=True, label="'Roll' from dirload", show_label=True, scale=0)
        with gr.Row():
            txt_profile = gr.Textbox(value=shared.pv.profile_name, placeholder=",".join(get_profiles()), label="Profile")
        with gr.Row():
            txt_deck = gr.Textbox(value=shared.pv["txt_deck"], label="Deck name", max_lines=1, placeholder="anki deck, e.g. Perso::Lessons")
            txt_whisp_lang = gr.Textbox(value=shared.pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr")
        txt_tags = gr.Textbox(value=shared.pv["txt_tags"], label="Tags", lines=2, placeholder="anki tags, e.g. science::math::geometry university_lectures::01")
        with gr.Row():
            txt_whisp_prompt = gr.Textbox(value=shared.pv["txt_whisp_prompt"], lines=2, label="SpeechToText context", placeholder="context for whisper")
            txt_chatgpt_context = gr.Textbox(value=shared.pv["txt_chatgpt_context"], lines=2, label="LLM context", placeholder="context for ChatGPT")

        # output
        output_elem = gr.Textbox(value=get_log, label="Logging", lines=10, max_lines=1000, every=1, interactive=False, placeholder="this string should never appear")

    with gr.Tab(label="Memories") as tab_memories:
        txt_memories = gr.Textbox(
                value="",
                label="Saved memories",
                lines=1000,
                max_lines=1000,
                interactive=False,
                placeholder="this string should never appear")

    # events
    tab_memories.select(
            fn=show_memories,
            inputs=[txt_profile],
            outputs=[txt_memories],
            )

    # darkmode
    dark_mode_btn.click(fn=None, js=darkmode_js)

    # sync anki
    sync_btn.click(fn=threaded_sync_anki, queue=True)

    # kill threads before timeout
    kill_threads_btn.click(fn=kill_threads)

    # display card status
    txt_chatgpt_cloz.change(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[txt_card_done],
            preprocess=False,
            postprocess=False,
            queue=True,
            every=1,
            trigger_mode="once",
            )

    # display pricing then save values
    sld_max_tkn.change(
            fn=display_price,
            inputs=[sld_max_tkn, check_gpt4],
            outputs=[txt_price],
            ).success(
                    fn=shared.pv.save_sld_max_tkn,
                    inputs=[sld_max_tkn],
                    )
    check_gpt4.change(
            fn=display_price,
            inputs=[sld_max_tkn, check_gpt4],
            outputs=[txt_price],
            ).success(
                    fn=shared.pv.save_check_gpt4,
                    inputs=[check_gpt4],
                    )

    # change some values to profile
    sld_whisp_temp.change(fn=shared.pv.save_sld_whisp_temp, inputs=[sld_whisp_temp])
    sld_buffer.change(fn=shared.pv.save_sld_buffer, inputs=[sld_buffer])
    sld_temp.change(fn=shared.pv.save_sld_temp, inputs=[sld_temp])
    roll_dirload_check.change(fn=shared.pv.save_dirload_check, inputs=[roll_dirload_check])
    txt_tags.change(fn=shared.pv.save_txt_tags, inputs=[txt_tags])
    txt_deck.change(fn=shared.pv.save_txt_deck, inputs=[txt_deck])
    txt_chatgpt_context.change(fn=shared.pv.save_txt_chatgpt_context, inputs=[txt_chatgpt_context])
    txt_whisp_prompt.change(fn=shared.pv.save_txt_whisp_prompt, inputs=[txt_whisp_prompt])
    txt_whisp_lang.change(fn=shared.pv.save_txt_whisp_lang, inputs=[txt_whisp_lang])
    txt_keywords.change(fn=shared.pv.save_txt_keywords, inputs=[txt_keywords])

    # change profile and load previous data
    txt_profile.submit(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_slots[0], txt_audio, txt_chatgpt_cloz, txt_profile])

    # load image then OCR it then save it to profile
    paste_image_event = img_btn.click(
            fn=get_image,
            inputs=[gallery],
            outputs=[gallery],
            queue=True).success(
                    fn=get_img_source,
                    inputs=[gallery],
                    queue=True,
                    ).success(
                            fn=shared.pv.save_gallery,
                            inputs=[gallery],
                            )
    rst_img_btn.click(
            fn=reset_image,
            outputs=[gallery],
            queue=True,
            cancels=[paste_image_event],
            )

    # audio
    audio_corrector.stop_recording(
            fn=audio_edit,
            inputs=[audio_corrector, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context],
            outputs=[txt_chatgpt_cloz, audio_corrector],
            )

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
            ).success(
                    fn=dirload_splitted_last,
                    inputs=[
                        roll_dirload_check,
                        ],
                    outputs=[audio_slots[-1]],
                    preprocess=False,
                    # postprocess=False,
                    queue=True,
                    ).success(
                            fn=transcribe,
                            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                            outputs=[txt_audio],
                            preprocess=False,
                            postprocess=False,
                            queue=True,
                            ).success(
                                fn=alfred,
                                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                                outputs=[txt_chatgpt_cloz],
                                queue=True,
                                )
    rollaudio2_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=True,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                        outputs=[txt_chatgpt_cloz],
                        preprocess=False,
                        postprocess=False,
                        queue=True,
                        ).success(
                            fn=to_anki,
                            inputs=[
                                audio_slots[0],
                                txt_audio,
                                txt_chatgpt_cloz,
                                txt_chatgpt_context,
                                txt_deck,
                                txt_tags,
                                gallery,
                                check_marked,
                                ],
                            preprocess=False,
                            postprocess=False,
                            queue=True,
                            ).success(
                                fn=dirload_splitted_last,
                                inputs=[
                                    roll_dirload_check,
                                    ],
                                outputs=[audio_slots[-1]],
                                preprocess=False,
                                # postprocess=False,
                                queue=True,
                                ).success(fn=reset_marked, outputs=[check_marked])

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use V2FT on the computer but record the audio
    # on another distance device
    dir_load_btn.click(
            fn=dirload_splitted,
            inputs=[roll_dirload_check] + audio_slots,
            outputs=audio_slots,
            queue=True,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                        outputs=[txt_chatgpt_cloz],
                        queue=True,
                        )

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
            outputs=[txt_chatgpt_cloz],
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
                txt_deck,
                txt_tags,
                gallery,
                check_marked,
                ],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).success(fn=reset_marked, outputs=[check_marked])

    # 1+2
    semiauto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).success(
                    fn=to_anki,
                    inputs=[
                        audio_slots[0],
                        txt_audio,
                        txt_chatgpt_cloz,
                        txt_chatgpt_context,
                        txt_deck,
                        txt_tags,
                        gallery,
                        check_marked,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).success(fn=reset_marked, outputs=[check_marked])

    # 1+2+3
    auto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=True,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
                queue=True,
                ).success(
                    fn=to_anki,
                    inputs=[
                        audio_slots[0],
                        txt_audio,
                        txt_chatgpt_cloz,
                        txt_chatgpt_context,
                        txt_deck,
                        txt_tags,
                        gallery,
                        check_marked,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=True,
                    ).then(fn=reset_marked, outputs=[check_marked])

    improve_btn.click(
            fn=recur_improv,
            inputs=[
                txt_profile,
                txt_audio,
                txt_whisp_prompt,
                txt_chatgpt_cloz,
                txt_chatgpt_context,
                sld_improve,
                check_gpt4,
                ],
            preprocess=False,
            postprocess=False,
            queue=True,
            )

    if shared.pv.profile_name == "default":
        gr.Warning("Enter a profile then press enter.")
