import time
import gradio as gr

from .profiles import get_profiles, switch_profile
from .main_anki import transcribe, alfred, to_anki, dirload_splitted, dirload_splitted_last, kill_threads, audio_edit, flag_audio
from .anki_utils import threaded_sync_anki, get_card_status, mark_previous_note
from .logger import get_log
from .memory import recur_improv, display_price, show_memories
from .media import get_image, reset_audio, reset_gallery, get_img_source, ocr_image, load_future_galleries, create_audio_compo, roll_audio
from .shared_module import shared

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
if shared.compact_js:
    hide_some_components = """
    () => {
    const hideElements = (selector) => {
      const elements = document.querySelectorAll(selector);
      elements.forEach(el => {
        el.style.setProperty('display', 'none', 'important');
      });
    }
    try {
        hideElements('#Audio_component_V2FT > div.component-wrapper.svelte-7hmw24 > div.controls.svelte-nq0yvd > select');
    } catch(e) {};
    try {
        hideElements('#Audio_component_V2FT > div.component-wrapper.svelte-1n70sxb > div.controls.svelte-t8ovdf > div.control-wrapper.svelte-t8ovdf')
    } catch(e) {};
    try {
        hideElements('#Audio_component_V2FT > div.component-wrapper.svelte-1n70sxb > div.controls.svelte-t8ovdf > div.settings-wrapper.svelte-t8ovdf')
    } catch(e) {};
    }
    """
else:
    hide_some_components = None
css = """
.app.svelte-1kyws56.svelte-1kyws56 { max-width: 100%; }
"""


with gr.Blocks(
        analytics_enabled=False,
        title="VoiceToFormattedText - Anki",
        theme=theme,
        css=css,
        ) as demo_anki:

    with gr.Group():
        with gr.Row():
            gr.Button(value="VoiceToFormattedText - Anki", variant="primary", scale=3, interactive=True)
            dark_mode_btn = gr.Button("Dark Mode", variant="secondary", scale=0)
            sync_btn = gr.Button(value="Sync anki", variant="secondary", scale=0)
            update_status_btn = gr.Button(value="Card status", variant="secondary", scale=0, interactive=True)

    with gr.Tab(label="Main"):

        with gr.Row():

            with gr.Column(scale=1, min_width=50):

                # audio
                audio_number = shared.audio_slot_nb
                audio_slots = []
                with gr.Group():
                    for i in range(audio_number):
                        audio_mp3 = create_audio_compo()
                        audio_slots.append(audio_mp3)
                with gr.Row():
                    rst_audio_btn = gr.Button(value="Clear audio", variant="primary", min_width=50)
                    dir_load_btn = gr.Button(value="Dirload 1+2", variant="secondary", min_width=50)

                # image
                gallery = gr.Gallery(value=shared.pv["gallery"], label="Source images", columns=[1], rows=[1], object_fit="scale-down", height="20%", container=False, min_width=50)
                with gr.Group():
                    with gr.Row():
                        rst_img_btn = gr.Button(value="Clear image", variant="secondary", min_width=50)
                        img_btn = gr.Button(value="Add image from clipboard", variant="secondary", min_width=50)
                txt_extra_source = gr.Textbox(value=shared.pv["txt_extra_source"], label="Extra source", lines=1, placeholder="Will be added to the source.", visible=False)

            with gr.Column(scale=5):

                # whisper and chatgpt text output
                txt_audio = gr.Textbox(label="Transcript", lines=8, max_lines=100, placeholder="The transcript of the audio recording will appear here", container=False, interactive=True)
                txt_chatgpt_cloz = gr.Textbox(label="LLM cloze(s)", lines=15, max_lines=100, placeholder="The anki flashcard will appear here", container=False, interactive=True)

                # rolls
                with gr.Group():
                    with gr.Row():
                        small_sync_check = gr.Button("Status", variant="secondary", min_width=20, scale=1)
                        rollaudio_123_btn = gr.Button(value="Roll + 1+2+3", variant="primary", scale=5)
                        rollaudio_12_btn = gr.Button(value="Roll + 1+2", variant="primary", scale=5)

                # 1/2/3
                with gr.Group():
                    with gr.Row():
                        transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary")
                        chatgpt_btn = gr.Button(value="2. Transcript to cloze", variant="secondary")
                        anki_btn = gr.Button(value="3. Cloze to Anki", variant="secondary")

                # 1+2 / 1+2+3
                with gr.Accordion(open=False, label="Edit"):
                    with gr.Row():
                        audio_corrector = gr.Microphone(
                                format="mp3",
                                value=None,
                                label="AudioEdit via GPT-4",
                                show_download_button=True,
                                show_share_button=False,
                                type="filepath",
                                min_length=2,
                                container=True,
                                show_label=True,
                                #scale=1,
                                elem_id="Audio_component_V2FT",
                                elem_classes="Audio_component_V2FT",
                                #min_width=300,
                                editable=False,
                                )
                        auto_btn = gr.Button(value="1+2+3", variant="secondary", scale=1, min_width=50, visible=False)
                        semiauto_btn = gr.Button(
                                value="1+2",
                                variant="secondary",
                                #scale=3,
                                #min_width=50,
                                visible=False,
                                )

                with gr.Row():
                    mark_previous = gr.Button(value="Mark previous")
                    check_marked = gr.Checkbox(value=False, interactive=True, label="Mark next card", show_label=True)
                    sld_improve = gr.Number(minimum=0, maximum=10, value=5.0, step=1.0, label="Feedback priority")
                    improve_btn = gr.Button(value="LLM Feedback", variant="secondary")

                # quick settings
                with gr.Accordion(label="Quick settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=10):
                            with gr.Row():
                                sld_max_tkn = gr.Number(minimum=500, maximum=15000, value=shared.pv["sld_max_tkn"], step=100.0, label="LLM avail. tkn.", scale=1)
                                sld_whisp_temp = gr.Number(minimum=0, maximum=1, value=shared.pv["sld_whisp_temp"], step=0.1, label="Whisper temp", scale=1)
                                sld_temp = gr.Number(minimum=0, maximum=2, value=shared.pv["sld_temp"], step=0.1, label="LLM temp", scale=1)
                                sld_buffer = gr.Number(minimum=0, maximum=float(shared.max_message_buffer), step=1.0, value=shared.pv["sld_buffer"], label="Buffer size", scale=1)

                    with gr.Row():
                        check_gpt4 = gr.Checkbox(value=shared.pv["check_gpt4"], interactive=True, label="Use GPT4?", show_label=True, scale=0)
                        txt_price = gr.Textbox(value=lambda: display_price(shared.pv["sld_max_tkn"], shared.pv["check_gpt4"]), label="Price", interactive=False, max_lines=2, lines=2, scale=5)

                with gr.Row():
                    roll_gall_btn = gr.Button(value="Roll gallery", min_width=50)
                    flag_audio_btn = gr.Button(value="Flag audio", scale=0)

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
        with gr.Row():
            txt_keywords = gr.Textbox(value=shared.pv["txt_keywords"], lines=3, max_lines=2, label="Keywords", placeholder="Comma separated regex that, if present in the transcript, increase chances of matching memories to be selected. Each regex is stripped, case insensitive and can be used multiple times to increase the effect.")
        with gr.Row():
            if shared.memory_metric == "embeddings":
                sld_pick_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_pick_weight"], step=0.25, label="Embeddings weight")
            elif shared.memory_metric == "length":
                sld_pick_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_pick_weight"], step=0.25, label="Length weight")
            sld_prio_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_prio_weight"], step=0.25, label="Priority weight")
            sld_keywords_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_keywords_weight"], step=0.25, label="Keywords weight")
        with gr.Row():
            txt_openai_api_key = gr.Textbox(value=shared.pv["txt_openai_api_key"], label="OpenAI API key", lines=1)
        with gr.Row():
            kill_threads_btn = gr.Button(value="Kill threads", variant="secondary", scale=0)

    with gr.Tab(label="Logging") as tab_logging:
        output_elem = gr.Textbox(value=None, label="Logging", lines=100, max_lines=1000, interactive=False, placeholder="this string should never appear")

    with gr.Tab(label="Memories") as tab_memories:
        txt_memories = gr.Textbox(
                value="",
                label="Saved memories",
                lines=1000,
                max_lines=1000,
                interactive=False,
                placeholder="this string should never appear")

    with gr.Tab(label="Dirload Queue") as tab_dirload_queue:
        queue_df = gr.Dataframe(
                value=shared.dirload_queue,
                type="pandas",
                label="Dirload queue",
                interactive=False,
                )


    with gr.Tab(label="Future galleries") as tab_galleries:

        with gr.Row():
            with gr.Column():
                source_txt_btn = gr.Button("OCR the gallery")
                source_txt = gr.Textbox(value=None, interactive=False, lines=1, max_lines=20)
                source_txt_btn.click(
                        fn=ocr_image,
                        inputs=[gallery],
                        outputs=[source_txt],
                        queue=False,
                        preprocess=False,
                        postprocess=False,
                        )

        load_fg_btn = gr.Button(value="Load future galleries")

        future_galleries = []

        for fg in range(1, shared.future_gallery_slot_nb + 1):
            with gr.Row(equal_height=False):
                with gr.Column(scale=10):
                    gal_ = gr.Gallery(
                        value=None,
                        label=f"Gallery {fg}",
                        columns=[2],
                        rows=[1],
                        object_fit="scale-down",
                        #height=100,
                        #min_width=50,
                        container=True,
                        show_download_button=True,
                        preview=False,
                        )
                with gr.Column(scale=0):
                    send_ = gr.Button(value="Send to gallery", size="sm", variant="primary", min_width=50, scale=10)
                    add_ = gr.Button(value="Add image from clipboard", size="sm", min_width=50, scale=10)
                    with gr.Row():
                        rst_ = gr.Button(value="Clear", variant="primary", size="sm", min_width=50, scale=0)
                        ocr_ = gr.Button("OCR", variant="secondary", size="sm", scale=1)

            # add image
            add_.click(
                    fn=get_image,
                    inputs=[gal_],
                    outputs=[gal_],
                    queue=False).success(
                            fn=getattr(shared.pv, f"save_future_gallery_{fg:03d}"),
                            inputs=[gal_],
                            )
            # send image
            send_.click(
                    fn=lambda x: x,
                    inputs=[gal_],
                    outputs=[gallery],
                    preprocess=False,
                    postprocess=False,
                    queue=False).success(
                            fn=shared.pv.save_gallery,
                            inputs=[gallery]
                            ).then(
                                    fn=get_img_source,
                                    inputs=[gallery],
                                    queue=False,
                                    )
            # reset image
            rst_.click(
                    fn=lambda: None,
                    outputs=[gal_],
                    queue=False).then(
                        fn=getattr(shared.pv, f"save_future_gallery_{fg:03d}"),
                        inputs=[gal_],
                        )
            ocr_.click(
                    fn=ocr_image,
                    inputs=[gal_],
                    outputs=[source_txt],
                    queue=False,
                    preprocess=False,
                    postprocess=False,
                    )
            future_galleries.append([rst_, gal_, send_, add_, ocr_])

        clear_fg_btn = gr.ClearButton(
                components=[elem[1] for elem in future_galleries],
                value="Empty all (not saved)",
                variant="primary",
                )

        load_fg_btn.click(
                fn=load_future_galleries,
                outputs=[row[1] for row in future_galleries],
                )
        roll_gall_btn.click(
                fn=load_future_galleries,
                outputs=[row[1] for row in future_galleries],
                ).then(
                        fn=lambda x: x,
                        inputs=[future_galleries[1][1]],
                        outputs=[gallery],
                        preprocess=False,
                        postprocess=False,
                        queue=False,
                        ).success(
                            fn=shared.pv.save_gallery,
                            inputs=[gallery]
                            ).success(
                                    fn=get_img_source,
                                    inputs=[gallery],
                                    queue=False,
                                    ).success(
                                            fn=lambda: None,
                                            outputs=[future_galleries[0][1]]
                                            ).success(
                                                    fn=shared.pv.save_future_gallery_001,
                                                    inputs=[future_galleries[0][1]],
                                                    )

    # with gr.Tab(label="Files") as tab_files:
    #     with gr.Accordion(label="Done", open=False):
    #         fex_done = gr.FileExplorer(
    #                 root="user_directory/done",
    #                 label="Done",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Splitted", open=False):
    #         fex_splitted = gr.FileExplorer(
    #                 root="user_directory/splitted",
    #                 label="Splitted",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Unsplitted", open=False):
    #         fex_unsplitted = gr.FileExplorer(
    #                 root="user_directory/unsplitted",
    #                 label="Unsplitted",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Flagged", open=False):
    #         fex_flagged = gr.FileExplorer(
    #                 root="user_directory/flagged",
    #                 label="Flagged",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )

    # events ############################################################

    # reload df
    tab_dirload_queue.select(
            fn=lambda: shared.dirload_queue.sort_index().reset_index().set_index("n").reset_index(),
            outputs=[queue_df],
            )
    # copy audio to flag button
    flag_audio_btn.click(
            fn=flag_audio,
            inputs=[txt_audio, txt_chatgpt_cloz, txt_chatgpt_context],
            show_progress=False,
            )
    # trigger transcription when first audio stops recording
    audio_slots[0].stop_recording(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=False,
            ).success(
                    fn=alfred,
                    inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                    outputs=[txt_chatgpt_cloz],
                    queue=False,
                    preprocess=False,
                    postprocess=False,
                    )

    # load output elem if clicked
    tab_logging.select(fn=get_log, outputs=[output_elem])

    # load memories only if clickes
    tab_memories.select(
            fn=show_memories,
            inputs=[txt_profile],
            outputs=[txt_memories],
            show_progress=False,
            )

    # mark the previous card
    mark_previous.click(
            fn=mark_previous_note,
            show_progress=False,
            )

    # darkmode
    dark_mode_btn.click(fn=None, js=darkmode_js, show_progress=False)

    # sync anki
    sync_btn.click(fn=threaded_sync_anki, queue=False, show_progress=False)

    # kill threads before timeout
    kill_threads_btn.click(fn=kill_threads, show_progress=False)

    # display card status
    update_status_btn.click(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[update_status_btn],
            queue=False,
            # show_progress=True,
            # every=2,
            # trigger_mode="once",
            preprocess=False,
            postprocess=False,
            )
    small_sync_check.click(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[update_status_btn],
            queue=False,
            preprocess=False,
            postprocess=False,
            )
    txt_chatgpt_cloz.change(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[update_status_btn],
            queue=False,
            # show_progress=True,
            # every=2,
            # trigger_mode="once",
            preprocess=False,
            postprocess=False,
            )


    # display pricing then save values
    sld_max_tkn.change(
            fn=display_price,
            inputs=[sld_max_tkn, check_gpt4],
            outputs=[txt_price],
            show_progress=False,
            ).then(
                    fn=shared.pv.save_sld_max_tkn,
                    inputs=[sld_max_tkn],
                    show_progress=False,
                    )
    check_gpt4.change(
            fn=display_price,
            inputs=[sld_max_tkn, check_gpt4],
            outputs=[txt_price],
            show_progress=False,
            ).success(
                    fn=shared.pv.save_check_gpt4,
                    inputs=[check_gpt4],
                    show_progress=False,
                    )

    # change some values to profile
    sld_whisp_temp.change(fn=shared.pv.save_sld_whisp_temp, inputs=[sld_whisp_temp], show_progress=False)
    sld_buffer.change(fn=shared.pv.save_sld_buffer, inputs=[sld_buffer], show_progress=False)
    sld_temp.change(fn=shared.pv.save_sld_temp, inputs=[sld_temp], show_progress=False)
    roll_dirload_check.change(fn=shared.pv.save_dirload_check, inputs=[roll_dirload_check], show_progress=False)
    txt_tags.change(fn=shared.pv.save_txt_tags, inputs=[txt_tags], show_progress=False)
    txt_deck.change(fn=shared.pv.save_txt_deck, inputs=[txt_deck], show_progress=False)
    txt_chatgpt_context.change(fn=shared.pv.save_txt_chatgpt_context, inputs=[txt_chatgpt_context], show_progress=False)
    txt_whisp_prompt.change(fn=shared.pv.save_txt_whisp_prompt, inputs=[txt_whisp_prompt], show_progress=False)
    txt_whisp_lang.change(fn=shared.pv.save_txt_whisp_lang, inputs=[txt_whisp_lang], show_progress=False)
    txt_keywords.change(fn=shared.pv.save_txt_keywords, inputs=[txt_keywords], show_progress=False)
    sld_pick_weight.change(fn=shared.pv.save_sld_pick_weight, inputs=[sld_pick_weight], show_progress=False)
    sld_prio_weight.change(fn=shared.pv.save_sld_prio_weight, inputs=[sld_prio_weight], show_progress=False)
    sld_keywords_weight.change(fn=shared.pv.save_sld_keywords_weight, inputs=[sld_keywords_weight], show_progress=False)
    txt_extra_source.change(fn=shared.pv.save_txt_extra_source, inputs=[txt_extra_source], show_progress=False)
    txt_openai_api_key.change(fn=shared.pv.save_txt_openai_api_key, inputs=[txt_openai_api_key], show_progress=False)

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
            show_progress=False,
            queue=False,
            ).success(
                    fn=shared.pv.save_gallery,
                    inputs=[gallery],
                    show_progress=False,
                    ).success(
                            fn=get_img_source,
                            inputs=[gallery],
                            queue=False,
                            show_progress=False,
                            )
    rst_img_btn.click(
            fn=reset_gallery,
            outputs=[gallery],
            queue=False,
            show_progress=False,
            )

    # audio
    audio_corrector.stop_recording(
            fn=audio_edit,
            inputs=[audio_corrector, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context],
            outputs=[txt_chatgpt_cloz, audio_corrector],
            show_progress=False,
            )

    rst_audio_btn.click(
            fn=reset_audio,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            ).then(
                    fn=None,
                    js=hide_some_components,
                    show_progress=False,
                    )

    rollaudio_12_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                        outputs=[txt_chatgpt_cloz],
                        queue=False,
                        preprocess=False,
                        postprocess=False,
                        ).then(
                                fn=dirload_splitted_last,
                                inputs=[
                                    roll_dirload_check,
                                    txt_whisp_prompt,
                                    txt_whisp_lang,
                                    sld_whisp_temp,

                                    txt_chatgpt_context,
                                    txt_profile,
                                    sld_max_tkn,
                                    sld_temp,
                                    sld_buffer,
                                    check_gpt4,
                                    txt_keywords,
                                    ],
                                outputs=[audio_slots[-1]],
                                # preprocess=False,
                                # postprocess=False,
                                queue=False,
                                show_progress=False,
                                ).then(
                                        fn=None,
                                        js=hide_some_components,
                                        queue=False,
                                        show_progress=False,
                                        ).then(
                                                fn=get_card_status,
                                                inputs=[txt_chatgpt_cloz],
                                                outputs=[update_status_btn],
                                                queue=True,
                                                preprocess=False,
                                                postprocess=False,
                                                )
    rollaudio_123_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                        outputs=[txt_chatgpt_cloz],
                        preprocess=False,
                        postprocess=False,
                        queue=False,
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
                                txt_extra_source,
                                ],
                            preprocess=False,
                            postprocess=False,
                            queue=False,
                            ).then(
                                fn=dirload_splitted_last,
                                inputs=[
                                    roll_dirload_check,
                                    txt_whisp_prompt,
                                    txt_whisp_lang,
                                    sld_whisp_temp,

                                    txt_chatgpt_context,
                                    txt_profile,
                                    sld_max_tkn,
                                    sld_temp,
                                    sld_buffer,
                                    check_gpt4,
                                    txt_keywords,
                                    ],
                                outputs=[audio_slots[-1]],
                                # preprocess=False,
                                # postprocess=False,
                                queue=False,
                                show_progress=False,
                                ).success(
                                        fn=lambda: False,
                                        outputs=[check_marked],
                                        show_progress=False,
                                        ).then(
                                                fn=None,
                                                js=hide_some_components,
                                                queue=False,
                                                show_progress=False,
                                                ).then(
                                                        fn=get_card_status,
                                                        inputs=[txt_chatgpt_cloz],
                                                        outputs=[update_status_btn],
                                                        queue=True,
                                                        preprocess=False,
                                                        postprocess=False,
                                                        )

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use V2FT on the computer but record the audio
    # on another distance device
    dir_load_btn.click(
            fn=dirload_splitted,
            inputs=[
                roll_dirload_check,
                txt_whisp_prompt,
                txt_whisp_lang,
                sld_whisp_temp,

                txt_chatgpt_context,
                txt_profile,
                sld_max_tkn,
                sld_temp,
                sld_buffer,
                check_gpt4,
                txt_keywords,
                ] + audio_slots,
            outputs=audio_slots,
            queue=False,
            show_progress=False,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                        outputs=[txt_chatgpt_cloz],
                        queue=False,
                        preprocess=False,
                        postprocess=False,
                        ).then(
                                fn=None,
                                js=hide_some_components,
                                queue=False,
                                show_progress=False,
                                )

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=False,
            )

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
            outputs=[txt_chatgpt_cloz],
            queue=False,
            preprocess=False,
            postprocess=False,
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
                txt_extra_source,
                ],
            preprocess=False,
            postprocess=False,
            queue=False,
            ).then(
                    fn=lambda: False,
                    outputs=[check_marked],
                    show_progress=False,
                    )

    # 1+2
    semiauto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=False,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
                queue=False,
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
                        txt_extra_source,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).then(
                            fn=lambda: False,
                            outputs=[check_marked],
                            show_progress=False,
                            )

    # 1+2+3
    auto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            queue=False,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, check_gpt4, txt_keywords],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
                queue=False,
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
                        txt_extra_source,
                        ],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).then(
                            fn=lambda: False,
                            outputs=[check_marked],
                            show_progress=False,
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
                check_gpt4,
                ],
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            )
    gr.on(
            triggers=[
                # rollaudio_12_btn.click,
                # rollaudio_123_btn.click,
                dark_mode_btn.click,
                sync_btn.click,
                ],
            js=hide_some_components,
            fn=None,
            queue=False,
            show_progress=False,
            )

    demo_anki.load(
            fn=shared.reset,
            js=hide_some_components,
            show_progress=False,
            )
    if shared.pv.profile_name == "default":
        gr.Warning("Enter a profile then press enter.")
