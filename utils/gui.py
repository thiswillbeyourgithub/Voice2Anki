from datetime import datetime
import gradio as gr

from .profiles import get_profiles, switch_profile, load_user_functions, load_user_chain, call_user_chain
from .main import transcribe, alfred, to_anki, dirload_splitted, dirload_splitted_last, kill_threads, audio_edit, flag_audio, pop_buffer, clear_llm_cache
from .anki_utils import sync_anki, get_card_status, mark_previous_note, get_anki_tags, get_decks
from .logger import get_log, red
from .memory import recur_improv, display_price, get_memories_df, get_message_buffer_df, get_dirload_df
from .media import get_image, reset_audio, reset_gallery, get_img_source, ocr_image, roll_future_galleries, create_audio_compo, roll_audio, force_sound_processing, update_audio_slots_txts
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

shortcut_js = """
<script>
function shortcuts(e) {
    // the switch is so that keypress are ignored if an input element is
    // in focus
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {
        case "input":
        case "textarea":
        //case "select":
        //case "button":
        break;

        default:
        if (e.code == "KeyS" && e.shiftKey) {
            document.getElementById("syncankibtn").click();
        }
        else if (e.code == "KeyN" && e.shiftKey) {
            document.getElementById("darkmodebtn").click();
        }
        else if (e.key == "s") {
            document.getElementById("cardstatusbtn").click();
        }
        else if (e.key == "f") {
            document.getElementById("llmfeedbackbtn").click();
        }
        else if (e.key == "m") {
            document.getElementById("markpreviousbtn").click();
        }
        else if (e.key == "1" && e.shiftKey) {
            document.getElementById("transcribebtn").click();
        }
        else if (e.key == "2" && e.shiftKey) {
            document.getElementById("transcriptbtn").click();
        }
        else if (e.key == "3" && e.shiftKey) {
            document.getElementById("toankibtn").click();
        }
        else {
            alert(`Unrecognized shortcut: ${e.key}`);
            }

        }
}
document.addEventListener('keypress', shortcuts, false);
</script>
"""

css = """
/* Make tabs take all the width */
#BigTabV2A-button { flex-grow:1 !important; }

/* remove source selector */
#Audio_component_Voice2Anki > div.component-wrapper > div.controls > select {display: none !important; flex-grow:0 !important;}

/* remove volume and speed controls */
#Audio_component_Voice2Anki > div.component-wrapper > div.controls > div.control-wrapper {display: none !important; flex-grow:0 !important;}


/* remove clipping controls
#Audio_component_Voice2Anki > div.component-wrapper > div.controls > div.settings-wrapper {display: none !important; flex-grow:0 !important;}
*/
""".strip()

if shared.widen_screen:
    css += "\n.app { max-width: 100% !important; }"

with gr.Blocks(
        analytics_enabled=False,
        title=f"Voice2Anki V{shared.VERSION}",
        theme=theme,
        css=css,
        head=shortcut_js,
        ) as demo:

    with gr.Row():
        gr.Button(value=f"Voice2Anki V{shared.VERSION}", variant="primary", scale=3, interactive=True)
        dark_mode_btn = gr.Button("Dark/Light", variant="secondary", scale=0, elem_id="darkmodebtn")
        sync_btn = gr.Button(value="Sync anki", variant="secondary", scale=0, elem_id="syncankibtn")

    with gr.Tab(label="Main", elem_id="BigTabV2A"):

        with gr.Row():
            rst_audio_btn = gr.Button(value="Clear audio", variant="primary", min_width=50, scale=1)
            dir_load_btn = gr.Button(value="Dirload", variant="secondary", min_width=50, scale=5)

        # audio
        audio_number = shared.audio_slot_nb
        audio_slots = []
        audio_slots_txts = []
        for i in range(audio_number):
                audio_mp3 = create_audio_compo(scale=1, label=f"Audio #{i+1}", show_label=True, render=False)
                audio_slots.append(audio_mp3)
                audio_slots_txt = gr.Textbox(lines=8, max_lines=10, container=False, interactive=False, scale=2, render=False)
                audio_slots_txts.append(audio_slots_txt)
        # rendering afterwards to reverse the order
        for aud, t in zip(audio_slots[::-1], audio_slots_txts[::-1]):
            with gr.Row():
                aud.render()
                t.render()

        # whisper and chatgpt text output
        txt_audio = gr.Textbox(label="Transcript", lines=8, max_lines=100, placeholder="The transcript of the audio recording will appear here", container=False, interactive=True)
        txt_chatgpt_cloz = gr.Textbox(label="LLM cloze(s)", lines=15, max_lines=100, placeholder="The anki flashcard will appear here", container=False, interactive=True)

        # rolls
        with gr.Group():
            with gr.Row():
                rollaudio_123_btn = gr.Button(value="Roll + 1+2+3", variant="primary", scale=5)
                rollaudio_12_btn = gr.Button(value="Roll + 1+2", variant="primary", scale=5)
                rollaudio_1_btn = gr.Button(value="Roll + 1", variant="primary", scale=5)
                update_status_btn = gr.Button(value="Card status", variant="secondary", scale=0, interactive=True, elem_id="cardstatusbtn")

        # 1/2/3
        with gr.Group():
            with gr.Row():
                transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary", elem_id="transcribebtn")
                chatgpt_btn = gr.Button(value="2. Transcript to cloze", variant="secondary", elem_id="transcriptbtn")
                anki_btn = gr.Button(value="3. Cloze to Anki", variant="secondary", elem_id="toankibtn")

        with gr.Row():
            mark_previous = gr.Button(value="Mark previous", elem_id="markpreviousbtn")
            check_marked = gr.Checkbox(value=False, interactive=True, label="Mark next card", show_label=True)

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
                        scale=0,
                        elem_id="Audio_component_Voice2Anki",
                        #min_width=300,
                        editable=False,
                        )
                audio_corrector_txt = gr.Textbox(value=None, label="Edit via GPT-4", scale=2)
                auto_btn = gr.Button(value="1+2+3", variant="secondary", scale=1, min_width=50, visible=False)
                semiauto_btn = gr.Button(
                        value="1+2",
                        variant="secondary",
                        #scale=3,
                        #min_width=50,
                        visible=False,
                        )

        # quick settings
        with gr.Accordion(label="Quick settings", open=True):
            with gr.Row():
                sld_improve = gr.Number(minimum=0, maximum=10, value=5.0, step=1.0, label="Feedback priority")
                improve_btn = gr.Button(value="LLM Feedback", variant="secondary", elem_id="llmfeedbackbtn")
                prompt_manag = gr.Radio(choices=["messages", "stuff"], value=shared.pv["prompt_management"], interactive=True, label="Prompt management", show_label=True)
            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        sld_max_tkn = gr.Number(minimum=500, maximum=15000, value=shared.pv["sld_max_tkn"], step=100.0, label="LLM avail. tkn.", scale=1)
                        sld_whisp_temp = gr.Number(minimum=0, maximum=1, value=shared.pv["sld_whisp_temp"], step=0.1, label="Whisper temp", scale=1)
                        sld_temp = gr.Number(minimum=0, maximum=2, value=shared.pv["sld_temp"], step=0.1, label="LLM temp", scale=1)
                        sld_buffer = gr.Number(minimum=0, maximum=float(shared.max_message_buffer), step=1.0, value=shared.pv["sld_buffer"], label="Buffer size", scale=1)

            with gr.Row():
                llm_choice = gr.Dropdown(value=shared.pv["llm_choice"], choices=[llm for llm in shared.llm_price.keys()], label="LLM", show_label=True, scale=0, multiselect=False)
                txt_price = gr.Textbox(value=lambda: display_price(shared.pv["sld_max_tkn"], shared.pv["llm_choice"]), label="Price", interactive=False, max_lines=2, lines=2, scale=5)

            with gr.Row():
                flag_audio_btn = gr.Button(value="Flag audio", visible=shared.enable_flagging)
                force_sound_processing_btn = gr.Button(value="Sound processing")
                clear_llm_cache_btn = gr.Button(value="Clear LLM cache")
                pop_buffer_btn = gr.Button(value="Pop buffer", variant="secondary")

        # image
        with gr.Accordion(label="Main gallery", open=True, visible=shared.enable_gallery):
            with gr.Row():
                with gr.Column():
                    roll_gall_btn = gr.Button(value="Roll gallery", min_width=50, visible=all([shared.enable_gallery, shared.enable_queued_gallery]))
                    gallery = gr.Gallery(value=shared.pv["gallery"] if shared.enable_gallery else None, label="Source images", columns=[1], rows=[1], object_fit="scale-down", container=True)
                    with gr.Group():
                        with gr.Row():
                            rst_img_btn = gr.Button(value="Clear", variant="primary", min_width=50)
                            img_btn = gr.Button(value="Add image from clipboard", variant="secondary", min_width=50)

        txt_extra_source = gr.Textbox(value=shared.pv["txt_extra_source"], label="Extra source", lines=1, placeholder="Will be added to the source.", visible=True, max_lines=5)

        with gr.Row():
            btn_chains = []
            for i in range(5):
                but = gr.Button(
                        value=str(i),
                        visible=False,
                        elem_id=f"user_chain_btn #{i}",
                        )
                btn_chains.append(but)
            for ch in btn_chains:
                ch.click(
                        fn=call_user_chain,
                        inputs=[txt_audio],
                        outputs=[txt_audio],
                        preprocess=False,
                        postprocess=False,
                        show_progress=False,
                        )

    with gr.Tab(label="Settings", elem_id="BigTabV2A") as tab_settings:
        with gr.Row():
            roll_dirload_check = gr.Checkbox(value=shared.pv["dirload_check"] if shared.enable_dirload else False, interactive=True, label="Roll from queues", show_label=True, visible=shared.enable_dirload)
        with gr.Row():
            txt_profile = gr.Dropdown(value=shared.pv.profile_name, label="Profile", choices=get_profiles(), multiselect=False, allow_custom_value=True)
        with gr.Row():
            txt_deck = gr.Dropdown(value=shared.pv["txt_deck"], label="Deck name", multiselect=False, choices=get_decks(), allow_custom_value=True)
            txt_whisp_lang = gr.Textbox(value=shared.pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr")
        txt_tags = gr.Dropdown(value=shared.pv["txt_tags"], label="Tags", choices=get_anki_tags(), multiselect=True, allow_custom_value=True)
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
            sld_timestamp_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_timestamp_weight"], step=0.25, label="Timestamp weight")
            sld_keywords_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_keywords_weight"], step=0.25, label="Keywords weight")
        with gr.Row():
            embed_choice = gr.Dropdown(value=shared.pv["embed_choice"], choices=shared.embedding_models, label="Embedding model", show_label=True, scale=0, multiselect=False)
            with gr.Column():
                txt_openai_api_key = gr.Textbox(value=shared.pv["txt_openai_api_key"], label="OpenAI API key", lines=1)
                txt_replicate_api_key = gr.Textbox(value=shared.pv["txt_replicate_api_key"], label="Replicate API key", lines=1)
                txt_mistral_api_key = gr.Textbox(value=shared.pv["txt_mistral_api_key"], label="mistral API key", lines=1)
                txt_openrouter_api_key = gr.Textbox(value=shared.pv["txt_openrouter_api_key"], label="openrouter API key", lines=1)
        with gr.Row():
            kill_threads_btn = gr.Button(value="Kill threads", variant="secondary")
        with gr.Row():
            code_user_flashcard_editor = gr.Code(
                    value=None,
                    language="python",
                    lines=5,
                    label="User flashcard editor function",
                    interactive=False,
                    show_label=True,
                    )
        with gr.Row():
            code_user_chains = gr.Code(
                    value=None,
                    language="python",
                    lines=5,
                    label="User chains",
                    interactive=False,
                    show_label=True,
                    )

    with gr.Tab(label="Memories & Buffer", elem_id="BigTabV2A") as tab_memories_and_buffer:
        with gr.Tab(label="Memories", elem_id="BigTabV2A") as tab_memories:
            df_memories = gr.Dataframe(
                    label="Saved memories",
                    value=None,
                    interactive=False,
                    column_widths="5%",
                    wrap=True,
                    height="2048",
                    )

        with gr.Tab(label="Message buffer", elem_id="BigTabV2A") as tab_buffer:
            df_buffer = gr.Dataframe(
                    label="Message buffer",
                    value=None,
                    interactive=False,
                    column_widths="5%",
                    wrap=True,
                    height="2048",
                    )

    with gr.Tab(label="Queues", elem_id="BigTabV2A"):
        with gr.Tab(label="Queued galleries", elem_id="BigTabV2A") as tab_galleries:

            with gr.Row(visible=shared.enable_queued_gallery):
                with gr.Column():
                    source_txt_btn = gr.Button("OCR the main gallery")
                    source_txt = gr.Textbox(value=None, interactive=False, lines=1, max_lines=20)
                    source_txt_btn.click(
                            fn=ocr_image,
                            inputs=[gallery],
                            outputs=[source_txt],
                            queue=False,
                            preprocess=False,
                            postprocess=False,
                            )

            queued_galleries = []
            for qg in range(1, shared.queued_gallery_slot_nb + 1):
                with gr.Row(equal_height=False, visible=shared.enable_queued_gallery):
                    with gr.Column(scale=10):
                        gal_ = gr.Gallery(
                            # value=None,
                            value=shared.pv[f"queued_gallery_{qg:03d}"] if shared.enable_queued_gallery else None,
                            label=f"Gallery {qg}",
                            columns=[2],
                            rows=[1],
                            object_fit="scale-down",
                            # height=100,
                            # min_width=50,
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
                queued_galleries.append([rst_, gal_, send_, add_, ocr_])


        with gr.Tab(label="Queued audio", elem_id="BigTabV2A") as tab_dirload_queue:
            queue_df = gr.Dataframe(
                    value=None,
                    type="pandas",
                    label="Queued audio",
                    interactive=False,
                    column_widths="5%",
                    visible=shared.enable_dirload,
                    wrap=True,
                    height="2048",
                    )

    with gr.Tab(label="Logging", elem_id="BigTabV2A") as tab_logging:
        with gr.Column():
            logging_reload = gr.Button(value="Refresh")
            output_elem = gr.Textbox(value=None, label="Logging", lines=100, max_lines=1000, interactive=False, placeholder="this string should never appear")


    # with gr.Tab(label="Files", elem_id="BigTabV2A") as tab_files:
    #     with gr.Accordion(label="Done", open=False):
    #         fex_done = gr.FileExplorer(
    #                 root=f"profiles/{shared.pv.profile_name}/queues/audio_done",
    #                 label="Done",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Splitted", open=False):
    #         fex_splitted = gr.FileExplorer(
    #                 root=f"profiles/{shared.pv.profile_name}/queues/audio_splits",
    #                 label="Splitted",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Unsplitted", open=False):
    #         fex_unsplitted = gr.FileExplorer(
    #                 root=f"profiles/{shared.pv.profile_name}/queues/audio_untouched",
    #                 label="Unsplitted",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )
    #     with gr.Accordion(label="Flagged", open=False):
    #         fex_flagged = gr.FileExplorer(
    #                 root=f"profiles/{shared.pv.profile_name}/queues/flagged",
    #                 label="Flagged",
    #                 interactive=True,
    #                 # ignore_glob="**/.st*",
    #                 # visible=False,
    #                 )

    # events ############################################################

    # remove the last item from message buffer
    pop_buffer_btn.click(fn=pop_buffer)

    # reload df
    tab_dirload_queue.select(
            fn=get_dirload_df,
            outputs=[queue_df],
            )
    # load memories only if clicked
    tab_memories_and_buffer.select(
            fn=get_memories_df,
            inputs=[txt_profile],
            outputs=[df_memories],
            )
    tab_memories.select(
            fn=get_memories_df,
            inputs=[txt_profile],
            outputs=[df_memories],
            )
    # load message buffer only if clicked
    tab_buffer.select(
            fn=get_message_buffer_df,
            outputs=[df_buffer],
            show_progress=False,
            )

    # reload tags and deck list when clicking on settings:
    def reload_tags_decks():
        return {
                txt_tags: gr.Dropdown(value=shared.pv["txt_tags"], label="Tags", choices=get_anki_tags(), multiselect=True, allow_custom_value=True),
                txt_deck: gr.Dropdown(value=shared.pv["txt_deck"], label="Deck name", multiselect=False, choices=get_decks(), allow_custom_value=True),
                }
    tab_settings.select(
            fn=reload_tags_decks,
            outputs=[txt_tags, txt_deck],
            ).then(
                    fn=load_user_functions,
                    outputs=[code_user_flashcard_editor, code_user_chains]
                    )

    # copy audio to flag button
    flag_audio_btn.click(
            fn=flag_audio,
            inputs=[txt_profile, txt_audio, txt_whisp_lang, txt_whisp_prompt, txt_chatgpt_cloz, txt_chatgpt_context, gallery],
            show_progress=False,
            )
    # clear cache of llm. Forcing recomputation.
    clear_llm_cache_btn.click(
            fn=clear_llm_cache,
            )
    # force sound preprocessing for the first audio
    force_sound_processing_btn.click(
            fn=force_sound_processing,
            inputs=[audio_slots[0]],
            outputs=[audio_slots[0]],
            preprocess=False,
            postprocess=True,
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
                    inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                    outputs=[txt_chatgpt_cloz],
                    queue=False,
                    preprocess=False,
                    postprocess=False,
                    )

    # load output elem if clicked
    tab_logging.select(fn=get_log, outputs=[output_elem])
    logging_reload.click(fn=get_log, outputs=[output_elem])


    # mark the previous card
    mark_previous.click(
            fn=mark_previous_note,
            show_progress=False,
            )

    # darkmode
    dark_mode_btn.click(fn=None, js=darkmode_js, show_progress=False)

    # sync anki
    sync_btn.click(fn=sync_anki, queue=False, show_progress=False)

    # kill threads before timeout
    kill_threads_btn.click(fn=kill_threads, show_progress=False)

    # display card status
    update_status_btn.click(
            fn=get_card_status,
            inputs=[txt_chatgpt_cloz],
            outputs=[update_status_btn],
            # queue=False,
            # show_progress=True,
            # every=2,
            # trigger_mode="once",
            preprocess=False,
            postprocess=False,
            )
    # txt_chatgpt_cloz.change(
    #         fn=get_card_status,
    #         inputs=[txt_chatgpt_cloz],
    #         outputs=[update_status_btn],
    #         # queue=False,
    #         # show_progress=True,
    #         # every=2,
    #         # trigger_mode="once",
    #         )


    # display pricing then save values
    sld_max_tkn.change(
            fn=display_price,
            inputs=[sld_max_tkn, llm_choice],
            outputs=[txt_price],
            show_progress=False,
            ).then(
                    fn=shared.pv.save_sld_max_tkn,
                    inputs=[sld_max_tkn],
                    show_progress=False,
                    )
    llm_choice.change(
            fn=display_price,
            inputs=[sld_max_tkn, llm_choice],
            outputs=[txt_price],
            show_progress=False,
            ).success(
                    fn=shared.pv.save_llm_choice,
                    inputs=[llm_choice],
                    show_progress=False,
                    )

    # change some values to profile
    sld_whisp_temp.change(fn=shared.pv.save_sld_whisp_temp, inputs=[sld_whisp_temp], show_progress=False)
    sld_buffer.change(fn=shared.pv.save_sld_buffer, inputs=[sld_buffer], show_progress=False)
    sld_temp.change(fn=shared.pv.save_sld_temp, inputs=[sld_temp], show_progress=False)
    roll_dirload_check.change(fn=shared.pv.save_dirload_check, inputs=[roll_dirload_check], show_progress=False)
    prompt_manag.change(fn=shared.pv.save_prompt_management, inputs=[prompt_manag], show_progress=False)
    txt_tags.select(fn=shared.pv.save_txt_tags, inputs=[txt_tags], show_progress=False)
    txt_deck.select(fn=shared.pv.save_txt_deck, inputs=[txt_deck], show_progress=False)
    txt_chatgpt_context.change(fn=shared.pv.save_txt_chatgpt_context, inputs=[txt_chatgpt_context], show_progress=False)
    txt_whisp_prompt.change(fn=shared.pv.save_txt_whisp_prompt, inputs=[txt_whisp_prompt], show_progress=False)
    txt_whisp_lang.change(fn=shared.pv.save_txt_whisp_lang, inputs=[txt_whisp_lang], show_progress=False)
    txt_keywords.change(fn=shared.pv.save_txt_keywords, inputs=[txt_keywords], show_progress=False)
    sld_pick_weight.change(fn=shared.pv.save_sld_pick_weight, inputs=[sld_pick_weight], show_progress=False)
    sld_prio_weight.change(fn=shared.pv.save_sld_prio_weight, inputs=[sld_prio_weight], show_progress=False)
    sld_timestamp_weight.change(fn=shared.pv.save_sld_timestamp_weight, inputs=[sld_timestamp_weight], show_progress=False)
    sld_keywords_weight.change(fn=shared.pv.save_sld_keywords_weight, inputs=[sld_keywords_weight], show_progress=False)
    txt_extra_source.change(fn=shared.pv.save_txt_extra_source, inputs=[txt_extra_source], show_progress=False)
    txt_openai_api_key.change(fn=shared.pv.save_txt_openai_api_key, inputs=[txt_openai_api_key], show_progress=False)
    txt_replicate_api_key.change(fn=shared.pv.save_txt_replicate_api_key, inputs=[txt_replicate_api_key], show_progress=False)
    txt_mistral_api_key.change(fn=shared.pv.save_txt_mistral_api_key, inputs=[txt_mistral_api_key], show_progress=False)
    txt_openrouter_api_key.change(fn=shared.pv.save_txt_openrouter_api_key, inputs=[txt_openrouter_api_key], show_progress=False)
    embed_choice.change(fn=shared.pv.save_embed_choice, inputs=[embed_choice], show_progress=False)

    # change profile and load previous data
    txt_profile.change(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_slots[0], txt_audio, txt_chatgpt_cloz, txt_profile]
            ).then(
                    fn=shared.reset,
                    )

    # load image then OCR it then save it to profile
    img_btn.click(
            fn=get_image,
            inputs=[gallery],
            outputs=[gallery],
            show_progress=False,
            queue=False,
            )

    # when gallery changes, save the image then run ocr
    gallery.change(
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

    # queued gallery
    for qg_cnt, qg in enumerate(range(1, shared.queued_gallery_slot_nb + 1)):
        rst_, gal_, send_, add_, ocr_ = queued_galleries[qg_cnt]

        # save any change to the gallery
        gal_.change(
                fn=getattr(shared.pv, f"save_queued_gallery_{qg:03d}"),
                inputs=[gal_],
                show_progress=False,
                queue=True,
                )

        # for the first few galleries only: run ocr in advance
        if qg_cnt <= 2:
            gal_.change(
                    fn=get_img_source,
                    inputs=[gal_],
                    queue=True,
                    )

        # add image
        add_.click(
                fn=get_image,
                inputs=[gal_],
                outputs=[gal_],
                queue=False)

        # send image
        send_.click(
                fn=lambda x: x,
                inputs=[gal_],
                outputs=[gallery],
                preprocess=False,
                postprocess=False,
                queue=False,
                )

        # reset image
        rst_.click(
                fn=lambda: None,
                outputs=[gal_],
                queue=False,
                ).then(  # force deletion
                        fn=getattr(shared.pv, f"save_queued_gallery_{qg:03d}"),
                        inputs=[gal_],
                        show_progress=False,
                        queue=True,
                        )


        # ocr image
        ocr_.click(
                fn=ocr_image,
                inputs=[gal_],
                outputs=[source_txt],
                queue=False,
                preprocess=False,
                postprocess=False,
                )

    roll_gall_btn.click(
            fn=roll_future_galleries,
            inputs=[row[1] for row in queued_galleries],
            outputs=[gallery] + [row[1] for row in queued_galleries],
            # preprocess=False,
            # postprocess=False,
            )

    # audio
    audio_corrector.stop_recording(
            fn=audio_edit,
            inputs=[audio_corrector, audio_corrector_txt, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context],
            outputs=[txt_chatgpt_cloz, audio_corrector, audio_corrector_txt],
            show_progress=False,
            )
    audio_corrector_txt.submit(
            fn=audio_edit,
            inputs=[audio_corrector, audio_corrector_txt, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_cloz, txt_chatgpt_context],
            outputs=[txt_chatgpt_cloz, audio_corrector, audio_corrector_txt],
            show_progress=False,
            )

    rst_audio_btn.click(
            fn=reset_audio,
            outputs=audio_slots,
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            )

    rollaudio_1_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
            queue=False,
            show_progress=False,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0], txt_whisp_prompt, txt_whisp_lang, sld_whisp_temp],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    queue=False,
                    ).then(
                        lambda: None,
                        outputs=[txt_chatgpt_cloz],
                    ).success(
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
                                llm_choice,
                                txt_keywords,
                                prompt_manag,
                                ],
                            outputs=[audio_slots[-1]],
                            preprocess=False,
                            postprocess=True,
                            queue=False,
                            show_progress=False,
                            ).then(
                                    fn=get_card_status,
                                    inputs=[txt_chatgpt_cloz],
                                    outputs=[update_status_btn],
                                    # queue=True,
                                    preprocess=False,
                                    postprocess=False,
                                    )
    rollaudio_12_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
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
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                        outputs=[txt_chatgpt_cloz],
                        queue=False,
                        preprocess=False,
                        postprocess=False,
                    ).success(
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
                                llm_choice,
                                txt_keywords,
                                prompt_manag,
                                ],
                            outputs=[audio_slots[-1]],
                            # preprocess=False,
                            # postprocess=False,
                            queue=False,
                            show_progress=False,
                            ).then(
                                    fn=get_card_status,
                                    inputs=[txt_chatgpt_cloz],
                                    outputs=[update_status_btn],
                                    # queue=True,
                                    preprocess=False,
                                    postprocess=False,
                                    )
    rollaudio_123_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
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
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                        outputs=[txt_chatgpt_cloz],
                        preprocess=False,
                        postprocess=False,
                        queue=False,
                        ).success(
                                fn=lambda: "Rolling",
                                outputs=[update_status_btn],
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
                                                llm_choice,
                                                txt_keywords,
                                                prompt_manag,
                                                ],
                                            outputs=[audio_slots[-1]],
                                            # preprocess=False,
                                            # postprocess=False,
                                            queue=False,
                                            show_progress=False,
                                            ).then(
                                                    fn=lambda: False,
                                                    outputs=[check_marked],
                                                    show_progress=False,
                                                    ).then(
                                                            fn=get_card_status,
                                                            inputs=[txt_chatgpt_cloz],
                                                            outputs=[update_status_btn],
                                                            # queue=True,
                                                            preprocess=False,
                                                            postprocess=False,
                                                            )

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use Voice2Anki on the computer but record the audio
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
                llm_choice,
                txt_keywords,
                prompt_manag,
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
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                        outputs=[txt_chatgpt_cloz],
                        queue=False,
                        preprocess=False,
                        postprocess=False,
                        ).then(
                                fn=get_card_status,
                                inputs=[txt_chatgpt_cloz],
                                outputs=[update_status_btn],
                                # queue=True,
                                preprocess=False,
                                postprocess=False,
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
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
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
                    ).then(
                            fn=get_card_status,
                            inputs=[txt_chatgpt_cloz],
                            outputs=[update_status_btn],
                            # queue=True,
                            preprocess=False,
                            postprocess=False,
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
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
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
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
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
                llm_choice,
                ],
            preprocess=False,
            postprocess=False,
            queue=False,
            show_progress=False,
            )

    init = demo.load(
            fn=shared.reset,
            show_progress=False,
            js=darkmode_js if not (datetime.now().hour <= 8 or datetime.now().hour >= 19) else None,
            ).then(
                    fn=load_user_chain,
                    inputs=btn_chains,
                    outputs=btn_chains,
                    )
    init.then(
            fn=update_audio_slots_txts,
            inputs=audio_slots_txts,
            outputs=audio_slots_txts,
            every=0.5,
            show_progress=False,
            postprocess=False,
            preprocess=False,
            trigger_mode="once",
            )
    init.then(
            fn=sync_anki,
            queue=False,
            show_progress=False,
            )

    if shared.pv.profile_name == "default":
        gr.Warning("Enter a profile then press enter.")
