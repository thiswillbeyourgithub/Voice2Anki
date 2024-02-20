import gradio as gr
from datetime import datetime
from functools import partial

from gradio.themes.utils import sizes as theme_size

from .profiles import get_profiles, switch_profile, load_user_functions, load_user_chain, call_user_chain
from .main import transcribe, alfred, to_anki, dirload_splitted, dirload_splitted_last, kill_threads, audio_edit, flag_audio, pop_buffer, clear_cache
from .anki_utils import sync_anki, get_card_status, mark_previous_notes, suspend_previous_notes, get_anki_tags, get_decks
from .logger import get_log, red, yel
from .memory import recur_improv, display_price, get_memories_df, get_message_buffer_df, get_dirload_df
from .media import get_image, reset_audio, reset_gallery, get_img_source, ocr_image, roll_queued_galleries, create_audio_compo, roll_audio, force_sound_processing, update_audio_slots_txts, qg_add_to_new, qg_add_to_latest
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

html_head = """
<script>

function unfocus(e) {
    document.activeElement.blur();
    document.body.focus();
    document.documentElement.focus();
    e.preventDefault();  // make sure to avoid scrolling after pressing spacebar
}

function shortcuts(e) {
    // the switch is so that keypress are ignored if an input element is
    // in focus
    var event = document.all ? window.event : e;
    switch (e.target.tagName.toLowerCase()) {

        // unselect anything by pressing shift+space or escape
        case "input":
        case "textarea":
            if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
                unfocus(e);
            }

        //case "select":
        //case "button":
        break;

        default:
        // suspend previous card
        if (e.code == "KeyS" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_suspendpreviousbtn").click();
        }
        // mark previous card
        else if (e.code == 'Semicolon' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_markpreviousbtn").click();
        }
        // untoggle check next card
        else if (e.key == 'm') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_marknext").children[1].children[0].checked = false;
        }
        // get card status
        else if (e.key == "s") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_cardstatusbtn").click();
        }
        // recur improvement
        else if (e.code == "KeyF" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_llmfeedbackbtn").click();
        }

        // toggle nightmode
        else if (e.code == "KeyN" && e.shiftKey) {
            document.getElementById("js_darkmodebtn").click();
        }

        // select textbox
        else if (e.key == "e") {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtchatgpt").children;
            ch[ch.length - 1].focus();
        }
        else if (e.code == "KeyE" && e.shiftKey) {
            e.preventDefault();  // dont type the e
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            ch = document.getElementById("js_txtwhisper").children;
            ch[ch.length - 1].focus();
        }

        // roll 1 2 3
        else if (e.key == "&") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll1").click();
        }
        else if (e.key == "é") {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll12").click();
        }
        else if (e.key == '"') {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_roll123").click();
        }

        // 123
        else if (e.key == "3" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_toankibtn").click();
        }
        else if (e.key == "2" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcriptbtn").click();
        }
        else if (e.key == "1" && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            document.getElementById("js_transcribebtn").click();
        }

        // roll gallery
        else if (document.getElementById('js_guienablequeuedgallery').children[1].children[0].checked == true && e.code == 'KeyG' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Roll gallery?")) {
                document.getElementById("js_rollgallbtn").click();
            }
        }
        // add to next queued gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyR' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                    document.getElementById("js_btnqgnew").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }
        // append to latest queueud gallery
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyQ' && e.shiftKey) {
            if (!document.querySelector('.js_tabqueues').checkVisibility()) { alert("Shortcut only available in tab 'Queues'") ; return ;}
            // only active if the right tabs are enabled
            if (document.querySelector('.js_tabqueues').checkVisibility() && document.querySelector('.js_queueqgclass').checkVisibility()) {
                document.getElementById("js_btnqgadd").click();
            }
            else {
                alert("To add to queued gallery, go to 'Queues' then 'Queud audio'");
            }
        }

        // dirload
        else if (document.getElementById('js_guienabledirload').children[1].children[0].checked == true && e.code == 'KeyD' && e.shiftKey) {
            if (!document.querySelector('.js_tabmain').checkVisibility()) { alert("Shortcut only available in tab 'Main'") ; return ;}
            if (confirm("Load from dir?")) {
                document.getElementById("js_dirloadbtn").click();
            }
        }

        // unfocus
        else if ((e.code == 'Space' && e.shiftKey) || (e.key == 'Escape') || (e.keyCode == 27)) {
            unfocus(e);
        }

        // ignore space
        else if ((e.code == 'Space' && e.shiftKey) || (e.code == 'Space')) {
            // do nothing
        }

        // no shortcut found
        else {
            alert(`Unrecognized shortcut: ${e.key} (or ${e.code})`);
            }

        }
}
function tabswitcher(e) {
    // tab to switch tab
    if (event.key === 'Tab') {
        event.preventDefault();

        // the subtabs cycled depend on the main tab focused
        if (document.querySelector('.js_tabmain').checkVisibility()) {
            var selector = '.js_subtab_main'
        }
        else if (document.querySelector('.js_tabsettings').checkVisibility()) {
            var selector = '.js_subtab_settings'
        }
        else if (document.querySelector('.js_tabqueues').checkVisibility()) {
            var selector = '.js_subtab_queues'
        }
        else if (document.querySelector('.js_tabmemoriesandbuffer').checkVisibility()) {
            var selector = '.js_subtab_memoriesandbuffer'
        }
        else {
            // alert("No subtab to switch here.");
            return;
        }


        var tabs = document.querySelectorAll(selector);
        tabs.forEach((tab, index) => {
            if (tab.checkVisibility()) {
                if (event.shiftKey) {
                    var newIndex = (index > 0) ? index - 1 : tabs.length - 1;
                } else {
                    var newIndex = (index + 1) % tabs.length;
                }
                tabs[0].parentElement.childNodes[0].children[newIndex].click();
                return;
            }
        });

    }
}

document.addEventListener('keypress', shortcuts, false);
document.addEventListener('keydown', tabswitcher, false);

</script>
"""

# dynamically adjust the height of the app to avoid scrolling up abruptly
js_longer = """() => {
    document.querySelectorAll(".app")[0].style.height='5000px';
}
"""
js_reset_height = """() => {
    document.querySelectorAll(".app")[0].style.height='';
}
"""

# executed on load
js_load = """() => {
    // make sure the audios keep the same size even when they are unset
    var h = Math.floor(2.3 * document.getElementsByClassName("js_audiocomponent")[0].clientHeight);

    Array.from(document.getElementsByClassName("js_audiocomponent")).forEach(el => el.style.height = `${h}px`)

}
"""

css = """
/* make sure those tabs take all the width */
#js_widetabs-button { flex-grow: 1 !important;}

/* remove source selector */
.mic-select {display: none !important; flex-grow:0 !important;}

/* Larger font for some text elements */
#js_txtchatgpt > label > textarea {font-size: 20px;}
#js_txtwhisper > label > textarea {font-size: 20px;}
"""

if shared.widen_screen:
    css += "\n.app { max-width: 100% !important; }"

with gr.Blocks(
        analytics_enabled=False,
        title=f"Voice2Anki V{shared.VERSION}",
        theme=theme,
        css=css,
        head=html_head,
        ) as demo:

    with gr.Group():
        with gr.Row():
            gr.Button(value=f"Voice2Anki V{shared.VERSION}", variant="primary", scale=2, interactive=True, size="sm", min_width=100)
            sync_btn = gr.Button(value="Sync anki", variant="secondary", scale=1, elem_id="js_syncankibtn", size="sm", min_width=100)
            dark_mode_btn = gr.Button("Dark/Light", variant="secondary", scale=1, elem_id="js_darkmodebtn", size="sm", min_width=100)

    with gr.Tab(label="Main", elem_id="js_widetabs", elem_classes=["js_tabmain"]) as tab_main:

        with gr.Row():
            rst_audio_btn = gr.Button(value="Clear audio", variant="primary", min_width=50, scale=1, size="sm")
            dir_load_btn = gr.Button(value="Dirload", variant="secondary", min_width=50, scale=5, elem_id="js_dirloadbtn", visible=shared.pv["enable_dirload"], size="sm")

        # audio
        audio_number = shared.audio_slot_nb
        audio_slots = []
        audio_slots_txts = []
        for i in range(audio_number):
                audio_mp3 = create_audio_compo(scale=1, label=f"Audio #{i+1}", render=False)
                audio_slots.append(audio_mp3)
                audio_slots_txt = gr.HTML(render=False, visible=shared.pv["enable_dirload"])
                audio_slots_txts.append(audio_slots_txt)
        # rendering afterwards to reverse the order
        for aud, t in zip(audio_slots[::-1], audio_slots_txts[::-1]):
            with gr.Row():
                with gr.Column(min_width=50):
                    aud.render()
                with gr.Column(min_width=50):
                    t.render()

        # whisper and chatgpt text output
        with gr.Row():
            txt_audio = gr.Textbox(label="Transcript", lines=15, max_lines=25, placeholder="The transcript of the audio recording will appear here", container=True, interactive=True, scale=1, elem_id="js_txtwhisper", show_label=True)
            txt_chatgpt_cloz = gr.Textbox(label="LLM cloze(s)", lines=15, max_lines=25, placeholder="The anki flashcard will appear here", container=True, interactive=True, scale=1, elem_id="js_txtchatgpt", show_label=True)

        # rolls
        with gr.Group():
            with gr.Row():
                rollaudio_123_btn = gr.Button(value="Roll+123", variant="primary", scale=5, elem_id="js_roll123", size="lg", min_width=100)
                rollaudio_12_btn = gr.Button(value="Roll+12", variant="primary", scale=5, elem_id="js_roll12", size="lg", min_width=100)
                rollaudio_1_btn = gr.Button(value="Roll+1", variant="primary", scale=5, visible=False, elem_id="js_roll1", size="lg", min_width=100)
                update_status_btn = gr.Button(value="Card status", variant="secondary", scale=0, interactive=True, elem_id="js_cardstatusbtn", size="lg", min_width=100)

        # 1/2/3
        with gr.Group():
            with gr.Row():
                transcript_btn = gr.Button(value="Transcribe", variant="secondary", elem_id="js_transcribebtn", size="sm", min_width=100)
                chatgpt_btn = gr.Button(value="Clozify", variant="secondary", elem_id="js_transcriptbtn", size="sm", min_width=100)
                anki_btn = gr.Button(value="Ankify", variant="secondary", elem_id="js_toankibtn", size="sm", min_width=100)

        with gr.Row():
            with gr.Column(scale=2, variant="compact", min_width=100):
                with gr.Row():
                    mark_previous = gr.Button(value="Mark prev.", elem_id="js_markpreviousbtn", size="sm", scale=3, min_width=75)
                    check_marked = gr.Checkbox(value=False, interactive=True, label="Mark next", show_label=False, elem_id="js_marknext", scale=1, min_width=75)
            with gr.Column(scale=1, min_width=100):
                suspend_previous = gr.Button(value="Suspend prev.", elem_id="js_suspendpreviousbtn", size="sm", scale=2, min_width=75)

        # quick settings
        with gr.Tab(label="Controls", elem_classes=["js_subtab_main"], elem_id="js_widetabs"):
            with gr.Row():
                with gr.Column(scale=2, variant="compact", min_width=75):
                    with gr.Row():
                        sld_improve = gr.Number(minimum=0, maximum=10, value=5.0, step=1.0, label="Mem priority", min_width=100, scale=1, elem_id="js_mempriority", show_label=False)
                        improve_btn = gr.Button(value="Memorize", variant="secondary", elem_id="js_llmfeedbackbtn", size="sm", min_width=100, scale=3)
                prompt_manag = gr.Radio(choices=["messages", "stuff"], value=shared.pv["prompt_management"], interactive=True, label="Prompt style", show_label=False, scale=1)

            with gr.Row():
                with gr.Column(scale=10):
                    with gr.Row():
                        sld_max_tkn = gr.Number(minimum=0, maximum=15000, value=shared.pv["sld_max_tkn"], step=100.0, label="LLM avail. tkn.", scale=1)
                        sld_whisp_temp = gr.Number(minimum=0, maximum=1, value=shared.pv["sld_whisp_temp"], step=0.1, label="Whisper temp", scale=1)
                        stt_choice = gr.Dropdown(value=shared.pv["stt_choice"], choices=shared.stt_models, label="STT model", show_label=True, scale=0, multiselect=False)
                        sld_temp = gr.Number(minimum=0, maximum=2, value=shared.pv["sld_temp"], step=0.1, label="LLM temp", scale=1)
                        sld_buffer = gr.Number(minimum=0, maximum=float(shared.max_message_buffer), step=1.0, value=shared.pv["sld_buffer"], label="Buffer size", scale=1)

            with gr.Row():
                llm_choice = gr.Dropdown(value=shared.pv["llm_choice"], choices=[llm for llm in shared.llm_price.keys()], label="LLM", show_label=True, scale=0, multiselect=False)
                txt_price = gr.Textbox(value=lambda: display_price(shared.pv["sld_max_tkn"], shared.pv["llm_choice"]), label="Price", interactive=False, max_lines=2, lines=2, scale=5)

            with gr.Row():
                flag_audio_btn = gr.Button(value="Flag audio", visible=shared.pv["enable_flagging"], size="sm")
                force_sound_processing_btn = gr.Button(value="Extra sound processing", visible=shared.pv["enable_dirload"], size="sm")
                clear_cache_btn = gr.Button(value="Clear cache", size="sm")
                pop_buffer_btn = gr.Button(value="Pop buffer", variant="secondary", size="sm")

            txt_extra_source = gr.Textbox(value=shared.pv["txt_extra_source"], label="Extra source", lines=1, placeholder="Will be added to the source.", visible=True, max_lines=5)

        # image
        with gr.Tab(label="Gallery", visible=shared.pv["enable_gallery"], elem_classes=["js_subtab_main"], elem_id="js_widetabs") as tab_gallery:
            with gr.Column():
                roll_gall_btn = gr.Button(value="Roll gallery", min_width=50, visible=shared.pv["enable_queued_gallery"], elem_id="js_rollgallbtn", size="sm")
                gallery = gr.Gallery(value=shared.pv["gallery"], label="Source images", columns=[1], rows=[1], object_fit="scale-down", container=True)
                with gr.Group():
                    with gr.Row():
                        rst_img_btn = gr.Button(value="Clear", variant="primary", min_width=50, size="sm")
                        img_btn = gr.Button(value="Add image from clipboard", variant="secondary", min_width=50, size="sm")

        with gr.Tab(label="Edit", elem_classes=["js_subtab_main"], elem_id="js_widetabs"):
            with gr.Row():
                audio_corrector = create_audio_compo(
                        label="AudioEdit via GPT-4",
                        container=True,
                        scale=0,
                        editable=True,
                        )
                audio_corrector_txt = gr.Textbox(value=None, label="Edit via GPT-4", scale=2)
                auto_btn = gr.Button(value="1+2+3", variant="secondary", scale=1, min_width=50, visible=False, size="sm")
                semiauto_btn = gr.Button(
                        value="1+2",
                        variant="secondary",
                        #scale=3,
                        #min_width=50,
                        visible=False,
                        )

        with gr.Tab(label="User chains", elem_classes=["js_subtab_main"], elem_id="js_widetabs"):
            btn_chains = []
            for i in range(5):
                but = gr.Button(
                        value=str(i),
                        visible=False,
                        elem_id=f"js_userchainbtn_#{i}",
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

    with gr.Tab(label="Settings", elem_id="js_widetabs", elem_classes=["js_tabsettings"]) as tab_settings:
        with gr.Tab(label="GUI", elem_id="js_widetabs", elem_classes=["js_subtab_settings"]):
            with gr.Row():
                gui_enable_dirload = gr.Checkbox(value=shared.pv["enable_dirload"], interactive=True, label="Dirload", show_label=True, elem_id="js_guienabledirload")
                gui_rolldirloadcheck = gr.Checkbox(value=shared.pv["dirload_check"], interactive=True, label="Clicking on Roll loads from dirload", show_label=True)
            with gr.Row():
                gui_enable_gallery = gr.Checkbox(value=shared.pv["enable_gallery"], interactive=True, label="Gallery", show_label=True, elem_id="js_guienablegallery")
                gui_enable_queued_gallery = gr.Checkbox(value=shared.pv["enable_queued_gallery"], interactive=True, label="Gallery queue", show_label=True, elem_id="js_guienablequeuedgallery")
            with gr.Row():
                gui_enable_flagging = gr.Checkbox(value=shared.pv["enable_flagging"], interactive=True, label="Flagging", show_label=True)

        with gr.Tab(label="Anki", elem_id="js_widetabs", elem_classes=["js_subtab_settings"]):
            with gr.Row():
                txt_profile = gr.Dropdown(value=shared.pv.profile_name, label="Profile", choices=get_profiles(), multiselect=False, allow_custom_value=True)
            with gr.Row():
                txt_deck = gr.Dropdown(value=shared.pv["txt_deck"], label="Deck name", multiselect=False, choices=get_decks(), allow_custom_value=True)
                txt_whisp_lang = gr.Textbox(value=shared.pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr")
            txt_tags = gr.Dropdown(value=shared.pv["txt_tags"], label="Tags", choices=get_anki_tags(), multiselect=True, allow_custom_value=True)
        with gr.Tab(label="LLM", elem_id="js_widetabs", elem_classes=["js_subtab_settings"]):
            with gr.Row():
                txt_whisp_prompt = gr.Textbox(value=shared.pv["txt_whisp_prompt"], lines=2, label="SpeechToText context", placeholder="context for whisper")
                txt_chatgpt_context = gr.Textbox(value=shared.pv["txt_chatgpt_context"], lines=2, label="LLM context", placeholder="context for ChatGPT")
            with gr.Column():
                embed_choice = gr.Dropdown(value=shared.pv["embed_choice"], choices=shared.embedding_models, label="Embedding model", show_label=True, scale=0, multiselect=False)
                txt_openai_api_key = gr.Textbox(value=shared.pv["txt_openai_api_key"], label="OpenAI API key", lines=1)
                txt_replicate_api_key = gr.Textbox(value=shared.pv["txt_replicate_api_key"], label="Replicate API key", lines=1)
                txt_mistral_api_key = gr.Textbox(value=shared.pv["txt_mistral_api_key"], label="mistral API key", lines=1)
                txt_openrouter_api_key = gr.Textbox(value=shared.pv["txt_openrouter_api_key"], label="openrouter API key", lines=1)
        with gr.Tab(label="Memory retrieval", elem_id="js_widetabs", elem_classes=["js_subtab_settings"]):
            with gr.Row():
                txt_keywords = gr.Textbox(value=shared.pv["txt_keywords"], lines=3, max_lines=2, label="Keywords", placeholder="Comma separated regex that, if present in the transcript, increase chances of matching memories to be selected. Each regex is stripped, case insensitive and can be used multiple times to increase the effect.")
            with gr.Row():
                sld_pick_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_pick_weight"], step=0.1, label="Embeddings weight")
                sld_prio_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_prio_weight"], step=0.1, label="Priority weight")
                sld_timestamp_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_timestamp_weight"], step=0.1, label="Timestamp weight")
                sld_keywords_weight = gr.Slider(minimum=0, maximum=10, value=shared.pv["sld_keywords_weight"], step=0.1, label="Keywords weight")
        with gr.Tab(label="Misc", elem_id="js_widetabs", elem_classes=["js_subtab_settings"]):
            with gr.Row():
                kill_threads_btn = gr.Button(value="Kill threads", variant="secondary", size="sm")
            with gr.Accordion(label="User functions", open=False):
                with gr.Row():
                    code_user_flashcard_editor = gr.Code(
                            value=None,
                            language="python",
                            lines=5,
                            label="flashcard_editor.py",
                            interactive=False,
                            show_label=True,
                            )
                with gr.Row():
                    code_user_chains = gr.Code(
                            value=None,
                            language="python",
                            lines=5,
                            label="chains.py",
                            interactive=False,
                            show_label=True,
                            )

    with gr.Tab(label="Queues", elem_id="js_widetabs", elem_classes=["js_tabqueues"], visible=shared.pv["enable_queued_gallery"] or shared.pv["enable_dirload"]) as tab_queues:
        with gr.Tab(label="Queued galleries", elem_id="js_widetabs", elem_classes=["js_queueqgclass", "js_subtab_queues"], visible=shared.pv["enable_queued_gallery"]) as tab_queued_galleries:

            with gr.Row():
                with gr.Column():
                    source_txt_btn = gr.Button("OCR the main gallery", size="sm")
                    source_txt = gr.Textbox(value=None, interactive=False, lines=1, max_lines=20)
                    source_txt_btn.click(
                            fn=ocr_image,
                            inputs=[gallery],
                            outputs=[source_txt],
                            preprocess=False,
                            postprocess=False,
                            )

            queued_galleries = []
            for qg in range(1, shared.queued_gallery_slot_nb + 1):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=10):
                        gal_ = gr.Gallery(
                            # value=None,
                            value=shared.pv[f"queued_gallery_{qg:03d}"],
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
                        send_ = gr.Button(value="Send to gallery", variant="primary", min_width=50, scale=10, size="sm")
                        add_ = gr.Button(value="Add image from clipboard", min_width=50, scale=10, size="sm")
                        with gr.Row():
                            rst_ = gr.Button(value="Clear", variant="primary", min_width=50, scale=0, size="sm")
                            ocr_ = gr.Button("OCR", variant="secondary", scale=1, size="sm")
                queued_galleries.append([rst_, gal_, send_, add_, ocr_])

            btn_qg_new = gr.Button(visible=False, elem_id="js_btnqgnew")
            btn_qg_add = gr.Button(visible=False, elem_id="js_btnqgadd")


        with gr.Tab(label="Queued audio", elem_id="js_widetabs", visible=shared.pv["enable_dirload"], elem_classes=["js_subtab_queues"]) as tab_dirload_queue:
            queue_df = gr.Dataframe(
                    value=None,
                    type="pandas",
                    label="Queued audio",
                    interactive=False,
                    wrap=True,
                    height=2000,
                    column_widths=["1%", "20%", "20%", "5%", "5%", "20%", "20%", "5%", "5%"],
                    )

    with gr.Tab(label="Logging", elem_id="js_widetabs") as tab_logging:
        with gr.Column():
            logging_reload = gr.Button(value="Refresh", size="sm")
            output_elem = gr.Textbox(value=None, label="Logging", lines=100, max_lines=1000, interactive=False, placeholder="this string should never appear")


    with gr.Tab(label="Memories & Buffer", elem_id="js_widetabs", elem_classes=["js_tabmemoriesandbuffer"]) as tab_memories_and_buffer:
        with gr.Tab(label="Memories", elem_id="js_widetabs", elem_classes=["js_subtab_memoriesandbuffer"]) as tab_memories:
            df_memories = gr.Dataframe(
                    label="Saved memories",
                    value=None,
                    interactive=False,
                    wrap=True,
                    height=2000,
                    column_widths=["1%", "25%", "5%", "5%", "25%", "10%", "10%", "5%", "5%", "10%"],
                    )

        with gr.Tab(label="Message buffer", elem_id="js_widetabs", elem_classes=["js_subtab_memoriesandbuffer"]) as tab_buffer:
            df_buffer = gr.Dataframe(
                    label="Message buffer",
                    value=None,
                    interactive=False,
                    wrap=True,
                    height=2000,
                    column_widths=["1%", "10%", "10%", "10%", "10%", "5%"],
                    )

    # with gr.Tab(label="Console", elem_id="js_widetabs"):
    #     from pprint import pformat
    #     with gr.Column():
    #         console = gr.Chatbot(
    #                 value=None,
    #                 label="Console",
    #                 show_label=True,
    #                 sanitize_html=False,
    #                 render_markdown=False,
    #                 bubble_full_width=True,
    #                 layout="panel",
    #                 height="100%",
    #                 )

    #         with gr.Row():
    #             console_in = gr.Textbox(scale=5)
    #             console_reset = gr.Button("Reset", scale=1)
    #     def exec_console(history, msg):
    #         red(f"Console in:'{msg}'")
    #         try:
    #             answer = pformat(eval(msg, globals()))
    #         except Exception as err:
    #             answer = f"Error: {err}"
    #         red(f"Console out:'{answer}'")
    #         history.append([msg, answer])

    #         return [history, None]
    #     console_in.submit(
    #             fn=exec_console,
    #             inputs=[console, console_in],
    #             outputs=[console, console_in],
    #             postprocess=False,
    #             )
    #     console_reset.click(fn=lambda: None, outputs=[console])

    # with gr.Tab(label="Files", elem_id="js_widetabs") as tab_files:
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

    def save_and_load_gui(value: bool, name: str):
        if name == "enable_dirload":
            shared.pv[name] = value
            if value is False:
                shared.pv["gui_rolldirloadcheck"] = False
            outdict = {
                    tab_gallery: gr.update(visible=value),
                    tab_queues: gr.update(visible=shared.pv["enable_queued_gallery"] or shared.pv["enable_dirload"]),
                    dir_load_btn: gr.update(visible=value),
                    force_sound_processing_btn: gr.update(visible=value),
                    gui_rolldirloadcheck: gr.update(visible=value, value=False),
                    }
            for ast in audio_slots_txts:
                outdict[ast] = gr.update(visible=value)
            return outdict

        elif name == "enable_gallery":
            shared.pv[name] = value
            if value is False:
                shared.pv["enable_queued_gallery"] = False
            return {
                    tab_gallery: gr.update(visible=value),
                    roll_gall_btn: gr.update(visible=value),
                    tab_queued_galleries: gr.update(visible=value),
                    tab_queues: gr.update(visible=shared.pv["enable_queued_gallery"] or shared.pv["enable_dirload"]),
                    gui_enable_queued_gallery: gr.update(visible=value, value=False),
                    }

        elif name == "enable_queued_gallery":
            shared.pv[name] = value
            return {
                roll_gall_btn: gr.update(visible=value),
                tab_queued_galleries: gr.update(visible=value),
                tab_queues: gr.update(visible=shared.pv["enable_queued_gallery"] or shared.pv["enable_dirload"]),
                }

        elif name == "enable_flagging":
            shared.pv[name] = value
            return {
                flag_audio_btn: gr.update(visible=value),
                }
        else:
            raise ValueError(name)

    # update gui
    gui_outputs = [
            tab_queues,
            tab_queued_galleries,
            tab_gallery,
            gui_enable_gallery,
            roll_gall_btn,
            flag_audio_btn,
            dir_load_btn,
            force_sound_processing_btn,
            gui_rolldirloadcheck,
            gui_enable_queued_gallery,
            ] + audio_slots_txts
    gui_enable_dirload.input(
            fn=partial(save_and_load_gui, name="enable_dirload"),
            inputs=[gui_enable_dirload],
            outputs=gui_outputs,
            )
    gui_enable_flagging.input(
            fn=partial(save_and_load_gui, name="enable_flagging"),
            inputs=[gui_enable_flagging],
            outputs=gui_outputs,
            )
    gui_enable_queued_gallery.input(
            fn=partial(save_and_load_gui, name="enable_queued_gallery"),
            inputs=[gui_enable_queued_gallery],
            outputs=gui_outputs,
            )
    gui_enable_gallery.input(
            fn=partial(save_and_load_gui, name="enable_gallery"),
            inputs=[gui_enable_gallery],
            outputs=gui_outputs,
            )

    # shortcut to load to queued gallery
    btn_qg_new.click(
            fn=qg_add_to_new,
            inputs=[row[1] for row in queued_galleries],
            outputs=[row[1] for row in queued_galleries],
            show_progress=False,
            )
    btn_qg_add.click(
            fn=qg_add_to_latest,
            inputs=[row[1] for row in queued_galleries],
            outputs=[row[1] for row in queued_galleries],
            show_progress=False,
            )

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
    clear_cache_btn.click(
            fn=clear_cache,
            )
    # force sound preprocessing for the first audio
    force_sound_processing_btn.click(
            fn=force_sound_processing,
            outputs=[audio_slots[0]],
            ).then(
                fn=transcribe,
                inputs=[audio_slots[0]],
                outputs=[txt_audio],
                preprocess=False,
                postprocess=False,
                )
    # trigger transcription when first audio stops recording
    audio_slots[0].stop_recording(
            fn=transcribe,
            inputs=[audio_slots[0]],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            ).success(
                    fn=alfred,
                    inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                    outputs=[txt_chatgpt_cloz],
                    preprocess=False,
                    postprocess=False,
                    )

    # load output elem if clicked
    tab_logging.select(fn=get_log, outputs=[output_elem])
    logging_reload.click(fn=get_log, outputs=[output_elem])

    # mark the previous car
    mark_previous.click(
            fn=mark_previous_notes,
            show_progress=False,
            )

    # suspend the previous cards
    suspend_previous.click(
            fn=suspend_previous_notes,
            show_progress=False,
            )

    # darkmode
    dark_mode_btn.click(fn=None, js=darkmode_js, show_progress=False)

    # sync anki
    sync_btn.click(fn=sync_anki, show_progress=False)

    # kill threads before timeout
    kill_threads_btn.click(fn=kill_threads, show_progress=False)

    # display card status
    update_status_btn.click(
            fn=lambda: "Checking",
            outputs=[update_status_btn],
            show_progress=False,
            ).then(
                    fn=get_card_status,
                    inputs=[txt_chatgpt_cloz],
                    outputs=[update_status_btn],
                    preprocess=False,
                    postprocess=False,
                    show_progress=False,
                    )
    txt_chatgpt_cloz.input(
            fn=lambda: "?",
            outputs=[update_status_btn],
            preprocess=False,
            postprocess=False,
            trigger_mode="always_last",
            show_progress=False,
            )
    txt_audio.input(
            fn=lambda: "?",
            outputs=[update_status_btn],
            preprocess=False,
            postprocess=False,
            trigger_mode="always_last",
            show_progress=False,
            )


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
    stt_choice.change(
            fn=shared.pv.save_stt_choice,
            inputs=[stt_choice],
            show_progress=False,
            )

    # change some values to profile
    sld_whisp_temp.change(fn=shared.pv.save_sld_whisp_temp, inputs=[sld_whisp_temp], show_progress=False)
    sld_buffer.change(fn=shared.pv.save_sld_buffer, inputs=[sld_buffer], show_progress=False)
    sld_temp.change(fn=shared.pv.save_sld_temp, inputs=[sld_temp], show_progress=False)
    gui_rolldirloadcheck.change(fn=shared.pv.save_dirload_check, inputs=[gui_rolldirloadcheck], show_progress=False)
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
            ).then(
                fn=shared.pv.save_gallery,
                inputs=[gallery],
                show_progress=False,
                ).then(
                        fn=get_img_source,
                        inputs=[gallery],
                        show_progress=False,
                        )


    # when gallery changes, save the image then run ocr
    gallery.change(
            fn=shared.pv.save_gallery,
            inputs=[gallery],
            show_progress=False,
            ).then(
                    fn=get_img_source,
                    inputs=[gallery],
                    show_progress=False,
                    )

    rst_img_btn.click(
            fn=reset_gallery,
            outputs=[gallery],
            show_progress=False,
            ).then(
                fn=shared.pv.save_gallery,
                inputs=[gallery],
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
                )

        # for the first gallery only: run ocr in advance
        if qg_cnt == 0:
            gal_.change(
                    fn=get_img_source,
                    inputs=[gal_],
                    show_progress=False,
                    )

        # add image
        add_.click(
                fn=get_image,
                inputs=[gal_],
                outputs=[gal_],
                show_progress=False,
                ).then(  # force deletion
                        fn=getattr(shared.pv, f"save_queued_gallery_{qg:03d}"),
                        inputs=[gal_],
                        show_progress=False,
                        )

        # send image
        send_.click(
                fn=lambda x: x,
                inputs=[gal_],
                outputs=[gallery],
                )

        # reset image
        rst_.click(
                fn=lambda: None,
                outputs=[gal_],
                show_progress=False,
                ).then(  # force deletion
                        fn=getattr(shared.pv, f"save_queued_gallery_{qg:03d}"),
                        inputs=[gal_],
                        show_progress=False,
                        )


        # ocr image
        ocr_.click(
                fn=ocr_image,
                inputs=[gal_],
                outputs=[source_txt],
                preprocess=False,
                postprocess=False,
                show_progress=False,
                )

    roll_gall_btn.click(
            fn=roll_queued_galleries,
            inputs=[row[1] for row in queued_galleries],
            outputs=[gallery] + [row[1] for row in queued_galleries],
            show_progress=False,
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
            show_progress=False,
            )

    rollaudio_1_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
            #show_progress=False,
            ).success(
                fn=transcribe,
                inputs=[audio_slots[0]],
                outputs=[txt_audio],
                preprocess=False,
                postprocess=False,
                ).then(
                    lambda: None,
                    outputs=[txt_chatgpt_cloz],
                    show_progress=False,
                ).success(
                        fn=dirload_splitted_last,
                        inputs=[gui_rolldirloadcheck],
                        outputs=[audio_slots[-1]],
                        preprocess=False,
                        postprocess=True,
                        # show_progress=False,
                        ).then(
                                fn=get_card_status,
                                inputs=[txt_chatgpt_cloz],
                                outputs=[update_status_btn],
                                preprocess=False,
                                postprocess=False,
                                show_progress=False,
                                )
    rollaudio_12_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
            show_progress=False,
            ).success(
                fn=transcribe,
                inputs=[audio_slots[0]],
                outputs=[txt_audio],
                preprocess=False,
                postprocess=False,
                ).success(
                    fn=alfred,
                    inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                    outputs=[txt_chatgpt_cloz],
                    preprocess=False,
                    postprocess=False,
                ).success(
                        fn=dirload_splitted_last,
                        inputs=[gui_rolldirloadcheck],
                        outputs=[audio_slots[-1]],
                        # preprocess=False,
                        # postprocess=False,
                        show_progress=False,
                        ).then(
                                fn=get_card_status,
                                inputs=[txt_chatgpt_cloz],
                                outputs=[update_status_btn],
                                preprocess=False,
                                postprocess=False,
                                show_progress=False,
                                )
    rollaudio_123_btn.click(
            fn=roll_audio,
            inputs=audio_slots,
            outputs=audio_slots,
            preprocess=False,
            postprocess=True,
            #show_progress=False,
            ).success(
                fn=transcribe,
                inputs=[audio_slots[0]],
                outputs=[txt_audio],
                preprocess=False,
                postprocess=False,
                ).success(
                    fn=alfred,
                    inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                    outputs=[txt_chatgpt_cloz],
                    preprocess=False,
                    postprocess=False,
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
                            show_progress=False,
                            ).then(
                                fn=dirload_splitted_last,
                                inputs=[gui_rolldirloadcheck],
                                outputs=[audio_slots[-1]],
                                show_progress=False,
                                ).then(
                                        fn=get_card_status,
                                        inputs=[txt_chatgpt_cloz],
                                        outputs=[update_status_btn],
                                        preprocess=False,
                                        postprocess=False,
                                        show_progress=False,
                                        )
    # # roll texts then set status to Rolling
    # gr.on(
    #         triggers=[rollaudio_1_btn.click, rollaudio_12_btn.click, rollaudio_123_btn.click],
    #         fn=lambda *x: list(x)[:-1] + [None],
    #         inputs=audio_slots_txts,
    #         outputs=audio_slots_txts,
    #         )

    # write Rolling in the status button
    gr.on(
            triggers=[rollaudio_1_btn.click, rollaudio_12_btn.click, rollaudio_123_btn.click],
            fn=lambda: "Rolling",
            outputs=[update_status_btn],
            show_progress=False,
            )

    # reset check for mark next
    gr.on(
            triggers=[anki_btn.click, txt_chatgpt_cloz.change],
            fn=lambda: False,
            outputs=[check_marked],
            show_progress=False,
            )

    # clicking this button will load from a user directory the next sounds and
    # images. This allow to use Voice2Anki on the computer but record the audio
    # on another distance device
    dir_load_btn.click(
            fn=dirload_splitted,
            inputs=[gui_rolldirloadcheck] + audio_slots,
            outputs=audio_slots,
            #show_progress=False,
            ).success(
                    fn=transcribe,
                    inputs=[audio_slots[0]],
                    outputs=[txt_audio],
                    preprocess=False,
                    postprocess=False,
                    ).success(
                        fn=alfred,
                        inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                        outputs=[txt_chatgpt_cloz],
                        preprocess=False,
                        postprocess=False,
                        ).then(
                                fn=get_card_status,
                                inputs=[txt_chatgpt_cloz],
                                outputs=[update_status_btn],
                                preprocess=False,
                                postprocess=False,
                                #show_progress=False,
                                )

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0]],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            )

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
            outputs=[txt_chatgpt_cloz],
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
            show_progress=False,
            ).then(
                    fn=lambda: False,
                    outputs=[check_marked],
                    #show_progress=False,
                    ).then(
                            fn=get_card_status,
                            inputs=[txt_chatgpt_cloz],
                            outputs=[update_status_btn],
                            preprocess=False,
                            postprocess=False,
                            #show_progress=False,
                            )

    # 1+2
    semiauto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0]],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
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
                    show_progress=False,
                    ).then(
                            fn=lambda: False,
                            outputs=[check_marked],
                            show_progress=False,
                            )

    # 1+2+3
    auto_btn.click(
            fn=transcribe,
            inputs=[audio_slots[0]],
            outputs=[txt_audio],
            preprocess=False,
            postprocess=False,
            ).success(
                fn=alfred,
                inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp, sld_buffer, llm_choice, txt_keywords, prompt_manag],
                outputs=[txt_chatgpt_cloz],
                preprocess=False,
                postprocess=False,
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
                    show_progress=False,
                    ).then(
                            fn=lambda: False,
                            outputs=[check_marked],
                            #show_progress=False,
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
            # preprocess=False,
            # postprocess=False,
            show_progress=False,
            )

    # darkmode js depending on variables
    init = demo.load(
            fn=shared.reset,
            show_progress=False,
            ).then(
                    fn=load_user_chain,
                    inputs=btn_chains,
                    outputs=btn_chains,
                    )

    # # trigger darkmode depending on time of day
    # if (datetime.now().hour <= 8 or datetime.now().hour >= 19):
    #     yel("Triggering darkmode because of time of day")
    #     init.then(fn=None, js=darkmode_js)

    # larger height to avoid scrolling up when changing tabs
    # this height is reset when another tab is selected to avoid cropping galleries for example
    init.then(fn=None, js=js_longer)  # startup height
    tab_main.select(fn=None, js=js_longer)  # long height when in Main
    # reset height when in other tabs
    gr.on(
            triggers=[tab_settings.select, tab_queues.select, tab_logging.select, tab_memories_and_buffer.select],
            fn=None,
            js=js_reset_height,
            )

    # gr.on(
    #         triggers=[a.change for a in audio_slots] + [transcript_btn.click, chatgpt_btn.click],
    #         fn=update_audio_slots_txts,
    #         inputs=audio_slots_txts,
    #         outputs=audio_slots_txts,
    #         show_progress=False,
    #         postprocess=False,
    #         preprocess=False,
    #         )

    init.then(
            fn=update_audio_slots_txts,
            inputs=[gui_enable_dirload] + audio_slots_txts,
            outputs=audio_slots_txts,
            postprocess=False,
            preprocess=False,
            every=0.25,
            trigger_mode="once",
            )
    init.then(
            fn=sync_anki,
            show_progress=False,
            )
    init.then(fn=None, js=js_load)

    if shared.pv.profile_name == "default":
        gr.Warning("Enter a profile then press enter.")
