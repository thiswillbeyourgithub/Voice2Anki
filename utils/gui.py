import gradio as gr

from .profiles import get_profiles, switch_profile, previous_values
from .main import transcribe, alfred, main, auto_mode, semiauto_mode

from .logger import get_log
from .memory import recur_improv
from .media import get_image, reset_audio, reset_image, save_audio, load_next_audio

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
pv = previous_values()
# if last profile was not 'default', switch directly to the last used
if pv["profile"] != "default":
    pv = previous_values(pv["profile"])

with gr.Blocks(analytics_enabled=False, title="WhisperToAnki", theme=theme) as demo:

    gr.Markdown("WhisperToAnki")

    # hidden, to store the request answer from chatgpt
    txt_chatgpt_tkncost = gr.Textbox(value=None, visible=False)

    with gr.Row():
        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(value=pv["gallery"], label="Source images").style(columns=[2], rows=[1], object_fit="scale-down", height="auto", container=True)
                rst_img_btn = gr.Button(value="Clear then add", variant="secondary").style(size="sm")
                img_btn = gr.Button(value="Add image from clipboard", variant="primary").style(size="sm")
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=10):
                        txt_profile = gr.Textbox(value=pv["profile"], placeholder=",".join(get_profiles()), label="Profile")
                    with gr.Column(scale=1):
                         dark_mode_btn = gr.Button("Dark Mode", variant="secondary").style(full_width=True)
                txt_deck = gr.Textbox(value=pv["txt_deck"], label="Deck name", max_lines=1)
                txt_tags = gr.Textbox(value=pv["txt_tags"], label="Tags", lines=1)
                with gr.Row():
                    with gr.Column(scale=1):
                        txt_whisp_lang = gr.Textbox(value=pv["txt_whisp_lang"], label="SpeechToText lang")
                    with gr.Column(scale=9):
                        with gr.Row():
                            txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], label="SpeechToText context")
                            txt_chatgpt_context = gr.Textbox(value=pv["txt_chatgpt_context"], label="LLM context")

    with gr.Row():
        with gr.Column(scale=1):
            rst_audio_btn = gr.Button(value="Clear audio", variant="secondary")
            audio_numpy_1 = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=pv["audio_numpy_1"]).style(size="sm")
            audio_numpy_2 = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=None).style(size="sm")
            audio_numpy_3 = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=None).style(size="sm")
            audio_numpy_4 = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=None).style(size="sm")
            audio_numpy_5 = gr.Audio(source="microphone", type="numpy", label="Audio", format="wav", value=None).style(size="sm")
            load_audio_btn = gr.Button(value="Roll + 1+2", variant="secondary")
        with gr.Column(scale=3):
            txt_audio = gr.Textbox(value=pv["txt_audio"], label="Transcript", lines=5, max_lines=10)
            txt_chatgpt_cloz = gr.Textbox(value=pv["txt_chatgpt_cloz"], label="LLM cloze(s)", lines=5, max_lines=10)

    with gr.Row():
        transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary")
        chatgpt_btn = gr.Button(value="2. Transcript to cloze", variant="secondary")
        anki_btn = gr.Button(value="3. Cloze to Anki", variant="secondary")

    with gr.Row():
        semiauto_btn = gr.Button(value="1+2. Speech to Cloze", variant="primary")
        auto_btn = gr.Button(value="1+2+3. Autopilot", variant="primary")

    with gr.Row():
        with gr.Column(scale=9):
            with gr.Row():
                improve_btn = gr.Button(value="Feed prompt back to LLM", variant="primary")
                sld_improve = gr.Slider(minimum=0, maximum=10, value=None, step=1, label="Feedback priority")
        with gr.Column(scale=1):
            with gr.Row():
                sld_max_tkn = gr.Slider(minimum=500, maximum=3500, value=pv["max_tkn"], step=500, label="LLM maxhistory token size")
                sld_temp = gr.Slider(minimum=0, maximum=2, value=pv["temperature"], step=0.1, label="LLM temperature")

    # output
    output_elem = gr.Textbox(value=get_log, label="Logging", lines=10, max_lines=100, every=0.3, interactive=False)

    # events
    # darkmode
    dark_mode_btn.click(fn=None, _js=darkmode_js)

    # change profile and load previous data
    txt_profile.submit(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_numpy_1, txt_audio, txt_chatgpt_cloz, txt_profile])
    txt_profile.blur(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_deck, txt_tags, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, gallery, audio_numpy_1, txt_audio, txt_chatgpt_cloz, txt_profile])

    # image
    img_btn.click(
            fn=get_image,
            inputs=[gallery],
            outputs=[gallery])
    rst_img_btn.click(
            fn=reset_image,
            outputs=[gallery]
            ).then(
                    fn=get_image,
                    inputs=[gallery],
                    outputs=[gallery])

    # audio
    rst_audio_btn.click(
            fn=reset_audio,
            inputs=[audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5],
            outputs=[audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5])
    audio_numpy_1.change(fn=save_audio, inputs=[txt_profile, audio_numpy_1])
    load_audio_btn.click(
            fn=load_next_audio,
            inputs=[audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5],
            outputs=[audio_numpy_1, audio_numpy_2, audio_numpy_3, audio_numpy_4, audio_numpy_5]
            ).then(
                    fn=semiauto_mode,
                    inputs=[audio_numpy_1, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_tkncost, txt_chatgpt_cloz, txt_chatgpt_context, txt_deck, txt_tags, gallery, txt_profile, sld_max_tkn, sld_temp],
                    outputs=[txt_audio, txt_chatgpt_tkncost, txt_chatgpt_cloz])

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_numpy_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio])

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp],
            outputs=[txt_chatgpt_cloz, txt_chatgpt_tkncost])

    # send to anki
    anki_btn.click(
            fn=main,
            inputs=[
                audio_numpy_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ],
            )

    # 1+2
    semiauto_btn.click(
            fn=semiauto_mode,
            inputs=[
                audio_numpy_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,
                ],
            )

    # 1+2+3
    auto_btn.click(
            fn=auto_mode,
            inputs=[
                audio_numpy_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_cloz,

                txt_chatgpt_context,
                txt_deck,
                txt_tags,

                gallery,
                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
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
                ],
            )

demo.queue()
