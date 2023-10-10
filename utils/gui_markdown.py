from pathlib import Path
import gradio as gr

from .profiles import get_profiles, switch_profile, ValueStorage, save_path
from .main_markdown import transcribe, alfred, main, auto_mode, semiauto_mode

from .logger import get_log, whi
from .memory import recur_improv
from .media import reset_audio, audio_saver, load_next_audio

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
pv = ValueStorage("reload")

with gr.Blocks(analytics_enabled=False, title="VoiceToFormattedText - Markdown", theme=theme) as demo_markdown:

    gr.Markdown("VoiceToFormattedText - Markdown")

    # hidden, to store the request answer from chatgpt
    txt_chatgpt_tkncost = gr.Textbox(value=None, visible=False, placeholder="this string should never appear")

    with gr.Row():
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=10):
                        txt_profile = gr.Textbox(value=pv["latest_profile"], placeholder=",".join(get_profiles()), label="Profile")
                    with gr.Column(scale=1):
                         dark_mode_btn = gr.Button("Dark Mode", variant="secondary").style(full_width=True)
                txt_mdpath = gr.Textbox(value=pv["txt_mdpath"], label="Markdown file", max_lines=1, placeholder="markdown path")
                with gr.Row():
                    with gr.Column(scale=1):
                        txt_whisp_lang = gr.Textbox(value=pv["txt_whisp_lang"], label="SpeechToText lang", placeholder="language of the recording, e.g. fr")
                    with gr.Column(scale=9):
                        with gr.Row():
                            txt_whisp_prompt = gr.Textbox(value=pv["txt_whisp_prompt"], label="SpeechToText context", placeholder="context for whisper")
                            txt_chatgpt_context = gr.Textbox(value=pv["txt_chatgpt_context"], label="LLM context", placeholder="context for ChatGPT")

    with gr.Row():
        with gr.Column(scale=1):
            rst_audio_btn = gr.Button(value="Clear audio", variant="secondary")
            audio_mp3_1 = gr.Audio(source="microphone", type="filepath", label="Audio", format="mp3", value=None).style(size="sm")
            audio_mp3_2 = gr.Audio(source="microphone", type="filepath", label="Audio", format="mp3", value=None).style(size="sm")
            audio_mp3_3 = gr.Audio(source="microphone", type="filepath", label="Audio", format="mp3", value=None).style(size="sm")
            audio_mp3_4 = gr.Audio(source="microphone", type="filepath", label="Audio", format="mp3", value=None).style(size="sm")
            audio_mp3_5 = gr.Audio(source="microphone", type="filepath", label="Audio", format="mp3", value=None).style(size="sm")
            load_audio_btn = gr.Button(value="Roll + 1+2", variant="secondary")
        with gr.Column(scale=3):
            txt_audio = gr.Textbox(label="Transcript", lines=5, max_lines=10, placeholder="The transcript of the audio recording will appear here")
            txt_chatgpt_outputstr = gr.Textbox(label="ChatGPT output", lines=5, max_lines=10, placeholder="The markdown text will appear here")

    with gr.Row():
        transcript_btn = gr.Button(value="1. Transcribe audio", variant="secondary")
        chatgpt_btn = gr.Button(value="2. Transcript to md text", variant="secondary")
        tooutput_btn = gr.Button(value="3. md text to file", variant="secondary")

    with gr.Row():
        semiauto_btn = gr.Button(value="1+2. Speech to md text", variant="primary")
        auto_btn = gr.Button(value="1+2+3. Autopilot", variant="primary")

    with gr.Row():
        with gr.Column(scale=9):
            with gr.Row():
                improve_btn = gr.Button(value="Feed prompt back to LLM", variant="primary")
                sld_improve = gr.Slider(minimum=0, maximum=10, value=None, step=1, label="Feedback priority")
        with gr.Column(scale=1):
            with gr.Row():
                sld_max_tkn = gr.Slider(minimum=500, maximum=15000, value=pv["sld_max_tkn"], step=500, label="LLM maxhistory token size.")
                sld_temp = gr.Slider(minimum=0, maximum=2, value=pv["temperature"], step=0.1, label="LLM temperature")

    # output
    mdp = pv["txt_mdpath"]
    if mdp and Path(mdp).exists():
        output_content = "\n".join(Path(mdp).read_text().strip().split("\n")[-100:])
    else:
        output_content = ""

    gr.Markdown("Markdown file content:")
    mdoutput_elem = gr.Markdown(value=output_content)
    logoutput_elem = gr.Textbox(value=get_log, label="Logging", lines=10, max_lines=100, every=1, interactive=False, placeholder="this string should never appear")

    # events
    # darkmode
    dark_mode_btn.click(fn=None, _js=darkmode_js)

    # change profile and load previous data
    txt_profile.submit(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_mdpath, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, audio_mp3_1, txt_audio, txt_chatgpt_outputstr, txt_profile])
    txt_profile.blur(
            fn=switch_profile,
            inputs=[txt_profile],
            outputs=[txt_mdpath, txt_chatgpt_context, txt_whisp_prompt, txt_whisp_lang, audio_mp3_1, txt_audio, txt_chatgpt_outputstr, txt_profile])
    txt_mdpath.submit(fn=save_path, inputs=[txt_profile, txt_mdpath])


    # audio
    rst_audio_btn.click(
            fn=reset_audio,
            inputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            outputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5])
    audio_mp3_1.change(fn=audio_saver().n1, inputs=[txt_profile, audio_mp3_1])
    audio_mp3_2.change(fn=audio_saver().n2, inputs=[txt_profile, audio_mp3_2])
    audio_mp3_3.change(fn=audio_saver().n3, inputs=[txt_profile, audio_mp3_3])
    audio_mp3_4.change(fn=audio_saver().n4, inputs=[txt_profile, audio_mp3_4])
    audio_mp3_5.change(fn=audio_saver().n5, inputs=[txt_profile, audio_mp3_5])
    load_audio_btn.click(
            fn=load_next_audio,
            inputs=[txt_profile, audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5],
            outputs=[audio_mp3_1, audio_mp3_2, audio_mp3_3, audio_mp3_4, audio_mp3_5]
            ).then(
                    fn=semiauto_mode,
                    inputs=[audio_mp3_1, txt_audio, txt_whisp_prompt, txt_whisp_lang, txt_chatgpt_tkncost, txt_chatgpt_outputstr, txt_chatgpt_context, txt_mdpath, txt_profile, sld_max_tkn, sld_temp],
                    outputs=[txt_audio, txt_chatgpt_tkncost, txt_chatgpt_outputstr])

    # load previous values
    # audio_mp3_1.value = pv["audio_mp3_1"]
    # audio_mp3_2.value = pv["audio_mp3_2"]
    # audio_mp3_3.value = pv["audio_mp3_3"]
    # audio_mp3_4.value = pv["audio_mp3_4"]
    # audio_mp3_5.value = pv["audio_mp3_5"]

    # send to whisper
    transcript_btn.click(
            fn=transcribe,
            inputs=[audio_mp3_1, txt_whisp_prompt, txt_whisp_lang, txt_profile],
            outputs=[txt_audio])

    # send to chatgpt
    chatgpt_btn.click(
            fn=alfred,
            inputs=[txt_audio, txt_chatgpt_context, txt_profile, sld_max_tkn, sld_temp],
            outputs=[txt_chatgpt_outputstr, txt_chatgpt_tkncost])

    # send to file
    tooutput_btn.click(
            fn=main,
            inputs=[
                audio_mp3_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,

                txt_chatgpt_context,
                txt_mdpath,

                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                mdoutput_elem,
                ],
            )

    # 1+2
    semiauto_btn.click(
            fn=semiauto_mode,
            inputs=[
                audio_mp3_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,

                txt_chatgpt_context,
                txt_mdpath,

                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                mdoutput_elem,
                ],
            )

    # 1+2+3
    auto_btn.click(
            fn=auto_mode,
            inputs=[
                audio_mp3_1,
                txt_audio,
                txt_whisp_prompt,
                txt_whisp_lang,

                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,

                txt_chatgpt_context,
                txt_mdpath,

                txt_profile,
                sld_max_tkn,
                sld_temp,
                ],
            outputs=[
                txt_audio,
                txt_chatgpt_tkncost,
                txt_chatgpt_outputstr,
                mdoutput_elem,
                ],
            )

    improve_btn.click(
            fn=recur_improv,
            inputs=[
                txt_profile,
                txt_audio,
                txt_whisp_prompt,
                txt_chatgpt_outputstr,
                txt_chatgpt_context,
                sld_improve,
                ],
            )

