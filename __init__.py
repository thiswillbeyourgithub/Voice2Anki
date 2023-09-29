import fire
import sys
from pathlib import Path

from utils.logger import whi, yel, red
from utils.misc import backend_config

# misc init values
Path("./cache").mkdir(exist_ok=True)

def start_voice2formattedtext(
        backend,
        do_share=False,
        open_browser=False,
        debug=False,
        do_auth=False,
        localnetworkonly=False,
        use_ssl=False,
        *args,
        **kwargs
        ):
    """
    Parameters
    ----------
    backend: str
        either "anki" or "markdown"
    do_share: bool, default False
        will create a url reachable from the global internet
    open_browser: bool, default False
        automatically open the browser
    debug: bool, default False
        increase verbosity
    do_auth: bool, default False
        if True, will use the login/password pairs specified in __init__.py
    localnetworkonly: bool, default False
        restrict access to the local network only
    use_ssl: bool, default False
        if True, will use the ssl configuration specified in __init__.py
    """
    assert str(backend).lower() in ["anki", "markdown"], f"Backend argument has to be either 'anki' or 'markdown', not {backend}"
    backend = str(backend).lower()

    whi("Starting VoiceToFormattedText\n")
    if args:
        raise Exception(f"Unexpected arguments: {args}")
    if kwargs:
        raise Exception(f"Unexpected arguments: {kwargs}")

    if do_share:
        yel("Sharing enabled")
    else:
        whi("Sharing disabled")
    if open_browser:
        yel("Opening browser")
    else:
        whi("Not opening browser.")
    if debug:
        yel("Debug mode enabled")
    if do_auth:
        auth_args = {"auth": ("user", "password"), "auth_message": "Please login"}
        yel("Authentication enabled")
    else:
        auth_args = {}
    if localnetworkonly:
        server = "0.0.0.0"
        yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")
    else:
        server = None

    if backend == "anki":
        backend_config.backend = "anki"
        from utils.gui_anki import demo_anki as demo
    elif backend == "markdown":
        backend_config.backend = "markdown"
        from utils.gui_markdown import demo_markdown as demo
    else:
        raise ValueError(backend)

    demo.queue(concurrency_count=3)

    if use_ssl and Path("./utils/ssl").exists() and Path("./utils/ssl/key.pem").exists() and Path("./utils/ssl/cert.pem").exists():
        ssl_args = {
                "ssl_keyfile": "./utils/ssl/key.pem",
                "ssl_certfile": "./utils/ssl/cert.pem",
                "ssl_keyfile_password": "fd5d63390f1a45427acfe20dd0e24a95",  # random md5
                "ssl_verify": False,  # allow self signed
                }
    else:
        red(f"Will not use SSL")
        ssl_args = {}

    demo.launch(
            share=do_share,
            **auth_args,
            inbrowser=open_browser,
            debug=debug,
            show_error=True,
            server_name=server,
            server_port=7860,
            show_tips=True,
            **ssl_args,
            )
if __name__ == "__main__":
    instance = fire.Fire(start_voice2formattedtext)
