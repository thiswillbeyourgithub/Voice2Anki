import os
import fire
from pathlib import Path

from utils.logger import whi, yel, red, print_db
from utils.shared_module import shared

# misc init values
Path("./cache").mkdir(exist_ok=True)

os.environ["PYTHONTRACEMALLOC"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def start_voice2formattedtext(
        backend="anki",
        print_db_then_exit=False,
        share=False,
        open_browser=False,
        debug=False,
        authentication=True,
        localnetwork=True,
        use_ssl=True,
        media_folder=None,
        memory_metric="embeddings",
        disable_tracing=True,
        disable_timeout=False,
        compact_js=True,
        *args,
        **kwargs
        ):
    """
    Parameters
    ----------
    backend: str, default 'anki'
        used to allow to select anki or markdown, but the markdown backend
        is currently deprecated.
    print_db_then_exit: str
        if a string, must be the name of a database from ./databases
        Will just output the content of the database as json then quit.
        Example value: "anki_whisper.db"
    share: bool, default False
        will create a url reachable from the global internet
    open_browser: bool, default False
        automatically open the browser
    debug: bool, default False
        increase verbosity
    authentication: bool, default True
        if True, will use the login/password pairs specified in VoiceToFormattedText.py
        This if forced to True if share is True
    localnetwork: bool, default True
        restrict access to the local network only
    use_ssl: bool, default True
        if True, will use the ssl configuration specified in VoiceToFormattedText.py
        Disable if share is used as self signed certificate mess with it.
    media_folder: str, default None
        optional anki media database location
    memory_metric: str, default "embeddings"
        if "length", will not use embeddings to improve the memory filtering
        but instead rely on finding memories with adequate length.
    disable_tracing: bool, default True
        if True, disables the decorator that indicates which function were
        called
    disable_timeout: bool, default False
        if True, disables the decorator that creates a thread used for
        timeout of long functions
    compact_js: bool, default True
        if True, will regularly run a js code to disable some controls from
        the audio components that can be too large for mobile device.
    """
    if "help" in kwargs or "h" in args:
        return help(start_voice2formattedtext)

    if isinstance(print_db_then_exit, str):
        db_list = [str(f.name) for f in Path("./databases/").rglob("*db")]
        if print_db_then_exit in db_list:
            return print_db(print_db_then_exit)
        else:
            raise ValueError(
                f"Unexpected {print_db_then_exit} value, should be "
                f"a value from {','.join(db_list)}")
    else:
        assert print_db_then_exit is False, "Invalid value for print_db_then_exit"

    assert backend == "anki", "only the 'anki' backend is currently supported."
    assert memory_metric in ["embeddings", "length"], "Invalid memory_metric"

    whi("Starting VoiceToFormattedText\n")
    if args:
        raise Exception(f"Unexpected arguments: {args}")
    if kwargs:
        raise Exception(f"Unexpected arguments: {kwargs}")

    if share:
        yel("Sharing enabled")
    else:
        whi("Sharing disabled")
    if open_browser:
        yel("Opening browser")
    else:
        whi("Not opening browser.")
    if debug:
        yel("Debug mode enabled")
    if authentication or share:
        auth_args = {"auth": ("v2ft", "v2ft"), "auth_message": "Please login"}
        yel("Authentication enabled")
    else:
        auth_args = {}
    if localnetwork:
        server = "0.0.0.0"
        yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")
    else:
        server = None

    shared.memory_metric = memory_metric
    shared.media_folder = media_folder
    shared.debug = debug
    shared.disable_tracing = disable_tracing
    shared.disable_timeout = disable_timeout
    shared.compact_js = compact_js

    if backend == "anki":
        shared.backend = "anki"
        from utils.gui_anki import demo_anki as demo
    elif backend == "markdown":
        raise NotImplementedError("markdown backend was deprecated for lack of use.")
        shared.backend = "markdown"
        from utils.gui_markdown import demo_markdown as demo
    else:
        raise ValueError(backend)

    if (not share) and use_ssl and Path("./utils/ssl").exists() and Path("./utils/ssl/key.pem").exists() and Path("./utils/ssl/cert.pem").exists():
        ssl_args = {
                "ssl_keyfile": "./utils/ssl/key.pem",
                "ssl_certfile": "./utils/ssl/cert.pem",
                "ssl_keyfile_password": "fd5d63390f1a45427acfe20dd0e24a95",  # random md5
                "ssl_verify": False,  # allow self signed
                }
    else:
        red("Will not use SSL")
        ssl_args = {}

    # demo.queue()
    demo.launch(
            share=share,
            **auth_args,
            inbrowser=open_browser,
            quiet=False,
            debug=debug,
            prevent_thread_lock=True if debug else False,
            show_error=True,
            show_api=False,
            server_name=server,
            server_port=7860,
            **ssl_args,
            )

if __name__ == "__main__":
    instance = fire.Fire(start_voice2formattedtext)
