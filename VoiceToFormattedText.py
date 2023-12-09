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
        backend="",
        print_db_then_exit=False,
        do_share=False,
        open_browser=False,
        debug=False,
        do_auth=False,
        localnetworkonly=False,
        use_ssl=True,
        media_folder=None,
        memory_metric="embeddings",
        *args,
        **kwargs
        ):
    """
    Parameters
    ----------
    backend: str
        either "anki" or "markdown"
    print_db_then_exit: str
        if a string, must be the name of a database from ./databases
        Will just output the content of the database as json then quit.
        Example value: "anki_whisper.db"
    do_share: bool, default False
        will create a url reachable from the global internet
    open_browser: bool, default False
        automatically open the browser
    debug: bool, default False
        increase verbosity
    do_auth: bool, default False
        if True, will use the login/password pairs specified in VoiceToFormattedText.py
        This if forced to True if do_share is True
    localnetworkonly: bool, default False
        restrict access to the local network only
    use_ssl: bool, default True
        if True, will use the ssl configuration specified in VoiceToFormattedText.py
    media_folder: str, default None
        optional anki media database location
    memory_metric: str, default "embeddings"
        if "length", will not use embeddings to improve the memory filtering
        but instead rely on finding memories with adequate length.
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

    assert str(backend).lower() in ["anki", "markdown"], f"Backend argument has to be either 'anki' or 'markdown', not {backend}"
    backend = str(backend).lower()
    assert backend in ["anki", "markdown"], (
            "backend must be either 'anki' "
            "or 'markdown'")
    assert memory_metric in ["embeddings", "length"], "Invalid memory_metric"

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
    if do_auth or do_share:
        auth_args = {"auth": ("v2ft", "v2ft"), "auth_message": "Please login"}
        yel("Authentication enabled")
    else:
        auth_args = {}
    if localnetworkonly:
        server = "0.0.0.0"
        yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")
    else:
        server = None

    shared.VERSION = 0.2
    shared.memory_metric = memory_metric
    shared.media_folder = media_folder
    shared.debug = debug

    if backend == "anki":
        shared.backend = "anki"
        from utils.gui_anki import demo_anki as demo
    elif backend == "markdown":
        shared.backend = "markdown"
        from utils.gui_markdown import demo_markdown as demo
    else:
        raise ValueError(backend)

    if use_ssl and Path("./utils/ssl").exists() and Path("./utils/ssl/key.pem").exists() and Path("./utils/ssl/cert.pem").exists():
        ssl_args = {
                "ssl_keyfile": "./utils/ssl/key.pem",
                "ssl_certfile": "./utils/ssl/cert.pem",
                "ssl_keyfile_password": "fd5d63390f1a45427acfe20dd0e24a95",  # random md5
                "ssl_verify": False,  # allow self signed
                }
    else:
        red("Will not use SSL")
        ssl_args = {}

    demo.queue()
    demo.launch(
            share=do_share,
            **auth_args,
            inbrowser=open_browser,
            debug=debug,
            show_error=True,
            server_name=server,
            server_port=7860,
            **ssl_args,
            )

if __name__ == "__main__":
    instance = fire.Fire(start_voice2formattedtext)
