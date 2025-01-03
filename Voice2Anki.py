import gradio as gr
import os
import sys
import fire
from pathlib import Path, PosixPath
from typing import Optional, Union
import pdb
import faulthandler
import traceback

from utils.typechecker import optional_typecheck, beartype
from utils.logger import whi, yel, red, print_db
from utils.shared_module import shared


# misc init values
Path("./cache").mkdir(exist_ok=True)

os.environ["PYTHONTRACEMALLOC"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#@optional_typecheck
@beartype
def start_Voice2Anki(
    print_db_then_exit: Optional[str] = None,
    nb_audio_slots: Union[str, int] = 3,

    gui: bool = True,
    share: bool = False,
    open_browser: bool = False,
    debug: bool = False,
    authentication: bool = True,
    localnetwork: bool = True,
    use_ssl: bool = True,
    anki_media_folder: Optional[Union[str, PosixPath]] = None,
    disable_tracing: bool = False,
    disable_timeout: bool = True,
    disable_smartcache: bool = False,
    widen_screen: bool = True,
    big_font: bool = False,
    port: int = 7860,
    *args,
    **kwargs
    ) -> Optional[Union[str, gr.Blocks]]:
    """
    Parameters
    ----------
    print_db_then_exit: str
        if a string, must be the name of a database from ./databases
        Will just output the content of the database as json then quit.
        Example value: "anki_whisper.db"
    nb_audio_slots: int or str, default 3
        Number of audio slot. If 'auto', only used by cli
    gui: bool, default True
        False to use cli
    share: bool, default False
        will create a url reachable from the global internet
    open_browser: bool, default False
        automatically open the browser
    debug: bool, default False
        increase verbosity, also open the debugger in case of issue
    authentication: bool, default True
        if True, will use the login/password pairs specified in Voice2Anki.py
        This if forced to True if share is True
    localnetwork: bool, default True
        restrict access to the local network only
    use_ssl: bool, default True
        if True, will use the ssl configuration specified in Voice2Anki.py
        Disable if share is used as self signed certificate mess with it.
    anki_media_folder: str, default None
        optional anki media database location
    disable_tracing: bool, default False
        if True, disables the decorator that indicates which function were
        called
    disable_timeout: bool, default True
        if True, disables the decorator that creates a thread used for
        timeout of long functions
    disable_smartcache: bool, default False
        if True, disables smartcache, which is what tells a func to wait for
        results if they are already being cached by another call, to avoid
        concurrent calls
    widen_screen: bool, default True
        if True, will force width of app to be 100%. Might be a problem
        for some widen screens, but very handy for mobile and tablet use.
    big_font: bool, default False
        increase font size of text elements
    port: int, default 7860
        default port to use
    """
    if "help" in kwargs or "h" in args or "usage" in kwargs:
        return help(start_Voice2Anki)

    if isinstance(print_db_then_exit, str):
        db_list = [str(f.name) for f in Path("./databases/").rglob("*db")]
        if print_db_then_exit in db_list:
            return print_db(print_db_then_exit)
        else:
            raise ValueError(
                f"Unexpected {print_db_then_exit} value, should be "
                f"a value from {', '.join(db_list)}")
    else:
        assert print_db_then_exit is None, "Invalid value for print_db_then_exit. You must specify the name of a db"

    whi("Starting Voice2Anki\n")
    if args:
        raise Exception(f"Unexpected arguments: {args}")
    if gui:
        assert not kwargs, f"Unexpected kwarguments if using gui: {kwargs}"
        assert isinstance(nb_audio_slots, int), "nb_audio_slots must be int if using gui"


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
        def handle_exception(exc_type, exc_value, exc_traceback):
            if not issubclass(exc_type, KeyboardInterrupt):
                @optional_typecheck
                def p(message: str) -> None:
                    "print error in red if possible"
                    try:
                        red(message)
                    except Exception as err:
                        print(message)
                p("\n--verbose was used so opening debug console at the "
                    "appropriate frame. Press 'c' to continue to the frame "
                    "of this print.")
                [p(line) for line in traceback.format_tb(exc_traceback)]
                p(str(exc_type) + " : " + str(exc_value))
                pdb.post_mortem(exc_traceback)
                p("You are now in the exception handling frame.")
                breakpoint()
                sys.exit(1)

        sys.excepthook = handle_exception
        faulthandler.enable()

    if authentication or share:
        auth_args = {"auth": [("v2a", "v2a"), ("V2a", "v2a")], "auth_message": "Please login"}
        yel("Authentication enabled")
    else:
        auth_args = {}
    if localnetwork:
        server = "0.0.0.0"
        yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")
    else:
        server = None

    shared.audio_slot_nb = nb_audio_slots
    shared.anki_media = anki_media_folder
    shared.debug = debug
    shared.disable_tracing = disable_tracing
    shared.disable_timeout = disable_timeout
    shared.disable_smartcache = disable_smartcache
    shared.widen_screen = widen_screen
    shared.big_font = big_font
    shared.client_type = "gui" if gui else "cli"

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

    if gui:
        whi("Launching GUI")
        from utils.gui import demo

        # queueing seems to make things way slower
        # demo.queue()
        demo.launch(
                share=share,
                **auth_args,
                inbrowser=open_browser,
                quiet=False,
                debug=debug,
                # prevent_thread_lock=True if debug else False,
                max_threads=5,  # if not debug else 1,  # default 40
                show_error=True,
                show_api=False,
                server_name=server,
                server_port=port,
                # inline=True,
                width="100%",  # used if inline is True
                enable_monitoring=False,
                allowed_paths=["/tmp/gradio"],
                **ssl_args,
                )
        return demo
    else:
        whi("Not launching GUI, using cli mode")
        from utils.cli import Cli
        try:
            _ = Cli(
                nb_audio_slots=nb_audio_slots,
                **kwargs,
            )
        except KeyboardInterrupt as e:
            raise SystemExit("Quitting")
        return None


if __name__ == "__main__":
    try:
        demo = fire.Fire(start_Voice2Anki)
    except IndexError as e:
        print(f"Quitting ('{e}'))")
