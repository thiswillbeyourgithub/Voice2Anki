import sys
from pathlib import Path

from utils.gui_anki import demo_anki
from utils.gui_markdown import demo_markdown
from utils.logger import whi, yel, red

# misc init values
Path("./cache").mkdir(exist_ok=True)

if __name__ == "__main__":
    whi("Starting VoiceToFormattedText\n")
    args = sys.argv[1:]

    # default argument
    to_share = False
    op_br = False
    debug = False
    auth_args = {}
    server = None

    if args:
        whi(f"Startup arguments: {args}")

        if "--share" in args:
            to_share = True
            yel("Sharing enabled")

        if "--browser" in args:
            op_br = True
            yel("Opening browser")
        if "--debug" in args:
            debug = True
            yel("Debug mode enabled")
        if "--auth" in args:
            auth_args = {"auth": ("user", "password"), "auth_message": "Please login"}
            yel("Authentication enabled")
        if "--localnetwork" in args:
            server = "0.0.0.0"
            yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")

    for ar in args:
        if ar not in ["--share", "--browser", "--debug", "--auth", "--localnetwork", "--backend=anki", "--backend=markdown"]:
            raise SystemExit(f"Invalid argument: '{ar}'")

    if "--backend=anki" in args:
        demo = demo_anki
    elif "--backend=markdown" in args:
        demo = demo_markdown
    else:
        raise Exception(f"Invalid backend: {args}")

    if not to_share:
        whi("Sharing disabled")
    if not op_br:
        whi("Not opening browser.")

    demo.queue(concurrency_count=3)

    if Path("./utils/ssl").exists() and Path("./utils/ssl/key.pem").exists() and Path("./utils/ssl/cert.pem").exists():
        ssl_args = {
                "ssl_keyfile": "./utils/ssl/key.pem",
                "ssl_certfile": "./utils/ssl/cert.pem",
                "ssl_keyfile_password": "fd5d63390f1a45427acfe20dd0e24a95",  # random md5
                "ssl_verify": False,  # allow self signed
                }
    else:
        red(f"SSL certificate or key not found, disabling https")
        ssl_args = {}

    demo.launch(
            share=to_share,
            **auth_args,
            inbrowser=op_br,
            debug=debug,
            show_error=True,
            server_name=server,
            server_port=7860,
            show_tips=True,
            **ssl_args,
            )
