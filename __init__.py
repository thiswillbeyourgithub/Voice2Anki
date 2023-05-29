import sys
from pathlib import Path

from main import demo
from utils.logger import whi, yel

if __name__ == "__main__":
    whi(f"Starting WhisperToAnki\n")
    args = sys.argv[1:]

    # default argument
    to_share = False
    op_br = False
    debug = False
    auth_args = {"auth": ("g", "g"), "auth_message": "Please login using g/g"}
    server = None

    wavfiles = [p for p in Path(".").iterdir() if str(p).endswith(".wav")]
    if wavfiles:
        ans = input(whi(f"Found {len(wavfiles)} .wav files in root folder. Can I delete them? (y/n)\n>"))
        if ans == "y":
            whi("Deleting wav files")
            [p.unlink() for p in wavfiles]

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
        if "--noauth" in args:
            auth_args = {}
            yel("Disabling authentication")
        if "--localnetwork" in args:
            server = "0.0.0.0"
            yel("Will be accessible on the local network. Use `ifconfig` to find your local IP adress.")

    for ar in args:
        if ar not in ["--share", "--browser", "--debug", "--noauth", "--localnetwork"]:
            raise SystemExit(f"Invalid argument: '{ar}'")

    if not to_share:
        whi("Sharing disabled")
    if not op_br:
        whi("Not opening browser.")

    demo.queue(concurrency_count=3)

    demo.launch(
            share=to_share,
            **auth_args,
            inbrowser=op_br,
            debug=debug,
            show_error=True,
            server_name=server,
            server_port=7860,
            show_tips=True,
            ssl_keyfile="./utils/ssl/key.pem",
            ssl_certfile="./utils/ssl/cert.pem",
            ssl_keyfile_password="fd5d63390f1a45427acfe20dd0e24a95",  # random md5
            ssl_verify=False,  # allow self signed
            )
