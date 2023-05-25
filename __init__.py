from utils.logger import whi, yel
import sys
from main import demo
from utils.user_db import authentification

if __name__ == "__main__":
    args = sys.argv[1:]

    to_share = False
    auth = None
    op_br = True
    debug = False
    auth_message = None

    if args:
        whi(f"Startup arguments: {args}")

        if "--share" in args:
            to_share = True
            yel("Sharing enabled")

        if "--auth" in args:
            auth = authentification
            yel("Authentification enabled")
            # TODO: auth_message
        if "--open" in args:
            of_br = True
            yel("Opening browser")
        if "debug" in args:
            debug = True
            yel("Debug mode enabled")

    if not to_share:
        whi("Authentification disabled")
    if not auth:
        whi("Sharing disabled")
    if not op_br:
        whi("Not openning browser.")

    if to_share and not auth:
        raise Exception("Sharing mode without authentification is disabled")

    demo.queue(concurrency_count=3)

    demo.launch(
            share=to_share,
            auth=auth,
            auth_message="Please login",
            inbrowser=op_br,
            debug=debug,
            show_error=True,
            server_name="0.0.0.0",  # TODO, this make it sharable by hand?!
            server_port=7860,
            show_tips=True,
            # ssl_keyfile=,
            # ssl_certfile=,
            # ssl_keyfile_password=,
            )
