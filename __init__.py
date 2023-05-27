from utils.logger import whi, yel
import sys
from main import demo

if __name__ == "__main__":
    args = sys.argv[1:]

    to_share = False
    op_br = False
    debug = False

    if args:
        whi(f"Startup arguments: {args}")

        if "--share" in args:
            to_share = True
            yel("Sharing enabled")

        if "--browser" in args:
            op_br = True
            yel("Opening browser")
        if "debug" in args:
            debug = True
            yel("Debug mode enabled")

    if not to_share:
        whi("Sharing disabled")
    if not op_br:
        whi("Not opening browser.")

    demo.queue(concurrency_count=3)

    demo.launch(
            share=to_share,
            #auth=("g", "g"),
            #auth_message="Please login",
            inbrowser=op_br,
            debug=debug,
            show_error=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_tips=True,
            ssl_keyfile="./utils/ssl/key.pem",
            ssl_certfile="./utils/ssl/cert.pem",
            ssl_keyfile_password="fd5d63390f1a45427acfe20dd0e24a95",  # random md5
            ssl_verify=False,  # allow self signed
            )
