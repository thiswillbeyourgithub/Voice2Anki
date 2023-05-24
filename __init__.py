from utils.logger import whi, yel
import sys
from main import demo
from utils.user_db import authentification

if __name__ == "__main__":
    args = sys.argv[1:]

    to_share = False
    auth = None

    if args:
        whi(f"Startup arguments: {args}")

        if "share" in args:
            to_share = True
            yel("Sharing enabled")

        if "auth" in args:
            auth = authentification
            yel("Authentification enabled")
    if not to_share:
        whi("Authentification disabled")
    if not auth:
        whi("Sharing disabled")

    if to_share and not auth:
        raise Exception("Sharing mode without authentification is disabled")

    demo.queue()

    demo.launch(
            share=to_share,
            auth=auth,
            )
