from typing import Callable
import os
from beartype import beartype, BeartypeConf


if "VOICE2ANKI_TYPECHECKING" not in os.environ:
    os.environ["VOICE2ANKI_TYPECHECKING"] = "warn"

if os.environ["VOICE2ANKI_TYPECHECKING"] == "full":
    def optional_typecheck(func: Callable) -> Callable:
        return func
elif os.environ["VOICE2ANKI_TYPECHECKING"] == "crash":
    optional_typecheck = beartype
elif os.environ["VOICE2ANKI_TYPECHECKING"] == "warn":
    optional_typecheck = beartype(
        conf=BeartypeConf(violation_type=UserWarning))
elif os.environ["VOICE2ANKI_TYPECHECKING"] == "disabled":
    @beartype
    def optional_typecheck(func: Callable) -> Callable:
        return func
else:
    raise ValueError("Unexpected VOICE2ANKI_TYPECHECKING env value")
