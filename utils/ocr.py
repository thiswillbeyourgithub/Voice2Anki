import os
from joblib import Memory
from OCR_with_format import OCR_with_format

from .logger import yel, trace, smartcache, cache_dir
from .typechecker import optional_typecheck

memory = Memory(cache_dir / "ocr_cache", verbose=0)

ocr_engine = OCR_with_format()

@optional_typecheck
@trace
@smartcache
@memory.cache
def get_text(
    img: str,
    **extra_kwargs,
    ) -> str:
    yel(f"Starting OCR for image '{img}'")
    default_args = {
        "img_path": img,
    }
    if "VOICE2ANKI_DEFAULT_OCR_LANGUAGE" in os.environ:
        default_args["language"] = os.environ["VOICE2ANKI_DEFAULT_OCR_LANGUAGE"]
    default_args.update(extra_kwargs)
    return ocr_engine.OCR(**default_args)
