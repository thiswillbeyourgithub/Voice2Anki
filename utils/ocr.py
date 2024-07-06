from joblib import Memory
from OCR_with_format import OCR_with_format

from .logger import yel, trace, smartcache

memory = Memory("cache/ocr_cache", verbose=0)

ocr_engine = OCR_with_format()

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
    default_args.update(extra_kwargs)
    return ocr_engine.OCR(**default_args)
