import time
import os
import tempfile
from pathlib import Path, PosixPath
from threading import Lock
import gradio as gr
import pandas as pd
import litellm
from typing import Optional, Any, Callable, List

from .typechecker import optional_typecheck

# used to print in red
col_red = "\033[91m"
col_rst = "\033[0m"

class DF(pd.DataFrame):
    """Pandas DataFrame but forbid new column creation
    This is to make sure no bugs happened silently
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __setitem__(self, key, value):
        if key in self.columns:
            super().__setitem__(key, value)
        else:
            raise ValueError(f"Adding new columns is not allowed: key={key} value={value}")


@optional_typecheck
class SharedModule:
    """module used to store information from Voice2Anki.py to
    the main .py files"""
    # things that are not changed when self.reset is called
    VERSION: str = "2.0"
    _instance: Optional[Any] = None  # singleton
    anki_media: Optional[PosixPath] = None
    debug: Optional[bool] = None
    disable_tracing: Optional[bool] = None
    disable_timeout: Optional[bool] = None
    widen_screen: Optional[bool] = None
    big_font: Optional[bool] = None
    timeout_lock: Lock = Lock()
    dirload_lock: Lock = Lock()
    thread_lock: Lock = Lock()
    db_lock: Lock = Lock()
    openai_client: Optional[Any] = None
    user_chains: Optional[Callable] = None
    anki_notetype: Optional[str] = None

    stt_models: List[str] = ["openai:whisper-1", "replicate:vaibhavs10/incredibly-fast-whisper"]

    llm_price: dict = {k: v for k, v in litellm.model_cost.items()}

    # embeddings are so cheap I don't even count the number of tokens
    embedding_models: List[str] = [
            "openai/text-embedding-3-large",
            "openai/text-embedding-3-small",
            "mistral/mistral-embed",
            ]

    audio_backends: List[str] = ["openai/whisper-1", "deepgram/whisper-v3"]

    # sox effect when loading a sound
    preprocess_sox_effects: List[str] = [
            # normalize audio
            ["norm"],

            # isolate voice frequency
            # -2 is for a steeper filtering
            # ["highpass", "-1", "100"],
            # ["lowpass", "-1", "3000"],
            # # removes high frequency and very low ones
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],

            # max silence should be 1s
            ["silence", "-l", "1", "0", "0.5%", "-1", "1.0", "0.5%"],

            # # remove leading silence
            # ["vad", "-p", "0.2", "-t", "5"],
            # # and ending silence, this might be unecessary for splitted audio
            # ["reverse"],
            # ["vad", "-p", "0.2", "-t", "5"],
            # ["reverse"],

            # add blank sound to help whisper
            ["pad", "0.2@0"],
            ]

    # sox effect when forcing the processing of a sound
    force_preprocess_sox_effects: List[str] = [
            # normalize audio
            ["norm"],

            # filter for voice
            ["highpass", "-2", "50"],
            ["lowpass", "-2", "5000"],

            # max silence should be 1s
            ["silence", "-l", "1", "0", "2%", "-1", "1.0", "2%"],

            # # remove leading silence
            # ["vad", "-p", "0.2", "-t", "5"],
            # # and ending silence, this might be unecessary for splitted audio
            # ["reverse"],
            # ["vad", "-p", "0.2", "-t", "5"],
            # ["reverse"],

            # add blank sound to help whisper
            ["pad", "0.2@0"],
            ]

    # sox effects when splitting long audio
    splitter_sox_effects: List[str] = [
            # ["norm"],  # normalize audio

            # isolate voice frequency
            # ["highpass", "-1", "100"],
            # ["lowpass", "-1", "3000"],
            # -2 is for a steeper filtering: removes high frequency and very low ones
            # ["highpass", "-2", "50"],
            # ["lowpass", "-2", "5000"],

            ["norm"],  # normalize audio

            # max silence should be 3s
            ["silence", "-l", "1", "0", "1%", "-1", "3.0", "1%"],

            ["norm"],
            ]

    max_message_buffer: int = 20

    audio_slot_nb: Optional[int] = None
    queued_gallery_slot_nb: int = 99

    dirload_queue_columns: List[str] = [
            "n",
            "path",
            "temp_path",
            "loaded",
            "sound_preprocessed",
            "transcribed",
            "alfreded",
            "ankified",
            "moved",
            ]

    # things that are reset on self.reset
    pv: Optional = None  # ValueStorage, for 'previous values'
    initialized: int = 0
    request = None

    tmp_dir: PosixPath = Path(tempfile.NamedTemporaryFile().name).parent
    splitted_dir: Optional[PosixPath] = None
    done_dir: Optional[PosixPath] = None
    unsplitted_dir: Optional[PosixPath] = None
    func_dir: Optional[PosixPath] = None

    message_buffer: List[dict] = []
    dirload_queue: DF = DF(columns=dirload_queue_columns).set_index("path")

    llm_to_db_buffer: dict = {}

    running_threads: dict = {
            "saving_chatgpt": [],
            "saving_whisper": [],
            "transcribing_audio": [],
            "add_audio_to_anki": [],
            "ocr": [],
            "timeout": [],
            }
    smartcache: dict = {}

    added_note_ids: List[int] = []

    def __new__(cls):
        "make sure the instance will be a singleton"
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise Exception("Tried to create another instance of SharedModule")

    def reset(self, request: gr.Request) -> None:
        "used to reset the values when the gradio page is reloaded"
        self.dirload_queue = DF(columns=self.dirload_queue_columns).set_index("path")
        self.llm_to_db_buffer = {}
        for k in self.running_threads:
            self.running_threads[k] = []
        self.smartcache = {}
        self.added_note_ids = []
        self.pv.running_tasks = {k: None for k in self.pv.running_tasks.keys()}
        self.pv.cache_values = {k: None for k in self.pv.cache_values.keys()}

        os.environ["OPENAI_API_KEY"] = self.pv["txt_openai_api_key"].strip()
        os.environ["REPLICATE_API_KEY"] = self.pv["txt_replicate_api_key"].strip()
        os.environ["MISTRAL_API_KEY"] = self.pv["txt_mistral_api_key"].strip()
        os.environ["OPENROUTER_API_KEY"] = self.pv["txt_openrouter_api_key"].strip()

        self.request = {
                "user-agent": request.headers["user-agent"],
                "headers": dict(request.headers),
                "IP adress": f"{request.client.host}:{request.client.port}",
                "query_params": request.query_params._dict,
                }

        self.splitted_dir = Path("profiles/" + self.pv.profile_name + "/queues/audio_splits")
        self.done_dir = Path("profiles/" + self.pv.profile_name + "/queues/audio_done")
        self.unsplitted_dir = Path("profiles/" + self.pv.profile_name + "/queues/audio_untouched")
        self.splitted_dir.parent.mkdir(exist_ok=True)  # create queues
        (self.splitted_dir.parent / "galleries").mkdir(exist_ok=True)  # create galleries
        for dirs in [self.splitted_dir, self.done_dir, self.unsplitted_dir]:
            if not dirs.exists():
                p(f"Created directory {dirs}")
                dirs.mkdir()
        self.func_dir = Path("profiles/" + self.pv.profile_name + "/functions")
        self.func_dir.mkdir(exist_ok=True)
        self.user_chains = None
        self.anki_notetype = None

        self.initialized += 1
        if self.initialized > 1:
            p(f"Shared module initialized {self.initialized} times.")

    def __setattr__(self, name: str, value) -> None:
        "forbid creation of new attributes."
        if hasattr(self, name):

            # if the object is a df, make sure it replaces a df and
            # contains the same columns
            if isinstance(value, (pd.core.frame.DataFrame, DF)):
                assert isinstance(getattr(self, name), pd.core.frame.DataFrame), name
                assert sorted(value.columns.tolist()) == sorted(getattr(self, name).columns.tolist()), f"The new df has different columns: {name}"
            object.__setattr__(self, name, value)
        else:
            raise TypeError(f'Cannot set name {name} on object of type {self.__class__.__name__}')


@optional_typecheck
def p(message: str) -> None:
    print(col_red + message + col_rst)
    gr.Warning(message)


shared = SharedModule()
