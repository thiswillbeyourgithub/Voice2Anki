from joblib import hash as jhash
import asyncio
import threading
import time
import sqlite3
import zlib
import re
import os
from pathlib import Path
from tqdm import tqdm
import logging
from logging import handlers
import rtoml
import json

try:
    from .shared_module import shared
except:
    # needed when calling audio_splitter instead of Voice2Anki
    from shared_module import shared

Path("utils/logs").mkdir(exist_ok=True)
log_file = Path("utils/logs/logs.txt")
log_file.touch(exist_ok=True)
log_formatter = logging.Formatter(
        fmt='%(asctime)s ##%(levelname)s %(funcName)s(%(lineno)d)## %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
file_handler = handlers.RotatingFileHandler(
        log_file,
        mode='a',
        maxBytes=1000000,
        backupCount=100,
        encoding=None,
        delay=0,
        )
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log_regex = re.compile(" ##.*?##")


def store_to_db(dictionnary, db_name):
    """
    take a dictionnary and add it to the sqlite db. This is used to store
    all interactions with LLM and can be used later to create a dataset for
    finetuning.
    """

    Path("databases").mkdir(exist_ok=True)
    data = zlib.compress(
            json.dumps(dictionnary, ensure_ascii=False).encode(),
            level=9,  # 1: fast but large, 9 slow but small
            )
    with shared.db_lock:
        conn = sqlite3.connect(f"./databases/{db_name}.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS dictionaries
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          data TEXT)''')
        cursor.execute("INSERT INTO dictionaries (data) VALUES (?)", (data,))
        conn.commit()
        conn.close()
    return True


def print_db(db_filename):
    Path("databases").mkdir(exist_ok=True)
    assert Path(f"./databases/{db_filename}").exists(), (
        f"db not found: '{db_filename}'")
    conn = sqlite3.connect(f"./databases/{db_filename}")
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM dictionaries")
    rows = cursor.fetchall()
    dictionaries = []
    for row in rows:
        dictionary = json.loads(zlib.decompress(row[0]))
        dictionaries.append(dictionary)
    return json.dumps(dictionaries, ensure_ascii=False, indent=4)


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"
    col_prpl = "\033[95m"

    # all logs are considered "errors" otherwise the datascience libs just
    # overwhelm the logs

    if color_asked == "white":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_rst + string + col_rst, **args)
            return string
    elif color_asked == "yellow":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_yel + string + col_rst, **args)
            return string
    elif color_asked == "red":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_red + string + col_rst, **args)
            return string
    elif color_asked == "purple":
        def printer(string, **args):
            if isinstance(string, dict):
                try:
                    string = rtoml.dumps(string)
                except Exception:
                    string = json.dumps(string)
            if isinstance(string, list):
                string = ",".join(string)
            string = str(string)
            log.info(string)
            tqdm.write(col_prpl + string + col_rst, **args)
            return string
    return printer

def get_log():
    "frequently called: read the most recent log entries and display it in the output field"
    global last_log_content, latest_tail
    logcontent = []
    # updates only if the last line has changed
    with open(str(log_file), "rb") as f:
        # source: https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        lastline = f.readline().decode().strip()
        lastline = re.sub(log_regex, " >        ", lastline)[11:]
        if last_log_content and (lastline[23:] == latest_tail[23:] or "HTTP Request: POST" in lastline):
            return last_log_content

    latest_tail = lastline
    with open(str(log_file), "r") as f:
        for line in f.readlines()[-100:]:
            line = line.strip()
            if "HTTP Request: POST" in line:
                continue
            if not line:
                continue
            line = re.sub(log_regex, " >        ", line)[11:]
            logcontent.append(line)
    if not logcontent:
        return "Empty log"
    logcontent.reverse()
    last_log_content = "\n".join(logcontent)
    return last_log_content

latest_tail = None
last_log_content = None


whi = coloured_log("white")
yel = coloured_log("yellow")
red = coloured_log("red")
purp = coloured_log("purple")

def trace(func):
    """simple wrapper to use as decorator to print when a function is used
    and for how long"""
    if shared.disable_tracing:
        return func
    if asyncio.iscoroutinefunction(func):
        async def wrapper(*args, **kwargs):
            purp(f"-> Entering {func}")
            t = time.time()
            result = await func(*args, **kwargs)
            tt = time.time() - t
            if tt > 0.5:
                red(f"    Exiting {func} after {tt:.1f}s")
            else:
                purp(f"   Exiting {func} after {tt:.1f}s")
            return result
    else:
        def wrapper(*args, **kwargs):
            purp(f"-> Entering {func}")
            t = time.time()
            result = func(*args, **kwargs)
            tt = time.time() - t
            if tt > 0.5:
                red(f"    Exiting {func} after {tt:.1f}s")
            else:
                purp(f"   Exiting {func} after {tt:.1f}s")
            return result
    return wrapper


def Timeout(limit):
    """wrapper to add a timeout to function. I had to use threading because
    signal could not be used outside of the main thread in gradio"""
    if shared.disable_timeout:
        def decorator(func):
            return func
        return decorator

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                async with asyncio.timeout(limit):
                    return await func(*args, **kwargs)
        else:
            def wrapper(*args, **kwargs):
                # return func(*args, **kwargs)  # for debugging
                result = []
                def appender(func, *args, **kwargs):
                    result.append(func(*args, **kwargs))
                thread = threading.Thread(
                        target=appender,
                        args=[func] + list(args),
                        kwargs=kwargs,
                        daemon=False,
                        )
                thread.start()

                # add the thread in the shared module, this way we can empty
                # the list to cut the timeout short
                with shared.timeout_lock:
                    shared.running_threads["timeout"].append(thread)

                start = time.time()
                while shared.running_threads["timeout"] and thread.is_alive():
                    time.sleep(0.1)
                    if time.time() - start > limit:
                        raise Exception(f"Reached timeout for {func} after {limit}s")
                if not shared.running_threads["timeout"]:
                    raise Exception(f"Thread of func {func} was killed")

                if not result:  # meaning an exception occured in the function
                    raise Exception(f"No result from {func} with args {args} {kwargs}")
                else:
                    return result[0]
        return wrapper
    return decorator

def smartcache(func):
    """used to decorate a function that is already decorated by a
    joblib.Memory decorator. It stores the hash of the arguments in
    shared.smartcache at the start of the run and removes it at the end.
    If it already exists that means the cache is already computing the same
    value so just wait for that to finish to avoid concurrent calls."""
    def wrapper(*args, **kwargs):
        h = jhash(jhash(args) + jhash(kwargs))
        if h in shared.smartcache:
            t = shared.smartcache[h]
            red(f"Cache already ongoing for {func}. Hash={h}")
            i = 0
            while h in shared.smartcache:
                time.sleep(0.1)
                i += 1
                if i % 10 == 0:
                    delay = time.time() - t
                    red(f"Waiting for {func} caching to finish for {delay:02f}s. Hash={h}")
            return func(*args, **kwargs)
        else:
            with shared.thread_lock:
                with shared.timeout_lock:
                    shared.smartcache[h] = time.time()
            try:
                result = func(*args, **kwargs)
            except:
                with shared.thread_lock:
                    with shared.timeout_lock:
                        del shared.smartcache[h]
                raise
            with shared.thread_lock:
                with shared.timeout_lock:
                    del shared.smartcache[h]
            return result
    return wrapper
