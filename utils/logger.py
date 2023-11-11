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

from .shared_module import shared

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
        backupCount=4,
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
    conn = sqlite3.connect(f"./databases/{db_name}.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS dictionaries
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      data TEXT)''')

    data = zlib.compress(
            json.dumps(dictionnary).encode(),
            level=9,  # 1: fast but large, 9 slow but small
            )
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
    return json.dumps(dictionaries, indent=4)


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

indent = -2

def trace(func):
    """simple wrapper to use as decorator to print when a function is used
    and for how long"""
    def wrapper(*args, **kwargs):
        global indent
        indent += 2
        spacer = indent * ' '
        purp(f"{spacer}-> Entering {func}")
        t = time.time()
        result = func(*args, **kwargs)
        tt = time.time() - t
        if tt > 0.5:
            red(f"{spacer}   Exiting {func} after {tt:.1f}s")
        else:
            purp(f"{spacer}   Exiting {func} after {tt:.1f}s")
        indent -= 2
        return result
    return wrapper


def Timeout(limit):
    """wrapper to add a timeout to function. I had to use threading because
    signal could not be used outside of the main thread in gradio"""
    # create methods like "t30"  to wrap with a 30s timeout
    def decorator(func):
        def wrapper(*args, **kwargs):
            # return func(*args, **kwargs)  # for debugging
            result = []
            def appender(func, *args, **kwargs):
                result.append(func(*args, **kwargs))
            thread = threading.Thread(
                    target=appender,
                    args=[func] + list(args),
                    kwargs=kwargs,
                    daemon=True,
                    )
            thread.start()

            # add the thread in the shared module, this way we can empty
            # the list to cut the timeout short
            with threading.Lock():
                shared.threads.append(thread)

            start = time.time()
            while shared.threads and thread.is_alive():
                time.sleep(0.1)
                if time.time() - start > limit:
                    raise Exception(f"Reached timeout for {func} after {limit}s")
            if not shared.threads:
                raise Exception(f"Thread of func {func} was killer")

            if not result:  # meaning an exception occured in the function
                raise Exception(f"No result from {func} with args {args} {kwargs}")
            else:
                return result[0]
        return wrapper
    return decorator
