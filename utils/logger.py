import re
import os
from pathlib import Path
from tqdm import tqdm
import logging
from logging import handlers
import rtoml
import json

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


def coloured_log(color_asked):
    """used to print color coded logs"""
    col_red = "\033[91m"
    col_yel = "\033[93m"
    col_rst = "\033[0m"

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
        lastline = re.sub(log_regex, " >           ", lastline)
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
            line = re.sub(log_regex, " >           ", line)
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
